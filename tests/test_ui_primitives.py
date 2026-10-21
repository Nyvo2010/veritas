"""Tests for the small pieces the whole UI leans on.

Run with: python -m unittest discover -s tests
"""

import io
import json
import os
import sys
import threading
import unittest

from rich.console import Console

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ui import theme  # noqa: E402
from ui import color_compat  # noqa: E402
from ui.live_renderer import LiveRenderer  # noqa: E402
from ui.timings import Timings  # noqa: E402
from utils import text  # noqa: E402
from utils import llm_client  # noqa: E402

BRAILLE_BLOCK = 0x2800


class ColorCompatTests(unittest.TestCase):
    """The pty proxy rewrites truecolor SGR runs for 256-colour terminals.

    Reads are arbitrary-sized, so an escape can be split across two chunks;
    the downconverter must never emit a partial sequence or miss a colour.
    """

    def _stream(self, chunks):
        tail = b""
        out = bytearray()
        for chunk in chunks:
            data, tail = color_compat.rewrite_truecolor(tail + chunk)
            out += data
        return bytes(out) + tail

    def test_rewrites_a_plain_run(self):
        out, tail = color_compat.rewrite_truecolor(b"\x1b[38;2;255;0;0mX")
        self.assertEqual(out, b"\x1b[38;5;196mX")
        self.assertEqual(tail, b"")

    def test_rewrites_truecolor_shared_with_attributes(self):
        out, _ = color_compat.rewrite_truecolor(b"\x1b[1;38;2;255;0;0m")
        self.assertEqual(out, b"\x1b[1;38;5;196m")
        out, _ = color_compat.rewrite_truecolor(b"\x1b[48;2;0;0;0;38;2;255;255;255m")
        self.assertEqual(out, b"\x1b[48;5;16;38;5;231m")

    def test_split_escapes_reassemble_the_same_as_whole(self):
        stream = b"\x1b[0m\x1b[38;2;12;34;56mA\x1b[?25l\x1b]0;title\x07\x1b[48;2;1;2;3mB"
        whole = self._stream([stream])
        for size in (1, 2, 3, 5, 7, 11):
            chunks = [stream[i:i + size] for i in range(0, len(stream), size)]
            self.assertEqual(self._stream(chunks), whole, f"chunk size {size}")

    def test_incomplete_trailing_escape_is_buffered(self):
        out, tail = color_compat.rewrite_truecolor(b"hi\x1b[38;2;12")
        self.assertEqual(out, b"hi")
        self.assertEqual(tail, b"\x1b[38;2;12")
        # resubmitted with the continuation it becomes a complete run
        out, tail = color_compat.rewrite_truecolor(tail + b";34;56m!")
        self.assertEqual(out, b"\x1b[38;5;17m!")
        self.assertEqual(tail, b"")

    def test_incomplete_osc_is_buffered(self):
        out, tail = color_compat.rewrite_truecolor(b"\x1b]66;w=1; ")
        self.assertEqual(out, b"")
        self.assertEqual(tail, b"\x1b]66;w=1; ")

    def test_complete_mode_sequences_are_not_held_back(self):
        out, tail = color_compat.rewrite_truecolor(b"\x1b[?25l\x1b[2J\x1b[H")
        self.assertEqual(out, b"\x1b[?25l\x1b[2J\x1b[H")
        self.assertEqual(tail, b"")

    def test_plain_text_is_untouched(self):
        out, tail = color_compat.rewrite_truecolor(b"no escapes here;2;still plain")
        self.assertEqual(out, b"no escapes here;2;still plain")
        self.assertEqual(tail, b"")


class JsonSalvageTests(unittest.TestCase):
    """A reply cut off at max_tokens must still yield the data it managed."""

    def test_complete_json(self):
        self.assertEqual(llm_client._extract_json('{"a": 1}'), {"a": 1})

    def test_strips_code_fences_and_prose(self):
        self.assertEqual(llm_client._extract_json('Sure:\n```json\n{"x": 2}\n```'), {"x": 2})

    def test_salvages_a_string_cut_mid_sentence(self):
        parsed = llm_client._extract_json('{"observation": "sources say", "analysis": "cut off here')
        self.assertEqual(parsed["observation"], "sources say")
        self.assertIn("cut off here", parsed["analysis"])

    def test_salvages_a_truncated_list(self):
        self.assertEqual(llm_client._extract_json('{"gaps": ["a", "b"'), {"gaps": ["a", "b"]})

    def test_drops_a_truncated_trailing_value(self):
        self.assertEqual(llm_client._extract_json('{"confidence": 0.8, "answer": tr'), {"confidence": 0.8})

    def test_detects_truncated_errors(self):
        try:
            raise json.JSONDecodeError("Unterminated string starting at", "x", 0)
        except json.JSONDecodeError as e:
            self.assertTrue(llm_client._looks_truncated(e, "x"))



class BrailleBarTests(unittest.TestCase):
    """The one-row progress bar: solid static fill, animated particle track."""

    SOLID = chr(BRAILLE_BLOCK + 0xFF)
    BLANK = chr(BRAILLE_BLOCK)

    def _parts(self, ratio, width, phase=0.0, particles=0):
        return theme.braille_bar(ratio, width, phase=phase, particles=particles)

    def test_fill_plus_track_is_exactly_width_cells(self):
        for width in (8, 24, 40, 97):
            for ratio in (0.0, 0.13, 0.5, 0.87, 1.0):
                fill, track = self._parts(ratio, width, phase=3.0, particles=20)
                self.assertEqual(len(fill) + len(track), width,
                                 f"ratio={ratio} width={width}")

    def test_every_character_is_braille(self):
        fill, track = self._parts(0.42, 60, phase=11.0, particles=24)
        for char in fill + track:
            self.assertGreaterEqual(ord(char), BRAILLE_BLOCK)
            self.assertLessEqual(ord(char), BRAILLE_BLOCK + 0xFF)

    def test_the_filled_part_is_solid_and_never_animates(self):
        fill, _ = self._parts(0.37, 40, phase=9.0, particles=20)
        self.assertEqual(fill, self.SOLID * 15, "fill must be one gap-free block")
        for phase in (0.0, 37.0, 400.0):
            self.assertEqual(self._parts(0.37, 40, phase=phase)[0], fill)

    def test_fill_grows_with_progress(self):
        previous = -1
        for step in range(11):
            fill, _ = self._parts(step / 10, 40)
            self.assertEqual(len(fill), 4 * step)
            self.assertGreaterEqual(len(fill), previous)
            previous = len(fill)
        self.assertEqual(self._parts(1.0, 40)[1], "")

    def test_ratio_is_clamped(self):
        fill, track = self._parts(-5, 12)
        self.assertEqual((fill, track), ("", self.BLANK * 12))
        fill, track = self._parts(9.9, 12)
        self.assertEqual((fill, track), (self.SOLID * 12, ""))

    def test_particles_add_dots_to_the_track(self):
        self.assertEqual(self._parts(0.0, 60, phase=0.0)[1], self.BLANK * 60)
        self.assertNotEqual(self._parts(0.0, 60, phase=5.0, particles=20)[1],
                            self.BLANK * 60)

    def test_particles_never_touch_the_fill(self):
        for phase in range(0, 400, 7):
            self.assertEqual(self._parts(0.5, 40, phase=phase)[0],
                             self._parts(0.5, 40, phase=phase, particles=36)[0])

    def test_more_particles_fill_more_of_the_track(self):
        def dots(count):
            track = self._parts(0.0, 60, phase=9.0, particles=count)[1]
            return sum(1 for c in track if ord(c) != BRAILLE_BLOCK)

        self.assertGreater(dots(36), dots(9))

    def test_particles_move_with_the_phase(self):
        frames = [self._parts(0.0, 40, phase=step, particles=12)[1] for step in range(6)]
        self.assertGreater(len(set(frames)), 1, "the track must animate")


class TextHelperTests(unittest.TestCase):
    def test_clip_flattens_and_caps(self):
        self.assertEqual(text.clip("a\n b\tc"), "a b c")
        clipped = text.clip("x" * 50, limit=10)
        self.assertEqual(len(clipped), 10)
        self.assertTrue(clipped.endswith("…"))

    def test_clip_handles_none(self):
        self.assertEqual(text.clip(None), "")
        self.assertEqual(text.clip(None, limit=0), "")

    def test_strip_markup_removes_rich_tags(self):
        self.assertEqual(text.strip_markup("[bold blue]hi[/bold blue]  there"), "hi there")

    def test_classify_message_levels(self):
        cases = {
            "Phase 2: Research and Analysis": "phase",
            "Searching: what is x": "search",
            "Key Insight: x": "insight",
            "Completed: theme one": "ok",
            "Error: boom": "err",
            "3 knowledge gaps found": "warn",
            "something else": "info",
        }
        for message, expected in cases.items():
            self.assertEqual(text.classify_message(message), expected, message)


class LiveRendererLayoutTests(unittest.TestCase):
    """The Rich CLI run screen must stay exactly as tall as its layout slots.

    A wrapped activity line grows the fixed-height log and pushes the progress
    bar down on narrow terminals, which is the jitter the renderer exists to
    avoid.
    """

    def _render(self, renderable, width=80):
        buffer = io.StringIO()
        console = Console(width=width, record=True, file=buffer, force_terminal=False)
        console.print(renderable)
        return console.export_text()

    def test_a_long_activity_line_is_clipped_to_the_panel(self):
        console = Console(width=80, record=True, file=io.StringIO(), force_terminal=False)
        renderer = LiveRenderer(console=console)
        renderer.log("ok", "Section ready in 17s: " + "Hardware acceleration " * 20)

        output = self._render(renderer._log_panel())

        for line in output.splitlines():
            self.assertLessEqual(len(line.rstrip()), 80, line)
        self.assertIn("…", output)

    def test_a_long_header_is_truncated_instead_of_wrapping(self):
        console = Console(width=80, record=True, file=io.StringIO(), force_terminal=False)
        renderer = LiveRenderer(console=console)
        renderer.title = "Research: " + "quantization quality per flop " * 6

        output = self._render(renderer._header_panel())

        self.assertEqual(len(output.strip("\n").splitlines()), 3, output)
        for line in output.splitlines():
            self.assertLessEqual(len(line.rstrip()), 80, line)


class TimingsTests(unittest.TestCase):
    def test_totals_and_counts(self):
        timings = Timings()
        timings.add("llm", 1.5)
        timings.add("llm", 2.5)
        timings.add("search", 4.0)
        snapshot = dict((kind, (seconds, calls)) for kind, seconds, calls in timings.snapshot())
        self.assertAlmostEqual(snapshot["llm"][0], 4.0)
        self.assertEqual(snapshot["llm"][1], 2)
        self.assertAlmostEqual(snapshot["search"][0], 4.0)
        self.assertEqual(snapshot["search"][1], 1)

    def test_bad_values_are_ignored(self):
        timings = Timings()
        timings.add("llm", "not a number")
        timings.add("llm", -3)
        self.assertEqual(dict((k, c) for k, _, c in timings.snapshot())["llm"], 1)

    def test_snapshot_is_a_comparable_tuple(self):
        timings = Timings()
        timings.add("llm", 1.0)
        first = timings.snapshot()
        self.assertEqual(first, timings.snapshot())
        timings.add("llm", 1.0)
        self.assertNotEqual(first, timings.snapshot())

    def test_thread_safety_under_concurrent_writes(self):
        timings = Timings()

        def hammer():
            for _ in range(200):
                timings.add("llm", 0.5)

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        entry = dict((k, (s, c)) for k, s, c in timings.snapshot())["llm"]
        self.assertEqual(entry[1], 800)
        self.assertAlmostEqual(entry[0], 400.0)


if __name__ == "__main__":
    unittest.main()
