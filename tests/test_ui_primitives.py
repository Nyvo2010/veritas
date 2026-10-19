"""Tests for the small pieces the whole UI leans on.

Run with: python -m unittest discover -s tests
"""

import json
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ui import theme  # noqa: E402
from ui import color_compat  # noqa: E402
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



class BrailleWaveTests(unittest.TestCase):
    """The two-row wave is the progress bar on both frontends."""

    def _parts(self, ratio, width, phase=0.0, particles=0):
        (fill_top, fill_bottom), (track_top, track_bottom) = theme.braille_wave(
            ratio, width, phase=phase, particles=particles
        )
        return fill_top, fill_bottom, track_top, track_bottom

    def test_each_line_is_exactly_width_cells(self):
        for width in (8, 24, 40, 97):
            for ratio in (0.0, 0.13, 0.5, 0.87, 1.0):
                top, bottom, track_top, track_bottom = self._parts(
                    ratio, width, phase=3.0, particles=20
                )
                self.assertEqual(len(top) + len(track_top), width,
                                 f"ratio={ratio} width={width}")
                self.assertEqual(len(bottom) + len(track_bottom), width,
                                 f"ratio={ratio} width={width}")

    def test_every_character_is_braille(self):
        for row in self._parts(0.42, 60, phase=11.0, particles=24):
            for char in row:
                self.assertGreaterEqual(ord(char), BRAILLE_BLOCK)
                self.assertLessEqual(ord(char), BRAILLE_BLOCK + 0xFF)

    def test_fill_grows_with_progress(self):
        previous = 0
        for step in range(11):
            top, _, _, _ = self._parts(step / 10, 40)
            filled = sum(1 for char in top if ord(char) != BRAILLE_BLOCK)
            self.assertGreaterEqual(filled, previous)
            previous = filled
        top, bottom, track_top, track_bottom = self._parts(1.0, 40)
        self.assertEqual(len(top), 40)
        self.assertEqual(track_top + track_bottom, "")

    def test_the_surface_ripples(self):
        top, _, _, _ = self._parts(1.0, 60, phase=4.0)
        self.assertGreater(len(set(top)), 1, "a flat top would be a single mask")

    def test_ratio_is_clamped(self):
        blank = chr(BRAILLE_BLOCK)
        top, bottom, track_top, track_bottom = self._parts(-5, 12)
        self.assertEqual(top + track_top, blank * 12)
        self.assertEqual(bottom + track_bottom, blank * 12)
        top, _, track_top, track_bottom = self._parts(9.9, 12)
        self.assertEqual(track_top + track_bottom, "")
        self.assertNotEqual(set(top), {blank})

    def test_particles_never_change_the_filled_part(self):
        for phase in range(0, 400, 7):
            plain = self._parts(0.5, 40, phase=phase, particles=0)
            busy = self._parts(0.5, 40, phase=phase, particles=36)
            self.assertEqual(plain[0], busy[0], "particles leaked into the fill")
            self.assertEqual(plain[1], busy[1], "particles leaked into the fill")

    def test_particles_add_dots_to_the_track(self):
        blank = chr(BRAILLE_BLOCK)
        _, _, plain_top, plain_bottom = self._parts(0.0, 60, phase=0.0)
        _, _, dotted_top, dotted_bottom = self._parts(0.0, 60, phase=5.0, particles=20)
        self.assertEqual(set(plain_top + plain_bottom), {blank})
        self.assertTrue(any(c != blank for c in dotted_top + dotted_bottom))

    def test_more_particles_fill_more_of_the_track(self):
        def dots(count):
            _, _, top, bottom = self._parts(0.0, 60, phase=9.0, particles=count)
            return sum(1 for c in top + bottom if ord(c) != BRAILLE_BLOCK)

        self.assertGreater(dots(36), dots(9))

    def test_particles_move_with_the_phase(self):
        frames = [self._parts(0.0, 40, phase=step, particles=12)[2:] for step in range(6)]
        self.assertGreater(len(set(frames)), 1, "the bar must animate")


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
