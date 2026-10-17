"""Tests for the small pieces the whole UI leans on.

Run with: python -m unittest discover -s tests
"""

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ui import theme  # noqa: E402
from ui.timings import Timings  # noqa: E402
from utils import text  # noqa: E402

BRAILLE_BLOCK = 0x2800


class BrailleBarTests(unittest.TestCase):
    def test_width_is_exact_suffix_and_all_cells(self):
        for width in (8, 24, 40, 97):
            for ratio in (0.0, 0.13, 0.5, 0.87, 1.0):
                filled, track = theme.braille_bar(ratio, width, phase=3.0, particles=4)
                self.assertEqual(
                    len(filled) + len(track), width, f"ratio={ratio} width={width}"
                )

    def test_every_character_is_braille(self):
        filled, track = theme.braille_bar(0.42, 60, phase=11.0, particles=6)
        for char in filled + track:
            self.assertGreaterEqual(ord(char), BRAILLE_BLOCK)
            self.assertLessEqual(ord(char), BRAILLE_BLOCK + 0xFF)

    def test_fill_grows_with_progress(self):
        previous = 0
        for step in range(11):
            filled, _ = theme.braille_bar(step / 10, 40, phase=0.0, particles=0)
            self.assertGreaterEqual(len(filled), previous)
            previous = len(filled)
        filled, track = theme.braille_bar(1.0, 40, phase=0.0, particles=0)
        self.assertEqual(len(filled), 40)
        self.assertEqual(track, "")

    def test_ratio_is_clamped(self):
        self.assertEqual(theme.braille_bar(-5, 12)[0], "")
        filled, track = theme.braille_bar(9.9, 12)
        self.assertEqual(len(filled), 12)
        self.assertEqual(track, "")

    def test_particles_never_enter_the_filled_part(self):
        for phase in range(0, 400, 7):
            filled, track = theme.braille_bar(0.5, 40, phase=phase, particles=6)
            self.assertEqual(len(filled), 20)
            self.assertEqual(len(track), 20)

    def test_track_gains_particle_dots(self):
        baseline = chr(BRAILLE_BLOCK + 0x24)   # bottom row only
        _, plain = theme.braille_bar(0.0, 60, phase=0.0, particles=0)
        _, dotted = theme.braille_bar(0.0, 60, phase=5.0, particles=8)
        self.assertEqual(set(plain), {baseline})
        self.assertTrue(any(ch != baseline for ch in dotted), "particles add dots")

    def test_particles_move_with_the_phase(self):
        frames = [theme.braille_bar(0.0, 40, phase=step, particles=3)[1] for step in range(6)]
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
