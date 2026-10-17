"""Headless render tests for every TUI screen.

These catch the failures that are painful to spot by hand: a panel that runs
off a short terminal, text that wraps and shifts the layout, a screen that
raises while being built. OpenTUI's own test renderer draws into a buffer we
can inspect.

Run with: python -m unittest discover -s tests
"""

import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    from opentui.testing import test_render

    HAVE_OPENTUI = True
except Exception:  # pragma: no cover - the TUI extra is optional
    HAVE_OPENTUI = False

if HAVE_OPENTUI:
    import tui_app

logging.getLogger("asyncio").setLevel(logging.WARNING)  # no "Executing <Task> took…"

SIZES = ((80, 24), (100, 40), (140, 50))
SAMPLE_LOG = (
    (0.0, "phase", "Phase 1: Query Analysis"),
    (1.2, "info", "Decomposing query into sub-questions"),
    (14.6, "ok", "Query decomposed into 6 questions (13.4s)"),
    (14.7, "phase", "Phase 2: Research and Analysis"),
    (36.4, "insight", "Insight: quantisation trades a little accuracy for a lot of speed"),
    (61.9, "err", "Search error: 429 rate limited by Tavily"),
)


@unittest.skipUnless(HAVE_OPENTUI, "opentui is not installed")
class ScreenRenderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # The frontend keeps its state in module-level signals; reset them so
        # one test cannot leak a query, a notice, or a log into the next.
        tui_app.log_sig.set(())
        tui_app.clear_notice()
        tui_app.stats_sig.set(())
        tui_app.prog_sig.set((0.0, 1, "Starting..."))
        tui_app.phase_sig.set("Preparing...")
        tui_app.state["query"] = ""

    async def render(self, screen, width, height, prepare=None):
        setup = await test_render(tui_app.App, {"width": width, "height": height})
        try:
            if prepare:
                prepare()
            tui_app.screen.set(screen)
            if prepare:
                prepare()
            return setup.capture_char_frame()
        finally:
            setup.destroy()

    async def test_every_screen_fits_the_terminal(self):
        for width, height in SIZES:
            for name in tui_app.SCREEN_BUILDERS:
                with self.subTest(screen=name, size=f"{width}x{height}"):
                    frame = await self.render(name, width, height)
                    lines = frame.split("\n")
                    self.assertLessEqual(len(lines), height, "drew past the bottom")
                    for line in lines:
                        self.assertLessEqual(len(line.rstrip()), width, "drew past the right edge")
                        self.assertNotIn("\t", line)

    async def test_every_screen_draws_something(self):
        for name in tui_app.SCREEN_BUILDERS:
            with self.subTest(screen=name):
                frame = await self.render(name, 100, 40)
                self.assertTrue(frame.strip(), "screen rendered empty")

    async def test_run_screen_shows_the_bar_and_a_heading(self):
        def prepare():
            tui_app.state["query"] = "How do on-device NPUs change mobile AI?"
            tui_app.log_sig.set(SAMPLE_LOG)
            tui_app.prog_sig.set((5.0, 28, "Iteration 2/10"))
            tui_app.phase_sig.set("Phase 2: Research and Analysis")
            tui_app.anim_sig.set(20)
            tui_app.stats_sig.set((("llm", 204.0, 14), ("search", 38.0, 6)))

        frame = await self.render("run", 100, 40, prepare)
        self.assertIn("Research running", frame)
        self.assertIn("\u28ff", frame, "progress bar should use braille cells")
        self.assertIn("\u258e", frame, "phase entries should render as headings")
        self.assertIn("LLM", frame, "the time breakdown belongs on the run screen")
        self.assertIn("Phase 2: Research and Analysis", frame)

    async def test_run_screen_keeps_its_height_as_events_arrive(self):
        def prepare():
            tui_app.state["query"] = "x"
            tui_app.log_sig.set(SAMPLE_LOG[:1])
            tui_app.anim_sig.set(1)

        def used_rows(text):
            return len([line for line in text.split("\n") if line.strip()])

        first = await self.render("run", 100, 40, prepare)
        prepare()
        tui_app.log_sig.set(SAMPLE_LOG)
        second = await self.render("run", 100, 40, prepare)
        self.assertEqual(used_rows(first), used_rows(second), "the run screen must not resize mid-run")

    async def test_activity_log_screen_lists_the_last_run(self):
        def prepare():
            tui_app.state["query"] = "How do on-device NPUs change mobile AI?"
            tui_app.log_sig.set(SAMPLE_LOG)

        frame = await self.render("logs", 100, 40, prepare)
        self.assertIn("Activity log", frame)
        self.assertIn("Phase 2: Research and Analysis", frame)
        self.assertIn("rate limited", frame)

    async def test_menu_offers_the_log_only_when_there_is_one(self):
        without = await self.render("menu", 100, 40)
        self.assertNotIn("Last run log", without)

        def prepare():
            tui_app.log_sig.set(SAMPLE_LOG)

        with_log = await self.render("menu", 100, 40, prepare)
        self.assertIn("Last run log", with_log)

    async def test_startup_notice_does_not_move_the_panel(self):
        """"A notice must not push the blocks above it around."""
        plain = await self.render("settings", 100, 40)
        tui_app.set_notice("error", "Cannot use that folder")
        try:
            with_notice = await self.render("settings", 100, 40)
        finally:
            tui_app.clear_notice()

        def first_panel_row(frame):
            for index, line in enumerate(frame.split("\n")):
                if "Settings" in line:
                    return index
            return -1

        self.assertEqual(first_panel_row(plain), first_panel_row(with_notice))
        self.assertIn("Cannot use that folder", with_notice)


if __name__ == "__main__":
    unittest.main()
