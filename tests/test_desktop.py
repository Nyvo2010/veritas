"""Desktop helpers: correct command per platform and a clear missing-file error."""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from utils import desktop  # noqa: E402


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="veritas-desktop-")
        self.report = os.path.join(self.tmp, "report.md")
        with open(self.report, "w", encoding="utf-8") as handle:
            handle.write("body")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @mock.patch.object(desktop.subprocess, "Popen")
    def test_open_uses_open_on_macos(self, popen):
        with mock.patch.object(desktop.sys, "platform", "darwin"):
            path = desktop.open_in_default_app(self.report)
        self.assertEqual(path, os.path.abspath(self.report))
        self.assertEqual(popen.call_args.args[0][:2], ["open", os.path.abspath(self.report)])

    @mock.patch.object(desktop.subprocess, "Popen")
    def test_reveal_selects_the_file_on_macos(self, popen):
        with mock.patch.object(desktop.sys, "platform", "darwin"):
            desktop.reveal_in_file_manager(self.report)
        self.assertEqual(
            popen.call_args.args[0],
            ["open", "-R", os.path.abspath(self.report)],
        )

    @mock.patch.object(desktop.subprocess, "Popen")
    def test_missing_report_is_reported(self, popen):
        with self.assertRaises(FileNotFoundError):
            desktop.open_in_default_app(os.path.join(self.tmp, "missing.md"))
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
