"""Settings persistence: the file holds API keys, so it is private and atomic.

Run with: python -m unittest discover -s tests
"""

import os
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import settings  # noqa: E402


class SaveSettingsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="veritas-settings-")
        self.addCleanup(shutil.rmtree, self.directory, True)
        patcher = mock.patch.object(settings, "settings_dir", return_value=self.directory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_values_round_trip(self):
        settings.save_settings({**settings.DEFAULTS, "llm_api_key": "secret-key"})

        self.assertEqual(settings.load_settings()["llm_api_key"], "secret-key")

    @unittest.skipIf(os.name == "nt", "POSIX permissions only")
    def test_the_file_is_owner_only(self):
        path = settings.save_settings({**settings.DEFAULTS, "llm_api_key": "secret-key"})

        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)

    def test_no_temporary_file_is_left_behind(self):
        settings.save_settings(dict(settings.DEFAULTS))

        leftovers = [name for name in os.listdir(self.directory) if name.startswith(".settings-")]
        self.assertEqual(leftovers, [])

    def test_unknown_keys_are_not_written(self):
        settings.save_settings({**settings.DEFAULTS, "unexpected": "value"})

        self.assertNotIn("unexpected", settings.load_settings())


if __name__ == "__main__":
    unittest.main()
