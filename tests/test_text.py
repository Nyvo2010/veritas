"""Coercion helpers for model-supplied values.

A model asked for a list can answer with one string, and a model asked for a
boolean can answer with ``"false"``. Both would behave wrongly in Python if
passed through, so they are normalised at the boundary.

Run with: python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from utils.text import as_bool, as_int, string_list  # noqa: E402


class AsBoolTests(unittest.TestCase):
    def test_the_string_false_is_false(self):
        self.assertFalse(as_bool("false"))
        self.assertFalse(as_bool(" FALSE "))
        self.assertFalse(as_bool("no"))
        self.assertFalse(as_bool("0"))

    def test_true_forms_are_true(self):
        for value in (True, "true", "yes", "1", 1):
            with self.subTest(value=value):
                self.assertTrue(as_bool(value))

    def test_missing_or_unknown_values_use_the_default(self):
        self.assertTrue(as_bool(None, default=True))
        self.assertFalse(as_bool(None))
        self.assertTrue(as_bool("maybe", default=True))


class AsIntTests(unittest.TestCase):
    def test_numbers_and_numeric_strings_are_accepted(self):
        self.assertEqual(as_int("8"), 8)
        self.assertEqual(as_int(7.9), 7)
        self.assertEqual(as_int(3), 3)

    def test_junk_falls_back_to_the_default(self):
        self.assertEqual(as_int("urgent", default=5), 5)
        self.assertEqual(as_int(None, default=5), 5)
        self.assertEqual(as_int([], default=5), 5)

    def test_the_result_is_clamped(self):
        self.assertEqual(as_int(90, low=0, high=10), 10)
        self.assertEqual(as_int(-4, low=0, high=10), 0)


class StringListTests(unittest.TestCase):
    def test_a_bare_string_is_one_item_not_characters(self):
        self.assertEqual(string_list("What next?"), ["What next?"])

    def test_lists_are_trimmed_and_empties_dropped(self):
        self.assertEqual(
            string_list([" a ", "", None, "b"]), ["a", "b"]
        )

    def test_missing_values_are_empty(self):
        self.assertEqual(string_list(None), [])
        self.assertEqual(string_list([]), [])

    def test_limit_is_applied_after_cleaning(self):
        self.assertEqual(string_list(["a", "", "b", "c"], limit=2), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
