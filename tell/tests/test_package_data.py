import unittest

from tell.package_data import get_ba_abbreviations


class TestPackageData(unittest.TestCase):
    """Tests for functionality within package_data.py"""

    def test_get_ba_abbreviations(self):
        """Test to ensure high level functionality of get_ba_abbreviations()"""

        ba_names = get_ba_abbreviations()

        # check length of list
        self.assertEqual(68, len(ba_names))

        # ensure all values in list are strings
        self.assertEqual(True, all(isinstance(i, str) for i in ba_names))

        # ensure all strings have a length greater than 0
        self.assertEqual(True, all(len(i) > 0 for i in ba_names))


if __name__ == '__main__':
    unittest.main()
