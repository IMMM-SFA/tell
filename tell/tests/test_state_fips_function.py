import unittest

from tell.states_fips_function import state_metadata_from_state_abbreviation


class TestStateFIPS(unittest.TestCase):
    """Tests for functionality within state_fips_function.py"""

    def test_state_fips(self):
        """Test to ensure high level functionality of state_fips_function.py()"""

        state_fips, state_name = state_metadata_from_state_abbreviation("WA")

        self.assertEqual(53000, state_fips)
        self.assertEqual("Washington", state_name)

        with self.assertRaises(KeyError):
            state_fips, state_name = state_metadata_from_state_abbreviation("XX")


if __name__ == '__main__':
    unittest.main()
