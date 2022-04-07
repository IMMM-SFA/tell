import unittest

from tell.states_fips_function import state_metadata_from_state_abbreviation


class TestStateFIPS(unittest.TestCase):
    """Tests for functionality within state_fips_function.py"""

    def test_state_fips(self):
        """Test to ensure high level functionality of state_fips_function.py()"""

        num = ['WA', 'VA', 'NC', 'RI']
        state_fips_df = state_metadata_from_state_abbreviation(num)

        # check length of list
        self.assertEqual(4, len(state_fips_df))

        # ensure BA name column is object in df are object
        self.assertEqual(True, all(isinstance(i, object) for i in state_fips_df['state_name']))


if __name__ == '__main__':
    unittest.main()
