import os
import unittest

import tell.data_process_population as dpp


class TestExecuteFor(unittest.TestCase):
    """Tests for functionality within execue forward.py"""

    # supporting data
    TEST_POP_DIR = os.path.join(os.path.dirname(__file__), 'data/')

    def test_fips_pop_yearly(self):
        """Test to ensure high level functionality of normalize_prediction_data()"""

        # interpolate pop data from hourly to annual
        pop_df = dpp.fips_pop_yearly(pop_input_dir=TestExecuteFor.TEST_POP_DIR,
                                        start_year=2000,
                                        end_year=2010)

        # check that length is as expected
        self.assertEqual(209, len(pop_df))

        # check that length is as expected
        self.assertEqual(3, pop_df.shape[1])

        # ensure all strings have a length greater than 0
        self.assertEqual(True, all(len(i) > 0 for i in hourly_df['county_fips']))


if __name__ == '__main__':
    unittest.main()