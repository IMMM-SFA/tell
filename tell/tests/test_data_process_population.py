import os
import pkg_resources
import unittest

import tell.data_process_population as dpp


class TestProcessPop(unittest.TestCase):
    """Tests for functionality within execue forward.py"""

    # supporting data
    TEST_POP_DIR = pkg_resources.resource_filename("tell", os.path.join("tests", "data"))

    def test_fips_pop_yearly(self):
        """Test to ensure high level functionality of fips_pop_yearly()"""

        # interpolate pop data from hourly to annual
        pop_df = dpp.fips_pop_yearly(pop_input_dir=TestProcessPop.TEST_POP_DIR,
                                     start_year=2000,
                                     end_year=2010)

        # check the shape of the data frame is expected
        self.assertEqual((209, 3), pop_df.shape)

        # ensure all strings have a length greater than 0
        self.assertFalse(pop_df["county_FIPS"].isna().all())


if __name__ == '__main__':
    unittest.main()
