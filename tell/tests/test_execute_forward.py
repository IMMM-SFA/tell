import os
import pkg_resources
import unittest

import tell


class TestExecuteFor(unittest.TestCase):
    """Tests for functionality within execue forward.py"""

    # supporting data
    TEST_POP_DIR = pkg_resources.resource_filename("tell", os.path.join("tests", "data"))

    def test_process_population_scenario(self):
        """Test to ensure high level functionality of process_population_scenario()"""

        # interpolate pop data from hourly to annual
        hourly_df = tell.process_population_scenario(scenario_to_process='rcp45hotter_ssp3',
                                                   population_data_input_dir=TestExecuteFor.TEST_POP_DIR)

        # check expected shape of dataframe
        self.assertEqual((251748, 3), hourly_df.shape)

        # ensure all strings have a length greater than 0
        self.assertFalse(hourly_df["Year"].isna().all())


if __name__ == '__main__':
    unittest.main()
