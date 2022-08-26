import os
import unittest

import numpy as np

import tell.execute_forward as ef


class TestExecuteFor(unittest.TestCase):
    """Tests for functionality within execue forward.py"""

    # supporting data
    TEST_POP = os.path.join(os.path.dirname(__file__), 'data/')

    def test_process_population_scenario(self):
        """Test to ensure high level functionality of normalize_prediction_data()"""

        # interpolate pop data from hourly to annual
        hourly_df = ef.process_population_scenario(scenario_to_process = 'rcp45hotter_ssp3',
                                                   population_data_input_dir = TestExecuteFor.TEST_POP)

        # check that length is as expected
        self.assertEqual(251748, len(hourly_df))

        # check that length is as expected
        self.assertEqual(3, hourly_df.shape[1])

        # ensure all strings have a length greater than 0
        self.assertEqual(True, all(len(i) > 0 for i in hourly_df['Year']))


if __name__ == '__main__':
    unittest.main()
