import os
import unittest
import pandas as pd

import tell.data_spatial_mapping as dsm


class TestSpatialMap(unittest.TestCase):
    """Tests for functionality within execue forward.py"""

    # supporting data
    TEST_INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data/')

    TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data/tell_quickstarter_data/outputs'
                                                              '/ba_service_territory/')

    def test_map_ba_service_territory(self):
        """Test to ensure high level functionality of normalize_prediction_data()"""

        # create ba service terriory mapping file for 2017 saved to output dir
        dsm.map_ba_service_territory(start_year=2017,
                                     end_year=2017,
                                     data_input_dir=TestSpatialMap.TEST_DATA_DIR)

        map_df = pd.read_csv(TestSpatialMap.TEST_OUTPUT_DIR)

        # check that length is as expected
        self.assertEqual(4413, len(map_df))

        # check that length is as expected
        self.assertEqual(7, map_df.shape[1])

        # ensure all strings have a length greater than 0
        self.assertEqual(True, all(len(i) > 0 for i in map_df['State_Name']))


if __name__ == '__main__':
    unittest.main()
