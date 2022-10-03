import unittest
import pkg_resources

import pandas as pd

import tell.mlp_predict as mp


class TestMlpPredict(unittest.TestCase):
    """Tests for functionality within mlp_predict.py"""

    COMP_PREDICT_DF = pd.read_parquet(pkg_resources.resource_filename("tell", "tests/data/comp_predict.parquet"))

    def test_predict(self):
        """Test to ensure high level functionality of predict()"""

        df = mp.predict(region="ERCO",
                        year=2039,
                        data_dir=pkg_resources.resource_filename("tell", "tests/data"))

        pd.testing.assert_frame_equal(TestMlpPredict.COMP_PREDICT_DF, df)

    def test_predict_batch(self):
        """Test to ensure high level functionality of predict_batch()"""

        df = mp.predict_batch(target_region_list=["ERCO"],
                              year=2039,
                              data_dir=pkg_resources.resource_filename("tell", "tests/data"))

        pd.testing.assert_frame_equal(TestMlpPredict.COMP_PREDICT_DF, df)


if __name__ == '__main__':
    unittest.main()
