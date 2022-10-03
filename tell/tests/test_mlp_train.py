import unittest
import pkg_resources

import numpy as np
import pandas as pd

import tell.mlp_train as mp


class TestMlpTrain(unittest.TestCase):
    """Tests for functionality within mlp_train.py"""

    COMP_TRAIN_ARR = pd.read_parquet(pkg_resources.resource_filename("tell", "tests/data/comp_train.parquet"))["a"].values

    def test_train_mlp_model(self):
        """Test to ensure high level functionality of predict()"""

        np.random.seed(123)

        arr = mp.train_mlp_model(region="ERCO",
                                 x_train=np.arange(0.1, 100.0, 1.1).reshape(-1, 1),
                                 y_train=np.arange(0.1, 100.0, 1.1).reshape(-1, 1),
                                 x_test=np.arange(0.1, 100.0, 1.1).reshape(-1, 1),
                                 mlp_hidden_layer_sizes=10,
                                 mlp_max_iter=2,
                                 mlp_validation_fraction=0.4)

        np.testing.assert_array_equal(TestMlpTrain.COMP_TRAIN_ARR, arr)

    # def test_predict_batch(self):
    #     """Test to ensure high level functionality of predict_batch()"""
    #
    #     df = mp.predict_batch(target_region_list=["ERCO"],
    #                           year=2039,
    #                           data_dir=pkg_resources.resource_filename("tell", "tests/data"))
    #
    #     pd.testing.assert_frame_equal(TestMlpTrain.COMP_PREDICT_DF, df)


if __name__ == '__main__':
    unittest.main()
