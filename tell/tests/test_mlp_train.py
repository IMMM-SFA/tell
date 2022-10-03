import unittest
import pkg_resources

import numpy as np
import pandas as pd

import tell.mlp_train as mp


class TestMlpTrain(unittest.TestCase):
    """Tests for functionality within mlp_train.py"""

    COMP_TRAIN_ARR = pd.read_parquet(pkg_resources.resource_filename("tell", "tests/data/comp_train.parquet"))["a"].values
    COMP_TRAIN_PRED_DF = pd.read_parquet(pkg_resources.resource_filename("tell", "tests/data/train_data/comp_train_pred.parquet"))
    COMP_TRAIN_VALID_DF = pd.read_parquet(pkg_resources.resource_filename("tell", "tests/data/train_data/comp_train_valid.parquet"))

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

    def test_train(self):
        """Test to ensure high level functionality of train()"""

        np.random.seed(123)

        prediction_df, validation_df = mp.train(region="ERCO",
                                                data_dir=pkg_resources.resource_filename("tell", "tests/data/train_data"))

        pd.testing.assert_frame_equal(TestMlpTrain.COMP_TRAIN_PRED_DF, prediction_df)
        pd.testing.assert_frame_equal(TestMlpTrain.COMP_TRAIN_VALID_DF, validation_df)

    def test_train_batch(self):
        """Test to ensure high level functionality of train()"""

        np.random.seed(123)

        prediction_df, validation_df = mp.train_batch(target_region_list=["ERCO"],
                                                      data_dir=pkg_resources.resource_filename("tell", "tests/data/train_data"))

        pd.testing.assert_frame_equal(TestMlpTrain.COMP_TRAIN_PRED_DF, prediction_df)
        pd.testing.assert_frame_equal(TestMlpTrain.COMP_TRAIN_VALID_DF, validation_df)


if __name__ == '__main__':
    unittest.main()
