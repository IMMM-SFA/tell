import unittest

import numpy as np
import pandas as pd

import tell.mlp_utils as mpu


class TestMlpUtils(unittest.TestCase):
    """Tests for functionality within mlp_utils.py"""

    SAMPLE_DATA_ARR = np.array([1.1, 2.2, 3.3])
    SAMPLE_TRAIN_ARR = np.array([[2.1, 3.2, 4.3],
                                 [5.4, 6.5, 8.7]])
    SAMPLE_TEST_ARR = np.array([[7.1, 4.2, 3.3],
                                [7.4, 3.5, 2.7]])
    SAMPLE_Y_TRAIN_ARR = np.array([1.1, 2.2, 3.3])
    SAMPLE_Y_TEST_ARR = np.array([4.1, 2.2, 1.3])

    COMP_NORM_ARR = np.array([-0.30, -0.30, -0.23])

    COMP_NORM_DICT = {'min_x_train': np.array([2.1, 3.2, 4.3]),
                      'max_x_train': np.array([5.4, 6.5, 8.7]),
                      'min_y_train': 1.1,
                      'max_y_train': 3.3,
                      'x_train_norm': np.array([[0., 0., 0.], [1., 1., 1.]]),
                      'y_train_norm': np.array([0., 0.5, 1.]),
                      'x_test_norm': np.array([[1.52,  0.3, -0.23],
                                              [1.61, 0.09, -0.36]]),
                      'y_test_norm': np.array([1.36, 0.5, 0.09])}

    COMP_DENORM_DF = pd.DataFrame({"datetime": [1.1, 2.2],
                                   "predictions": [3.4, 7.8],
                                   "ground_truth": [1.1, 2.2],
                                   "region": ["alpha"]*2})

    COMP_EVAL_DF = pd.DataFrame({"BA": "alpha",
                                 "RMS_ABS": [0.070711],
                                 "RMS_NORM": [0.041595],
                                 "MAPE": [0.022727],
                                 "R2": [0.983471]}).round(4)

    def test_normalize_prediction_data(self):
        """Test to ensure high level functionality of normalize_prediction_data()"""

        res = mpu.normalize_prediction_data(data_arr=TestMlpUtils.SAMPLE_DATA_ARR,
                                            min_train_arr=np.min(TestMlpUtils.SAMPLE_TRAIN_ARR, axis=0),
                                            max_train_arr=np.max(TestMlpUtils.SAMPLE_TRAIN_ARR, axis=0)).round(2)

        # run comp test
        np.testing.assert_array_equal(TestMlpUtils.COMP_NORM_ARR, res)

    def test_normalize_features(self):
        """Test to ensure high level functionality of normalize_features()"""

        res = mpu.normalize_features(x_train=TestMlpUtils.SAMPLE_TRAIN_ARR,
                                     x_test=TestMlpUtils.SAMPLE_TEST_ARR,
                                     y_train=TestMlpUtils.SAMPLE_Y_TRAIN_ARR,
                                     y_test=TestMlpUtils.SAMPLE_Y_TEST_ARR)

        for k in TestMlpUtils.COMP_NORM_DICT.keys():

            target = TestMlpUtils.COMP_NORM_DICT[k]

            if type(target) == float:
                self.assertEqual(target, res[k])
            else:
                np.testing.assert_array_equal(target, res[k].round(2))

    def test_get_balancing_authority_to_model_dict(self):
        """Test to ensure high level functionality of get_balancing_authority_to_model_dict()"""

        d = mpu.get_balancing_authority_to_model_dict()

        self.assertEqual(54, len(d))

    def test_denormalize_features(self):
        """Test to ensure high level functionality of denormalize_features()"""

        norm_dict = {"max_y_train": np.array([3.1, 4.2]),
                     "min_y_train": np.array([0.1, 1.2])}

        df = mpu.denormalize_features(region="alpha",
                                      normalized_dict=norm_dict,
                                      y_predicted_normalized=np.array([1.1, 2.2]),
                                      y_comparison=np.array([1.1, 2.2]),
                                      datetime_arr=np.array([1.1, 2.2]))

        pd.testing.assert_frame_equal(TestMlpUtils.COMP_DENORM_DF, df)

    def test_evaluate(self):
        """Test to ensure high level functionality of evaluate()"""

        df = mpu.evaluate(region="alpha",
                          y_predicted=np.array([1.1, 2.2]),
                          y_comparison=np.array([1.1, 2.3])).round(4)

        print(TestMlpUtils.COMP_EVAL_DF)
        print(df)

        pd.testing.assert_frame_equal(TestMlpUtils.COMP_EVAL_DF, df)


if __name__ == '__main__':
    unittest.main()
