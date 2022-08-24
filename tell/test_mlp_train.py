import unittest

import numpy as np

import tell.mlp_train as mpt


class TestMlpTrain(unittest.TestCase):
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
                      'x_test_norm': np.array([[1.52, 0.3, -0.23],
                                               [1.61, 0.09, -0.36]]),
                      'y_test_norm': np.array([1.36, 0.5, 0.09])}

    def test_normalize_features(self):
        """Test to ensure high level functionality of normalize_features()"""

        res = mpt.normalize_features(x_train=TestMlpTrain.SAMPLE_TRAIN_ARR,
                                     x_test=TestMlpTrain.SAMPLE_TEST_ARR,
                                     y_train=TestMlpTrain.SAMPLE_Y_TRAIN_ARR,
                                     y_test=TestMlpTrain.SAMPLE_Y_TEST_ARR)

        for k in TestMlpTrain.COMP_NORM_DICT.keys():

            target = TestMlpTrain.COMP_NORM_DICT[k]

            if type(target) == float:
                self.assertEqual(target, res[k])
            else:
                np.testing.assert_array_equal(target, res[k].round(2))


if __name__ == '__main__':
    unittest.main()
