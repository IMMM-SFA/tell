import unittest
import pandas as pd


class MyTestCase(unittest.TestCase):

    def test_something(self):

        self.assertEqual(True, True)

    def test_df(self):

        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]})
        df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]})

        pd.testing.assert_frame_equal(df1, df2)


if __name__ == '__main__':
    unittest.main()
