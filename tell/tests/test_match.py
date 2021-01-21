import unittest
import pandas as pd

import tell.match as match


class TestMatch(unittest.TestCase):
    """Tests for functions contained in match.py"""

    def test_data_format(self):
        """Tests to ensure the `data_format` function is providing the correct output."""

        # a list of expected column names for the output data frame
        expected_cols = ["year", "utility_number", "utility_name", "state_abbreviation", "state_name", "state_fips",
                         "county_name", "county_fips", "ba_number", "ba_abbreviation", "ba_name"]

        # create a comparison input data frame that should run successfully
        success_df = self.construct_sample_dataframe(success=True)

        # create a comparison input data frame that should produce a KeyError when tested
        fail_df = self.construct_sample_dataframe(success=False)

        # generate an output data frame that has had columns renamed and reformatted
        output_df = match.data_format(success_df)

        # test for a match in the shape (cols, rows) from the input data frame to the output data frame
        self.assertEqual(success_df.shape, output_df.shape)

        # test that the column names in the output match what is expected:
        self.assertEqual(expected_cols, list(output_df.columns))

        # test that a data frame with the incorrect number of columns in the input raise a KeyError
        with self.assertRaises(KeyError):
            output_df = match.data_format(fail_df)

    @staticmethod
    def construct_sample_dataframe(success=True):
        """Construct a sample dictionary that can be used as an input to the `data_format` function.

        :param success:                     If True, then a data frame will be created that is expected
                                            to be successful and not raise any test failures.  If False,
                                            a data frame with an incorrect number of columns will be
                                            created so that the test will be expected to fail by raising
                                            a KeyError.  Default:  True
        :type success:                      bool

        :return:                            Sample Pandas DataFrame

        """

        # expected column names
        if success:
            col_names = ['Data Year', 'Utility Number', 'Utility Name_x', 'state_abbreviation', 'state_name',
                         'state_FIPS', 'county_name', 'county_FIPS', 'BA ID', 'BA Code', 'Balancing Authority Name']

        else:
            col_names = ['failure']

        # create a sample dictionary that can be used as an input to the `data_format` function
        data_dict = {k: ['a', 'b, c'] for k in col_names}

        return pd.DataFrame(data_dict)


if __name__ == '__main__':
    unittest.main()

