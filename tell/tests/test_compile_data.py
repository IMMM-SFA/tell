import os
import unittest

from tell.data_process_compile_df import compile_data

class TestCompileData(unittest.TestCase):
    """Tests for functionality within data_process_compile_df.py"""

    def test_compile_data(self):
        """Test to ensure high level functionality of data_process_compile_df.py()"""

        start_year = 2018
        end_year = 2019
        data_input_dir = os.path.join((os.path.dirname(os.getcwd())), r'tell_data')
        compile_data(start_year, end_year, data_input_dir)


if __name__ == '__main__':
    unittest.main()
