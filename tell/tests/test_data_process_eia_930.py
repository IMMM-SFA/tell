import os
import unittest

from tell.data_process_eia_930 import process_eia_930_data


class TestEIA930Data(unittest.TestCase):
    """Tests for functionality within data_process_eia_930.py"""

    def test_eia_930_data(self):
        """Test to ensure high level functionality of data_process_eia_930.py()"""

        data_input_dir = os.path.join((os.path.dirname(os.getcwd())), r'tell_data')

        process_eia_930_data(data_input_dir, n_jobs=-1)


if __name__ == '__main__':
    unittest.main()
