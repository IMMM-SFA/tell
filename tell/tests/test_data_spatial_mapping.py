import os
import unittest

from tell.tell.data_spatial_mapping import map_ba_service_territory


class TestSpatialMapping(unittest.TestCase):
    """Tests for functionality within data_spatial_mapping.py"""

    def test_spatial_mapping(self):
        """Test to ensure high level functionality of data_spatial_mapping.py()"""

        start_year = 2018
        end_year = 2019
        data_input_dir = os.path.join((os.path.dirname(os.getcwd())), r'tell_data')

        map_ba_service_territory(start_year, end_year, data_input_dir)


if __name__ == '__main__':
    unittest.main()
