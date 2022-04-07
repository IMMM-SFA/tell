import unittest

from tell.execute_forward import execute_forward

class TestExecuteForward(unittest.TestCase):
    """Tests for functionality within execute_forward.py"""

    def test_execute_forward(self):
        """Test to ensure high level functionality of execute_forward.py()"""

        execute_forward(year_to_process, mlp_input_dir, ba_geolocation_input_dir, pop_input_dir, gcam_usa_input_dir,
                        data_output_dir)


if __name__ == '__main__':
    unittest.main()
