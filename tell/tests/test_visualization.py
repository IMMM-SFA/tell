import unittest

from tell.visualization import plot_ba_load_time_series

class TestVisualization(unittest.TestCase):
    """Tests for functionality within visualization.py"""

    def test_visualization(self):
        """Test to ensure high level functionality of visualization.py()"""

        plot_ba_load_time_series(ba_to_plot, year_to_plot, data_input_dir, image_output_dir,
        image_resolution, save_images = False)


if __name__ == '__main__':
    unittest.main()
