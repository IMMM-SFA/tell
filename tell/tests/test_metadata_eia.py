import unittest

from tell.metadata_eia import metadata_eia


class TestMetadataEIA(unittest.TestCase):
    """Tests for functionality within metadata_eia.py"""

    def test_metadata_eia(self):
        """Test to ensure high level functionality of metadata_eia.py()"""

        num = [189, 1, 317]
        metadata_df = metadata_eia(num)

        # check length of list
        self.assertEqual(3, len(metadata_df))

        # ensure BA name column is object in df are object
        self.assertEqual(True, all(isinstance(i, object) for i in metadata_df['BA_Name']))


if __name__ == '__main__':
    unittest.main()
