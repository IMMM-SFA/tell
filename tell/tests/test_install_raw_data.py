import unittest

import tell
import tell.install_raw_data as td


class TestInstallRawData(unittest.TestCase):

    def test_instantiate(self):

        zen = td.InstallRawData(data_dir="fake")

        # ensure default version is set
        self.assertEqual(str, type(zen.DEFAULT_VERSION))

        # ensure urls present for current version
        self.assertTrue(tell.__version__ in zen.DATA_VERSION_URLS)


if __name__ == '__main__':
    unittest.main()
