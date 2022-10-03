import unittest

import tell
from tell.install_quickstarter_data import InstallQuickstarterData


class TestInstallQuickstarterData(unittest.TestCase):

    def test_instantiate(self):

        zen = InstallQuickstarterData(data_dir="fake")

        # ensure default version is set
        self.assertEqual(str, type(zen.DEFAULT_VERSION))

        # ensure urls present for current version
        self.assertTrue(tell.__version__ in zen.DATA_VERSION_URLS)


if __name__ == '__main__':
    unittest.main()
