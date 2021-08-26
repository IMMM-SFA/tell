import os
import io
import requests
import zipfile
import logging
import sys

from pkg_resources import get_distribution


def install_supplement(download_directory):
    """Convenience wrapper for the InstallSupplement class.

    Download and unpack example data supplement from Zenodo that matches the current installed
    distribution.

    :param download_directory:                  Full path to the directory you wish to install
                                                the example data to.  Must be write-enabled
                                                for the user.

    """

    get = InstallSupplement(example_data_directory=download_directory)
    get.fetch_zenodo()


class InstallSupplement:
    """Download and unpack example data supplement from Zenodo that matches the current installed
    distribution.

    :param example_data_directory:              Full path to the directory you wish to install
                                                the example data to.  Must be write-enabled
                                                for the user.

    """

    # URL for DOI minted example data hosted on Zenodo matching the version of release
    # TODO:  this dictionary should really be brought in from a config file within the package
    # TODO:  replace current test link with a real data link
    DATA_VERSION_URLS = {'0.1.0': 'https://zenodo.org/record/3856417/files/test.zip?download=1'}

    def __init__(self, example_data_directory, model_name='tell'):

        self.initialize_logger()

        # full path to the Xanthos root directory where the example dir will be stored
        self.example_data_directory = self.valid_directory(example_data_directory)

        self.model_name = model_name

    def initialize_logger(self):
        """Initialize logger to stdout."""

        # initialize logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # logger console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    @staticmethod
    def close_logger():
        """Shutdown logger."""

        # Remove logging handlers
        logger = logging.getLogger()

        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        logging.shutdown()

    def valid_directory(self, directory):
        """Ensure the provided directory exists."""

        if os.path.isdir(directory):
            return directory
        else:
            msg = f"The write directory provided by the user does not exist:  {directory}"
            logging.exception(msg)
            self.close_logger()
            raise NotADirectoryError(msg)

    def fetch_zenodo(self):
        """Download and unpack the Zenodo example data supplement for the
        current distribution."""

        # get the current version of the package is installed
        current_version = get_distribution(self.model_name).version

        try:
            data_link = InstallSupplement.DATA_VERSION_URLS[current_version]

        except KeyError:
            msg = f"Link to data missing for current version:  {current_version}.  Please contact admin."
            logging.exception(msg)
            self.close_logger()
            raise

        # retrieve content from URL
        try:
            logging.info(f"Downloading example data for version {current_version} from {data_link}")
            r = requests.get(data_link)

            with zipfile.ZipFile(io.BytesIO(r.content)) as zipped:

                # extract each file in the zipped dir to the project
                for f in zipped.namelist():
                    logging.info("Unzipped: {}".format(os.path.join(self.example_data_directory, f)))
                    zipped.extract(f, self.example_data_directory)

            logging.info("Download and install complete.")

            self.close_logger()

        except requests.exceptions.MissingSchema:
            msg = f"URL for data incorrect for current version:  {current_version}.  Please contact admin."
            logging.exception(msg)
            self.close_logger()
            raise
