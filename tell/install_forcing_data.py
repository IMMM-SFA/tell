import os
import shutil
import zipfile
import requests

from io import BytesIO as BytesIO
from pkg_resources import get_distribution


class InstallForcingSample:
    """Download the TELL sample forcing data package from Zenodo that matches the current installed tell distribution

    :param data_dir:                    Optional. Full path to the directory you wish to store the data in. Default is
                                        to install it in data directory of the package.

    :type data_dir:                     str

    """

    # URL for DOI minted example data hosted on Zenodo

    DATA_VERSION_URLS = {
        '0.0.1': 'https://zenodo.org/record/6354665/files/sample_forcing_data.zip?download=1',
        '0.1.0': 'https://zenodo.org/record/6354665/files/sample_forcing_data.zip?download=1',
        '0.1.1': 'https://zenodo.org/record/6354665/files/sample_forcing_data.zip?download=1',
        '0.1.2': 'https://zenodo.org/record/6354665/files/sample_forcing_data.zip?download=1',
        '0.1.3': 'https://zenodo.org/record/6354665/files/sample_forcing_data.zip?download=1',
        '0.1.4': 'https://zenodo.org/record/6354665/files/sample_forcing_data.zip?download=1',
        '0.1.5': 'https://zenodo.org/record/6354665/files/sample_forcing_data.zip?download=1',
        '1.0.0': 'https://zenodo.org/record/6354665/files/sample_forcing_data.zip?download=1',
        '1.1.0': 'https://zenodo.org/records/13344803/files/sample_forcing_data.zip?download=1',
        '1.2.0': 'https://zenodo.org/records/13344803/files/sample_forcing_data.zip?download=1',
        '1.2.1': 'https://zenodo.org/records/13344803/files/sample_forcing_data.zip?download=1',
    }

    DEFAULT_VERSION = 'https://zenodo.org/records/13344803/files/sample_forcing_data.zip?download=1'

    def __init__(self, data_dir=None):
        self.data_dir = data_dir

    def fetch_zenodo(self):
        """Download the tell sample forcing data package from Zenodo that matches the current tell distribution"""

        # Get the current version of TELL that is installed:
        current_version = get_distribution('tell').version

        try:
            data_link = InstallForcingSample.DATA_VERSION_URLS[current_version]

        except KeyError:
            msg = f"Link to data missing for current version: {current_version}. Using default version: {InstallForcingSample.DEFAULT_VERSION}"

            data_link = InstallForcingSample.DEFAULT_VERSION

            print(msg)

        # Retrieve content from the URL:
        print(f"Downloading the sample forcing data package for tell version {current_version}...")
        r = requests.get(data_link)

        # Extract the data from the .zip file:
        with zipfile.ZipFile(BytesIO(r.content)) as zipped:
            zipped.extractall(self.data_dir)

        # Remove the empty "__MACOSX" directory:
        shutil.rmtree(os.path.join(self.data_dir, r'__MACOSX'))

        # Report that the download is complete:
        print(f"Done!")


def install_sample_forcing_data(data_dir=None):
    """Download the tell sample forcing data package from Zenodo

    :param data_dir:                    Optional. Full path to the directory you wish to store the data in. Default is
                                        to install it in data directory of the package.

    :type data_dir:                     str
    """

    zen = InstallForcingSample(data_dir=data_dir)

    zen.fetch_zenodo()
