import os
import shutil
import zipfile
import requests

from io import BytesIO as BytesIO
from pkg_resources import get_distribution


class InstallOutputSample:
    """Download the TELL sample output data package from Zenodo that matches the current installed tell distribution

    :param data_dir:                    Optional. Full path to the directory you wish to store the data in. Default is
                                        to install it in data directory of the package.
    :type data_dir:                     str

    """

    # URL for DOI minted example data hosted on Zenodo
    DATA_VERSION_URLS = {'0.0.1': 'https://zenodo.org/record/6338472/files/sample_output_data.zip?download=1'}

    def __init__(self, data_dir=None):

        self.data_dir = data_dir

    def fetch_zenodo(self):
        """Download the TELL sample output data package from Zenodo that matches the current installed tell distribution"""

        # Get the current version of TELL that is installed:
        current_version = get_distribution('tell').version

        # Try to download the data:
        try:
            data_link = InstallOutputSample.DATA_VERSION_URLS[current_version]

        except KeyError:
            msg = f"Link to data missing for current version: {current_version}. Please contact an administrator."

            raise KeyError(msg)

        # Retrieve content from the URL:
        print(f"Downloading the sample output data package for tell version {current_version}...")
        r = requests.get(data_link)

        # Extract the data from the .zip file:
        with zipfile.ZipFile(BytesIO(r.content)) as zipped:
            zipped.extractall(self.data_dir)

        # Remove the empty "__MACOSX" directory:
        shutil.rmtree(os.path.join(self.data_dir, r'__MACOSX'))

        # Report that the download is complete:
        print(f"Done!")


def install_sample_output_data(data_dir=None):
    """Download the TELL sample output data package from Zenodo

    :param data_dir:                    Optional. Full path to the directory you wish to store the data in. Default is
                                        to install it in data directory of the package.

    :type data_dir:                     str

    """

    zen = InstallOutputSample(data_dir=data_dir)

    zen.fetch_zenodo()
