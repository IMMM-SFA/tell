import os
import shutil
import tempfile
import zipfile
from io import BytesIO as BytesIO
from pkg_resources import get_distribution

import requests


class InstallSample:
    """Download and unpack example data supplement from Zenodo that matches the current installed
    tell distribution.
    :param data_dir:                    Optional.  Full path to the directory you wish to store the data in.  Default is
                                        to install it in data directory of the package.
    :type data_dir:                     str
    """

    # URL for DOI minted example data hosted on Zenodo
    DATA_VERSION_URLS = {'0.0.1': 'https://zenodo.org/record/5784704/files/tell_weather_forcing_sample_data.zip?download=1',
                         '0.1.0': 'https://zenodo.org/record/5784704/files/tell_weather_forcing_sample_data.zip?download=1'}

    def __init__(self, data_dir=None):

        self.data_dir = data_dir

    def fetch_zenodo(self):
        """Download and unpack the Zenodo example data supplement for the
        current tell distribution."""

        # full path to the root directory where the example dir will be stored
        data_directory = self.data_dir

        # get the current version of cerf that is installed
        current_version = get_distribution('tell').version

        try:
            data_link = InstallSample.DATA_VERSION_URLS[current_version]

        except KeyError:
            msg = f"Link to data missing for current version:  {current_version}.  Please contact admin."

            raise KeyError(msg)

        # retrieve content from URL
        print(f"Downloading example data for weather forecasting {current_version}...")
        r = requests.get(data_link)

        with zipfile.ZipFile(BytesIO(r.content)) as zipped:

            # extract each file in the zipped dir to the project
            for f in zipped.namelist():

                extension = os.path.splitext(f)[-1]

                if len(extension) > 0:

                    basename = os.path.basename(f)
                    out_file = os.path.join(data_directory, basename)

                    # extract to a temporary directory to be able to only keep the file out of the dir structure
                    with tempfile.TemporaryDirectory() as tdir:

                        # extract file to temporary directory
                        zipped.extract(f, tdir)

                        # construct temporary file full path with name
                        tfile = os.path.join(tdir, f)

                        # transfer only the file sans the parent directory to the data package
                        shutil.copy(tfile, out_file)
        print(f"Done!")


def install_sample_data(data_dir=None):
    """Download and unpack example data supplement from Zenodo that matches the current installed
    tell distribution.
    :param data_dir:                    Optional.  Full path to the directory you wish to store the data in.  Default is
                                        to install it in data directory of the package.
    :type data_dir:                     str
    """

    zen = InstallSample(data_dir=data_dir)

    zen.fetch_zenodo()
