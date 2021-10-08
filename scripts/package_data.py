import pkg_resources


def get_raw_data():
    """Return tell raw data zipfile"""

    return pkg_resources.resource_filename('tell', 'tell_raw_data.zip')

def get_data_directory():
    """Return the directory of where the cerf package data resides."""

    return pkg_resources.resource_filename('tell', 'data')
