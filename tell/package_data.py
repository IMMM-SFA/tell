import pkg_resources


def get_data_directory():
    """Return the directory of where the cerf package data resides."""

    return pkg_resources.resource_filename('tell', 'data')
