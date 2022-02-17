import os
import pkg_resources

import yaml


def get_ba_abbreviations() -> list:
    """Get balancing authority abbreviations from the reference YAML file.

    :return:                        List of BA abbreviations

    """

    file_name = pkg_resources.resource_filename('tell', os.path.join('data', 'balancing_authority_names.yml'))

    yaml_dict = read_yaml(file_name)

    return [i for i in yaml_dict.keys()]


def read_yaml(yaml_file: str) -> dict:
    """Read a YAML file.

    :param yaml_file:               Full path with file name and extension to the input YAML file
    :type yaml_file:                str

    :return:                        Dictionary

    """

    with open(yaml_file, 'r') as yml:
        return yaml.load(yml, Loader=yaml.FullLoader)
