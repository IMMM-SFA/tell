"""match.py

Functionality to join FIPS codes with utility zones.

"""

import os

import numpy as np
import pandas as pd


def count_matches(states_key, fips_key, ignore=['and', 'if', 'on', 'an', 'a', 'the']):
    """Count the number of word matches between two primary keys.

    :param states_key:                 The key representing the <state_abbrev>_<county_name> in the
                                       states data frame
    :type states_key:                  str

    :param fips_key:                   The key representing the <state_abbrev>_<county_name> in the
                                       FIPS data frame
    :type fips_key:                    str

    :param ignore:                     A list of common english words to ignore.
    :type ignore:                      list

    :return:                           Total number of matches or None if no matches

    """

    # get states state name
    states_split = states_key.split('_')
    states_state_name = states_split[0]

    # get fips state name
    fips_split = fips_key.split('_')
    fips_state_name = fips_split[0]

    # if state names match proceed, else return None
    if states_state_name == fips_state_name:

        # split out state abbreviation and space separators; only keep county info and ensure there
        #    are no underscores in the county name
        states_key_split = [ix[0] for ix in [i.split('-') for i in ' '.join(states_split[1:]).split(' ')]]

        # check for any "de" suffix and join to second position (e.g., "de witt" to "dewitt")
        states_key_split = combine_elements('de', states_key_split)

        # set initial count of matches to zero
        ct = 0

        # for each word in the states data frame key...
        for word in states_key_split:

            # if word in the fips data frame key, increment count
            fips_key_split = [ix[0] for ix in [i.split('-') for i in ' '.join(fips_split[1:]).split(' ')]]

            # check for any "de" suffix and join to second position (e.g., "de witt" to "dewitt")
            fips_key_split = combine_elements('de', fips_key_split)

            if (word not in ignore) and (word in fips_key_split):
                ct += 1

        # if there were no matches, return None
        if ct > 0:
            return {fips_key: ct}
        else:
            return None

    else:
        return None


def combine_elements(part, part_list):
    # check for any "de" suffix and join to second position (e.g., "de witt" to "dewitt")
    if part in part_list:
        one_idx = part_list.index(part)
        combined = f"{part_list[one_idx]}{part_list[one_idx + 1]}"
        part_list.pop(one_idx + 1)
        part_list.pop(one_idx)
        part_list.insert(0, combined)

    return part_list