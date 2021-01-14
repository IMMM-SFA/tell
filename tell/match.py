"""match.py

Functionality to join FIPS codes with utility zones.

"""

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
    """Check for any "de" suffix and join to second position (e.g., "de witt" to "dewitt")

        :param part:                 The suffix of interest to be searched for in the parts_list
        :type part:                  str

        :param part_list:           The str of interest separated into a list of parts
        :type part_list:            list

        :return:                    Joined part list with combined to suffix

        """

    if part in part_list:
        one_idx = part_list.index(part)
        combined = f"{part_list[one_idx]}{part_list[one_idx + 1]}"
        part_list.pop(one_idx + 1)
        part_list.pop(one_idx)
        part_list.insert(0, combined)

    return part_list


def keep_valid(x):
    """Keep only dictionaries that have a count for a county name present.

        :param x:                Dictionary with matches from filter_two

        :type x:                  str

        :return:                  Dictionary with count with county name

        """

    d = {}
    for value in x:
        if type(value) is dict:
            key = list(value.keys())[0]
            d[key] = value[key]

    return d


def find_county(d):
    """Add the FIPS key to the data frame where the optimal value with a count
    of 1 has been identified.

    :param d:              Dictionary with count with county name from keep_valid
    :type d:               str

    :return:               FIPS key combined to county dictionary with a count of 1 or more

    """

    if len(d) > 0:

        values = [d[k] for k in d.keys()]
        max_value = max(values)

        keys, count = np.unique(values, return_counts=True)
        val_dict = {}

        for k, v in zip(keys, count):
            val_dict[k] = v

        if val_dict[max_value] > 1:
            return None
        else:
            return list(d.keys())[list(d.values()).index(max_value)]

    else:
        return None


def get_max_count(d):
    """Generate a column that has the count of the optimal county name;
    ideally this is 1, if a different number investigate futher

    :param d:              Dictionary with FIPS key from find_county
    :type d:               str

    :return:               Dataframe of FIPS matches with column count of optimal county name

    """

    if len(d) > 0:

        values = [d[k] for k in d.keys()]

        max_value = max(values)

        keys, count = np.unique(values, return_counts=True)
        val_dict = {}

        for k, v in zip(keys, count):
            val_dict[k] = v

        return val_dict[max_value]

    else:
        return None


def prepare_data(fips_file, service_area_file):
    """Load and prepare data.  Reduce complexity by making state and county names lower case and splitting
    out commonly known trailing words that do not exist in both data sets.  Build key to join by
    where <state_abbrev>_<county_lower>.

    :param fips_file:              FIPS csv input
    :type fips_file:               str

    :param service_area_file:     Service area csv input
    :type service_area_file:      str

    :return:                      df_fips: Dataframe of prepared and cleaned FIPS data
    :return:                      df_states:: Dataframe of prepared and cleaned Service area data

    """

    # read in data
    df_fips = pd.read_excel(fips_file)
    df_states = pd.read_excel(service_area_file, sheet_name='Counties_States')

    # strip county name from full reference and make lower case for known trailing words replace apostrophes
    df_fips['county_lower'] = df_fips['county_name'].apply(lambda x: x.lower().split(' county')[0])
    df_fips['county_lower'] = df_fips['county_lower'].apply(lambda x: x.lower().split(' parish')[0]).str.replace("'",
                                                                                                                 "")

    # make state abbreviation lower case
    df_fips['state_abbreviation'] = df_fips['state_abbreviation'].str.lower()

    # create key as lower case county name replace apostrophes
    df_states['county_lower'] = df_states['County'].str.lower().str.replace("'", "")
    df_states['State'] = df_states['State'].str.lower()

    # create a unified <state_abbrev>_<county_lower> key to merge by
    df_fips['fips_key'] = df_fips['state_abbreviation'] + '_' + df_fips['county_lower']
    df_states['states_key'] = df_states['State'] + '_' + df_states['county_lower']

    return df_fips, df_states


def filter_one(df_fips, df_states):
    """Join datasets together where possible based on common key.

    :param fips_file:             Dataframe of prepared and cleaned FIPS data from prepare_data
    :type fips_file:              str

    :param service_area_file:     Dataframe of prepared and cleaned Service area data from prepare_data
    :type service_area_file:      str

    :return:                      Dataframe of valid data with a match between df_fips and df_states
    :return:                      Dataframe of data without a match between df_fips and df_states
    """


    # merge based on key
    df_valid = pd.merge(left=df_states, right=df_fips, left_on='states_key', right_on='fips_key', how='left')

    # filter out remaining rows that did not have a match
    df_nan = df_valid.loc[df_valid['county_lower_y'].isna()].copy()

    # only keep successfully joined data in the merged df
    df_valid = df_valid.loc[~df_valid['county_lower_y'].isna()].copy()

    # drop unneeded columns
    df_valid.drop(columns=['county_lower_y'], inplace=True)
    df_nan.drop(columns=['county_lower_y'], inplace=True)

    return df_valid, df_nan


def filter_two(df_fips, df_nan, df_valid):
    """Match NaN records by cleaning up naming conventions based on the most suitable match.

    :param df_fips:     Dataframe of prepared and cleaned FIPS data from prepare_data
    :type df_fips:      str

    :param df_nan:      Dataframe of data without a match between df_fips and df_states
    :type df_nan:       str

    :param df_valid:    Dataframe of valid data with a match between df_fips and df_states
    :type df_valid:     str

    :return:           df_valid: Dataframe of valid data with a match between df_fips and df_states
    :return:           df_nan_bad: Dataframe of data without a match between df_fips and df_states
    """

    # get keys from states that are in teh FIPS code data frame that have NaN records
    nan_keys_fips = df_fips.loc[df_fips['state_abbreviation'].isin(df_nan['State'].unique())]['fips_key'].unique()

    # df_nan['fips_key'] = None
    df_nan['matches'] = [[]] * df_nan.shape[0]

    for index, fips_key in enumerate(nan_keys_fips):
        df_nan['matches'] = df_nan['matches'] + df_nan['states_key'].apply(lambda x: [count_matches(x, fips_key)])

    # keep only dictionaries that have a count for a county name present
    df_nan['matches'] = df_nan['matches'].apply(keep_valid)

    # generate a column that has the count of the optimal county name;
    #   ideally this is 1, if a different number investigate futher
    df_nan['count_of_selected'] = df_nan['matches'].apply(get_max_count)

    # add the FIPS key to the data frame where the optimal value with a count of 1 has been identified
    df_nan['fips_key'] = df_nan['matches'].apply(find_county)

    # remove unneeded columns
    df_nan.drop(columns=['state_name', 'state_abbreviation', 'state_FIPS',
                         'county_name', 'county_FIPS'], inplace=True)

    # extract rows now having a vaild match
    df_nan_good = df_nan.loc[df_nan['count_of_selected'] == 1].copy()

    # extract rows still having an invaild match
    df_nan_bad = df_nan.loc[df_nan['count_of_selected'] != 1].copy()

    # merge good records with FIPS data frame
    mrx = pd.merge(left=df_nan_good, right=df_fips, left_on='fips_key', right_on='fips_key', how='left')

    # drop unneeded columns
    mrx.drop(columns=['count_of_selected', 'matches'], inplace=True)

    # reorder columns to match df_valid
    mrx = mrx[df_valid.columns].copy()

    # add newly valid records to master data frame
    df_valid = pd.concat([df_valid, mrx])

    return df_valid, df_nan_bad