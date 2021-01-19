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

        # split out state abbreviation and space seperators; only keep county info and ensure there
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


def keep_valid(x):
    """Keep only dictionaries that have a count for a county name present."""

    d = {}
    for value in x:
        if type(value) is dict:
            key = list(value.keys())[0]
            d[key] = value[key]

    return d


def find_county(d):
    """Add the FIPS key to the data frame where the optimal value with a count
    of 1 has been identified."""

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
    ideally this is 1, if a different number investigate futher"""

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


def prepare_data(fips_file, service_area_file, sales_ult_file, bal_auth_file):
    """Load and prepare data.  Reduce complexity by making state and county names lower case and splitting
    out commonly known trailing words that do not exist in both data sets.  Build key to join by
    where <state_abbrev>_<county_lower>.

    """

    # read in data
    df_fips = pd.read_excel(fips_file)
    df_states = pd.read_excel(service_area_file, sheet_name='Counties_States')
    df_ult = pd.read_excel(sales_ult_file, sheet_name='States', skiprows=2)
    df_ba = pd.read_excel(bal_auth_file)

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

    # filter df_ult and df_ba
    df_ult = df_ult[["Utility Number", "Utility Name", "BA Code"]]
    df_ba = df_ba[["BA Code", "BA ID", "Balancing Authority Name"]]

    return df_fips, df_states, df_ult, df_ba


def filter_one(df_fips, df_states, df_ult, df_ba):
    """Join datasets together where possible based on common key."""

    # merge states and fips based on key
    df_states_fips = pd.merge(left=df_states, right=df_fips, left_on='states_key', right_on='fips_key', how='left')

    # merge to states_fips merge to ba utility dataframe by ult number
    df_states_fips['Utility Number'] = df_states_fips['Utility Number'].astype(float)
    df_fips_ult = df_states_fips.merge(df_ult, on='Utility Number', how='left')

    # add the BA number and name by BA ulitity code
    df_valid = df_fips_ult.merge(df_ba, on='BA Code', how='left')

    # filter out remaining rows that did not have a match
    df_nan = df_valid.loc[df_valid['county_lower_y'].isna()].copy()

    # only keep successfully joined data in the merged df
    df_valid = df_valid.loc[~df_valid['county_lower_y'].isna()].copy()

    # drop unneeded columns
    df_valid.drop(columns=['county_lower_y', 'Utility Name_y'], inplace=True)
    df_nan.drop(columns=['county_lower_y'], inplace=True)

    return df_valid, df_nan


def filter_two(df_fips, df_nan, df_valid):
    """Match NaN records by cleaning up naming conventions based on the most suitable match."""

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


def data_format(df_valid):
    """Select for wanted columns and rename columns."""

    df_valid = df_valid.loc[['Data Year', 'Utility Number', 'Utility Name_x','state_abbreviation', 'state_name',
                            'state_FIPS', 'county_name', 'county_FIPS', 'BA ID', 'BA Code', 'Balancing Authority Name']]

    df_valid = df_valid.loc.rename(columns={"Date Year": "Year", "Utility Number": "Utility_Number",
                                        "Utility Name_x": "Utility_Name", "state_abbreviation": "State_Abbreviation",
                                        "state_name": "State_Name", "state_FIPS": "State_FIPS",
                                        "county_name": "County_Name", "county_FIPS": "County_FIPS",
                                        "BA ID": "BA_Number", "BA Code": "BA_Abbreviation",
                                        "Balancing Authority Name": "BA_Name"})
    return df_valid


data_dir = 'C:\\Users\\mcgr323\\projects\\TELL\\inputs\\'
data_dir_2 = 'C:\\Users\\mcgr323\\projects\\TELL\\inputs\\EIA_861\\Raw_Data\\2019\\'

# paths to files
fips_file = os.path.join(data_dir, 'state_and_county_fips_codes.xlsx')
service_area_file = os.path.join(data_dir_2, 'Service_Territory_2019.xlsx')
sales_ult_file = os.path.join(data_dir_2, 'Sales_Ult_Cust_2019.xlsx')
bal_auth_file = os.path.join(data_dir_2, 'Balancing_Authority_2019.xlsx')

# prepare data
df_fips, df_states, df_ult, df_ba = prepare_data(fips_file, service_area_file, sales_ult_file, bal_auth_file)

# apply filter one
df_valid, df_nan = filter_one(df_fips, df_states, df_ult, df_ba)

# apply filter two
df_valid, df_nan = filter_two(df_fips, df_nan, df_valid)

# format columns
df_valid = data_format(df_valid)

unmatched_counties = df_nan['county_lower_x'].unique()
list(unmatched_counties)

for i in unmatched_counties:
    possible_matches = df_nan.loc[df_nan['county_lower_x'] == i]['matches'].values[0]

    print(f"States file county: {i}")
    print(f"Possible FIPS matches:  {possible_matches}")
    print('\n')

df_fips.loc[~df_fips['fips_key'].isin(df_valid['fips_key'])]

df_valid.head()

df_valid.to_csv('FIPS_Service_match_2019.csv', sep=',')
