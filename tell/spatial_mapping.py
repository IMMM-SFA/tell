import os
import logging
import time

import numpy as np
import pandas as pd

from pandas import DataFrame
from tell.logger import Logger


def count_matches(states_key: str, fips_key: str, ignore=('and', 'if', 'on', 'an', 'a', 'the')) -> dict:
    """Count the number of word matches between two primary keys.

    :param states_key:                 Key of the <state_abbrev>_<county_name> in the 'df_states' dataframe
    :type states_key:                  str

    :param fips_key:                   Key of the <state_abbrev>_<county_name> in the 'df_fips' dataframe
    :type fips_key:                    str

    :param ignore:                     A list of common english words to ignore
    :type ignore:                      list

    :return:                           Total number of matches or None if there are no matches

    """

    # Split the states names from the 'states_key' list:
    states_split = states_key.split('_')
    states_state_name = states_split[0]

    # Split the FIPS code from the 'fips_key' list:
    fips_split = fips_key.split('_')
    fips_state_name = fips_split[0]

    # If the state names match then proceed, else return None:
    if states_state_name == fips_state_name:

        # Split out state abbreviation and space separator and remove underscores:
        states_key_split = [ix[0] for ix in [i.split('-') for i in ' '.join(states_split[1:]).split(' ')]]

        # Check for any "de" suffix and join to the second position (e.g., change "de witt" to "dewitt"):
        states_key_split = combine_elements('de', states_key_split)

        # Set initial the count of matches to zero:
        ct = 0

        # Loop over each word in the 'states_key' and count the matches:
        for word in states_key_split:

            # Split out state abbreviation and space separator and remove underscores:
            fips_key_split = [ix[0] for ix in [i.split('-') for i in ' '.join(fips_split[1:]).split(' ')]]

            # Check for any "de" suffix and join to the second position (e.g., change "de witt" to "dewitt"):
            fips_key_split = combine_elements('de', fips_key_split)

            # If the word matches the 'fips_key' list then increment the counter:
            if (word not in ignore) and (word in fips_key_split):
               ct += 1

        # Return the number of matches or, if there were no matches, return None:
        if ct > 0:
            return {fips_key: ct}
        else:
            return None

    else:
        return None


def combine_elements(part: str, part_list: list) -> list:
    """Check for any suffixes and join them to the second position (e.g., change "de witt" to "dewitt")

    :param part:                The suffix of interest to be searched for in the parts_list
    :type part:                 str

    :param part_list:           The string of interest separated into a list of parts
    :type part_list:            list

    :return:                    Joined part list with combined to suffix

    """

    # I don't entirely understand what is happening in this code block:
    if part in part_list:
        one_idx = part_list.index(part)
        combined = f"{part_list[one_idx]}{part_list[one_idx + 1]}"
        part_list.pop(one_idx + 1)
        part_list.pop(one_idx)
        part_list.insert(0, combined)

    return part_list


def keep_valid(x: dict) -> dict:
    """Keep only dictionaries that have a non-zero count for a county name present.

    :param x:                Dictionary with matches from filter_two
    :type x:                 dict

    :return:                 Dictionary with non-zero count of potential county names

    """

    # Initiate an empty dictionary to store the results:
    d = {}

    # Loop over the dictionary of matches from filter two:
    for value in x:
        # If there is a non-zero count of potential matches then extract that row:
        if type(value) is dict:
            key = list(value.keys())[0]
            d[key] = value[key]

    # Return the dictionary of valid matches:
    return d


# Can Casey M. or somebody please comment this code block:
def find_county(d: dict) -> dict:
    """Add the FIPS code to the data frame where the optimal value (i.e., >= 1) has been identified.

    :param d:              Dictionary with non-zero count of potential county names
    :type d:               dict

    :return:               Dictionary of FIPS codes combined to the county dictionary with a count of 1 or more
      
    """

    # I don't entirely understand what is happening in this code block:
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


# Can Casey M. or somebody please comment this code block:
def get_max_count(d: dict) -> DataFrame:
    """Count of the optimal county name matches: Ideally this is 1, but if it's different then investigate further

     :param d:              Dictionary of FIPS codes combined to the county dictionary with a count of 1 or more
     :type d:               dict

     :return:               DataFrame of FIPS codes with column count of optimal county name

     """

    # I don't entirely understand what is happening in this code block:
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


def prepare_data(fips_file: str, service_area_file: str, sales_ult_file: str, bal_auth_file: str) -> DataFrame:
    """Read in the raw Excel files and clean them. Reduce complexity by making state and county names lower case
    and splitting out commonly known trailing words that do not exist in both data sets. Build a key to join the
    multiple datasets using <state_abbrev>_<county_lower>.

    :param fips_file:                           County FIPS code .csv file
    :type fips_file:                            str

    :param service_area_file:                   Balancing authority service area Excel file
    :type service_area_file:                    str

    :param sales_ult_file:                      Balancing authority sales to ultimate customer Excel file
    :type sales_ult_file:                       str

    :param bal_auth_file:                       Balancing authority and ID codes Excel file
    :type bal_auth_file:                        str

    :return:                                    [0] df_fips: DataFrame of prepared and cleaned FIPS data
                                                [1] df_states:: DataFrame of prepared and cleaned service area data
                                                [2] df_ult:: DataFrame of prepared and cleaned sales data
                                                [2] df_ba:: DataFrame of prepared and cleaned BA data

    """

    # Read in the state and county FIPS code .csv file:
    df_fips = pd.read_csv(fips_file)

    # Read in the raw data from the EIA-861 Excel spreadsheets:
    try:
        df_states = pd.read_excel(service_area_file, sheet_name='Counties')
    except:
        df_states = pd.read_excel(service_area_file, sheet_name='Counties_States')
    df_ult = pd.read_excel(sales_ult_file, sheet_name='States', skiprows=2)
    df_ba = pd.read_excel(bal_auth_file)

    # Strip the word "county" name from full reference and make it consistently lower case:
    df_fips['county_lower'] = df_fips['county_name'].apply(lambda x: x.lower().split(' county')[0])

    # Replace apostrophes:
    df_fips['county_lower'] = df_fips['county_lower'].apply(lambda x: x.lower().split(' parish')[0]).str.replace("'", "")

    # Make the state abbreviations and state names lower case:
    df_fips['state_abbreviation'] = df_fips['state_abbreviation'].str.lower()

    # Make the state names lower case:
    df_states['State'] = df_states['State'].str.lower()

    # Replace apostrophes:
    df_states['county_lower'] = df_states['County'].str.lower().str.replace("'", "")

    # Create a unified <state_abbrev>_<county_lower> key to merge by:
    df_fips['fips_key'] = df_fips['state_abbreviation'] + '_' + df_fips['county_lower']
    df_states['states_key'] = df_states['State'] + '_' + df_states['county_lower']

    # Filter the "df_ult" and "df_ba" dataframes to only the columns we need:
    try:
        df_ult = df_ult[["Utility Number", "Utility Name", "BA_CODE"]].rename(columns={'BA_CODE': 'BA Code'})
    except:
        df_ult = df_ult[["Utility Number", "Utility Name", "BA Code"]]
    df_ba = df_ba[["BA Code", "BA ID", "Balancing Authority Name"]]

    # Output the cleaned dataframes:
    return df_fips, df_states, df_ult, df_ba


def filter_one(df_fips: str, df_states: str, df_ult: str, df_ba: str) -> DataFrame:
    """Join the cleaned EIA-861 datasets where possible using the common key created in the
    'prepare_data' function.

    :param df_fips:             DataFrame of prepared and cleaned FIPS data
    :type df_fips:              DataFrame

    :param df_states:           DataFrame of prepared and cleaned service area data
    :type df_states:            DataFrame

    :param df_ult:              DataFrame of prepared and cleaned sales data
    :type df_ult:               DataFrame

    :param df_ba:               DataFrame of prepared and cleaned BA data
    :type df_ba:                DataFrame

    :return:                    [0] DataFrame of ata with a valid match between df_fips and df_states
                                [1] DataFrame of data without a valid match between df_fips and df_states

    """

    # Merge the 'df_states' and 'df_fips' dataframes based on the common key:
    df_states_fips = pd.merge(left=df_states, right=df_fips, left_on='states_key', right_on='fips_key', how='left')

    # Reassign a single variable as a float:
    df_states_fips['Utility Number'] = df_states_fips['Utility Number'].astype(float)

    # Merge the 'df_states_fips' and 'df_ult' dataframes based on the utility number:
    df_fips_ult = df_states_fips.merge(df_ult, on='Utility Number', how='left')

    # Merge the 'df_fips_ult' and 'df_ba' dataframes based on the BA code:
    df_valid = df_fips_ult.merge(df_ba, left_on='BA Code', right_on='BA Code', how='left')

    # Filter out rows that did not have a valid match:
    df_nan = df_valid.loc[df_valid['county_lower_y'].isna()].copy()

    # Drop the rows that did not have a valid match:
    df_valid = df_valid.loc[~df_valid['county_lower_y'].isna()].copy()

    # Drop unneeded columns from the valid and invalid dataframes:
    df_valid.drop(columns=['county_lower_y', 'Utility Name_y'], inplace=True)
    df_nan.drop(columns=['county_lower_y'], inplace=True)

    # Output the valid and invalid dataframes:
    return df_valid, df_nan


def filter_two(df_fips: DataFrame, df_nan: DataFrame, df_valid: DataFrame) -> DataFrame:
    """Try to match invalid records by cleaning up naming conventions based on the most suitable match.

    :param df_fips:     DataFrame of prepared and cleaned FIPS data
    :type df_fips:      DataFrame

    :param df_nan:      DataFrame of data without a valid match between df_fips and df_states
    :type df_nan:       DataFrame

    :param df_valid:    DataFrame of ata with a valid match between df_fips and df_states
    :type df_valid:     DataFrame

    :return:            [0] df_valid: DataFrame of data with a valid match between df_fips and df_states
                        [1] df_nan_bad: DataFrame of data without a valid match between df_fips and df_states

    """

    # Get keys from states that are in the "df_fips" dataframe that have NaN records:
    nan_keys_fips = df_fips.loc[df_fips['state_abbreviation'].isin(df_nan['State'].unique())]['fips_key'].unique()

    # Extract the possible matches from the "df_nan" dataframe:
    df_nan['matches'] = [[]] * df_nan.shape[0]

    # Loop over the invalid keys and count the matches using the "count_matches" function:
    for index, fips_key in enumerate(nan_keys_fips):
        df_nan['matches'] = df_nan['matches'] + df_nan['states_key'].apply(lambda x: [count_matches(x, fips_key)])

    # Keep only dictionaries that have a non-zero count for a county name present using the "keep_valid" function:
    df_nan['matches'] = df_nan['matches'].apply(keep_valid)

    # Add a new column with the maximum possible county name matches sing the "get_max_count" function (ideally this is 1):
    df_nan['count_of_selected'] = df_nan['matches'].apply(get_max_count)

    # Add the FIPS key to the dataframe using the "find_county" function:
    df_nan['fips_key'] = df_nan['matches'].apply(find_county)

    # Drop columns we no longer need:
    df_nan.drop(columns=['state_name', 'state_abbreviation', 'state_FIPS', 'county_name', 'county_FIPS'], inplace=True)

    # Extract rows that now have a vaild match:
    df_nan_good = df_nan.loc[df_nan['count_of_selected'] == 1].copy()

    # Extract rows that still have an invalid match:
    df_nan_bad = df_nan.loc[df_nan['count_of_selected'] != 1].copy()

    # Merge the valid match dataframes together using the FIPS code:
    mrx = pd.merge(left=df_nan_good, right=df_fips, left_on='fips_key', right_on='fips_key', how='left')

    # Drop columns we no longer need:
    mrx.drop(columns=['count_of_selected', 'matches'], inplace=True)

    # Reorder the columns to match the pre-existing "df_valid" dataframe:
    mrx = mrx[df_valid.columns].copy()

    # Concatenate the two dataframes with valid matches together:
    df_valid = pd.concat([df_valid, mrx])

    # Output the valid and invalid dataframes:
    return df_valid, df_nan_bad


def data_format(df: DataFrame) -> DataFrame:
    """Format the 'df_valid' dataframe to clean up its naming conventions.

    :param df:                  DataFrame of data with a valid match between df_fips and df_states
    :type df:                   DataFrame

    :return:                    DataFrame with renamed columns

    """

    # Establish the column names we want to keep:
    col_names = ['Data Year', 'Utility Number', 'Utility Name_x', 'state_abbreviation', 'state_name',
                 'state_FIPS', 'county_name', 'county_FIPS', 'BA ID', 'BA Code',
                 'Balancing Authority Name']

    # Only keep the columns that are needed:
    df = df[col_names].copy()

    # Rename the columns using a consistent format:
    df.rename(columns={"Data Year": "year",
                       "Utility Number": "utility_number",
                       "Utility Name_x": "utility_name",
                       "state_abbreviation": "state_abbreviation",
                       "state_name": "state_name",
                       "state_FIPS": "state_fips",
                       "county_name": "county_name",
                       "county_FIPS": "county_fips",
                       "BA ID": "ba_number",
                       "BA Code": "ba_abbreviation",
                       "Balancing Authority Name": "ba_name"}, inplace=True)

    # Remove commas from the 'ba_name' variable:
    df['ba_name'] = df['ba_name'].str.replace(",", "")

    # Keep only the variables that are used later in tell:
    df = df[['year',
             'state_fips',
             'state_name',
             'county_fips',
             'county_name',
             'ba_number',
             'ba_abbreviation']].copy(deep=False)

    # Rename the columns to use leading capital letters:
    df.rename(columns={"year": "Year",
                       "state_fips": "State_FIPS",
                       "state_name": "State_Name",
                       "county_fips": "County_FIPS",
                       "county_name": "County_Name",
                       "ba_number": "BA_Number",
                       "ba_abbreviation": "BA_Code"}, inplace=True)

    # Output the dataframe with renamed columns:
    return df


def process_spatial_mapping(target_year: int, fips_file: str, service_area_file: str, sales_ult_file: str, bal_auth_file: str,
                 output_dir: str):
    """Workflow function to execute the mapping of BAs to counties for a given year

    :param target_year:                         Year to process; four digit year (e.g., 1990)
    :type target_year:                          int

    :param fips_file:                           County FIPS code .csv file
    :type fips_file:                            str

    :param service_area_file:                   Balancing authority service area Excel file
    :type service_area_file:                    str

    :param sales_ult_file:                      Balancing authority sales to ultimate customer Excel file
    :type sales_ult_file:                       str

    :param bal_auth_file:                       Balancing authority and ID codes Excel file
    :type bal_auth_file:                        str

    :param output_dir:                          Directory to store the output .csv file
    :type output_dir:                           str

    """

    # Initialize the logger in the "output_dir":
    logger = Logger(output_directory=output_dir)
    logger.initialize_logger()

    # Report the start time:
    logging.info("Start time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))

    # Run the "prepare_data" function:
    logging.info("Preparing data...")
    df_fips, df_states, df_ult, df_ba = prepare_data(fips_file, service_area_file, sales_ult_file, bal_auth_file)

    # Run the "filter_one" function to drop invalid matches:
    logging.info("Applying filter one...")
    df_valid, df_nan = filter_one(df_fips, df_states, df_ult, df_ba)

    # Run the "filter_two" function to try to find new valid matches:
    logging.info("Applying filter two...")
    df_valid, df_nan = filter_two(df_fips, df_nan, df_valid)

    # Format the "df_valid" dataframe to clean up its naming conventions:
    logging.info("Formatting output data...")
    df_valid = data_format(df_valid)

    # Drop the duplicates that result from more than one utility-BA combination per county:
    df_valid = df_valid.drop_duplicates()

    # Drop rows with missing values for the BA:
    df_valid = df_valid.dropna(axis = 0, how = 'any', subset=['BA_Number'])

    # Sort the dataframe by BA number first and then county FIPS code:
    df_valid = df_valid.sort_values(by=["BA_Number", "County_FIPS"])

    # Write the spatial mapping output to a .csv file:
    output_file = os.path.join(output_dir, f'ba_service_territory_{target_year}.csv')
    logging.info(f"Writing output file to:  {output_file}")
    df_valid.to_csv(output_file, sep=',', index=False)

    # Report the end time:
    logging.info("End time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))

    # Close the logger:
    logger.close_logger()


def map_ba_service_territory(start_year: int, end_year: int, data_input_dir: str):
    """Workflow function to run the "process_spatial_mapping" function to map BAs to counties

    :param start_year:                         Year to start process; four digit year (e.g., 1990)
    :type start_year:                          int

    :param end_year:                           Year to end process; four digit year (e.g., 1990)
    :type end_year:                            int

    :param data_input_dir:                     Top-level data directory for TELL
    :type data_input_dir:                      str

    """

    # Set the output directory based on the "raw_data_dir" variable:
    output_dir = os.path.join(data_input_dir, r'outputs', r'ba_service_territory')

    # If the output directory doesn't exist then create it:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a vector of years to process:
    years_to_process = range(start_year, end_year + 1)

    # Loop over the range of years to process:
    for target_year in years_to_process:
        # Set paths to files:
        fips_file = os.path.join(data_input_dir, r'tell_raw_data', 'state_and_county_fips_codes.csv')
        service_area_file = os.path.join(data_input_dir, r'tell_raw_data', r'EIA_861', f'{target_year}', f'Service_Territory_{target_year}.xlsx')
        sales_ult_file = os.path.join(data_input_dir, r'tell_raw_data', r'EIA_861', f'{target_year}', f'Sales_Ult_Cust_{target_year}.xlsx')
        bal_auth_file = os.path.join(data_input_dir, r'tell_raw_data', r'EIA_861', f'{target_year}', f'Balancing_Authority_{target_year}.xlsx')

        # Run the "process_spatial_mapping" function for that year:
        process_spatial_mapping(target_year, fips_file, service_area_file, sales_ult_file, bal_auth_file, output_dir)
