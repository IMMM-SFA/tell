import os

import numpy as np
import pandas as pd

from pandas import DataFrame


def process_spatial_mapping(target_year: int, fips_file: str, service_area_file: str, sales_ult_file: str,
                            bal_auth_file: str, output_dir: str):
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

    # Read in the raw data from the "Service_Territory_YYYY.xlsx" spreadsheet:
    try:
        df_territory = pd.read_excel(service_area_file, sheet_name='Counties')
    except:
        df_territory = pd.read_excel(service_area_file, sheet_name='Counties_States')

    # Only keep the columns that are needed:
    df_territory = df_territory[['Utility Number', 'State', 'County', 'Data Year']].copy()

    # Read in the raw data from the "Sales_Ult_Cust_YYYY.xlsx" spreadsheet:
    df_ultcust = pd.read_excel(sales_ult_file, sheet_name='States', skiprows=2)

    # Rename the "BA_CODE" column for EIA-861 data before 2017:
    try:
        df_ultcust = df_ultcust.rename(columns={'BA_CODE': 'BA Code'})
    except:
        df_ultcust = df_ultcust

    # Only keep the columns that are needed:
    df_ultcust = df_ultcust[['Utility Number', 'State', 'BA Code']].copy()

    # Read in the raw data from the "Balancing_Authority_YYYY.xlsx" spreadsheet:
    df_ba = pd.read_excel(bal_auth_file)

    # Rename the "BA_CODE" column for EIA-861 data before 2017:
    try:
        df_ba = df_ba.rename(columns={'BA_CODE': 'BA Code', 'BA ID': 'BA_Number'})
    except:
        df_ba = df_ba

    # Only keep the columns that are needed:
    df_ba = df_ba[['State', 'BA Code', 'BA_Number', 'Balancing Authority Name']].copy()

    # Merge the 'df_territory' and 'df_ultcust' dataframes based on common utility numbers and states:
    df_territory_ultcust = pd.merge(df_territory, df_ultcust, how='left', left_on=['Utility Number', 'State'],
                                    right_on=['Utility Number', 'State'])

    # Merge the 'df_territory_ultcust' and 'df_ba' dataframes based on common BA codes:
    df_territory_ultcust_ba = pd.merge(df_territory_ultcust, df_ba, how='left', left_on=['BA Code', 'State'],
                                       right_on=['BA Code', 'State'])

    # Only keep the columns that are needed:
    df_territory_ultcust_ba = df_territory_ultcust_ba[
        ['Data Year', 'State', 'County', 'BA Code', 'Balancing Authority Name', 'BA_Number']].copy()

    # Drop the duplicates that result from more than one utility-BA combination per county:
    df_territory_ultcust_ba = df_territory_ultcust_ba.drop_duplicates()

    # Drop the rows that have a NaN value for "BA Code":
    df_territory_ultcust_ba = df_territory_ultcust_ba.dropna(subset=['BA Code'])

    # Make the "County" variable consistently lower case:
    df_territory_ultcust_ba['County'] = df_territory_ultcust_ba['County'].apply(lambda x: x.lower())

    # Remove apostrophes from the "County" variable:
    df_territory_ultcust_ba['County'] = df_territory_ultcust_ba['County'].apply(lambda x: x.replace("'", ""))

    # Replace spaces and dashes with underscores in the "County" variable:
    df_territory_ultcust_ba['County'] = df_territory_ultcust_ba['County'].apply(lambda x: x.replace(" ", "_"))
    df_territory_ultcust_ba['County'] = df_territory_ultcust_ba['County'].apply(lambda x: x.replace("-", "_"))

    # Rename one county that has two spaces in the raw EIA-861 data:
    df_territory_ultcust_ba['County'] = df_territory_ultcust_ba['County'].apply(
        lambda x: x.replace("st__helena", "st_helena"))

    # Read in the state and county FIPS code .csv file:
    df_fips = pd.read_csv(fips_file)

    # Make the "county_name" variable consistently lower case and copy it to a new variable:
    df_fips['county_match'] = df_fips['county_name'].apply(lambda x: x.lower())

    # Strip the words "county", "borough", "parish", "municipality", and "census area" from the "county_match" variable:
    df_fips['county_match'] = df_fips['county_match'].apply(lambda x: x.split(' county')[0])
    df_fips['county_match'] = df_fips['county_match'].apply(lambda x: x.split(' borough')[0])
    df_fips['county_match'] = df_fips['county_match'].apply(lambda x: x.split(' parish')[0])
    df_fips['county_match'] = df_fips['county_match'].apply(lambda x: x.split(' municipality')[0])
    df_fips['county_match'] = df_fips['county_match'].apply(lambda x: x.split(' census area')[0])

    # Rename two counties that contain the prefix "De":
    df_fips['county_match'] = df_fips['county_match'].apply(lambda x: x.replace("de soto", "desoto"))
    df_fips['county_match'] = df_fips['county_match'].apply(lambda x: x.replace("de witt", "dewitt"))

    # Remove apostrophes and periods from the "county_match" variable:
    df_fips['county_match'] = df_fips['county_match'].apply(lambda x: x.replace("'", ""))
    df_fips['county_match'] = df_fips['county_match'].apply(lambda x: x.replace(".", ""))

    # Replace spaces and dashes with underscores in the "county_match" variable:
    df_fips['county_match'] = df_fips['county_match'].apply(lambda x: x.replace(" ", "_"))
    df_fips['county_match'] = df_fips['county_match'].apply(lambda x: x.replace("-", "_"))

    # Rename a few columns to match what is in the 'df_territory_ultcust_ba' dataframe:
    df_fips = df_fips.rename(columns={'county_match': 'County', 'state_abbreviation': 'State'})

    # Merge the 'df_territory_ultcust_ba' and 'df_fips' dataframes based on common county names and state abbreviations:
    df_merged = pd.merge(df_territory_ultcust_ba, df_fips, how='left', left_on=['County', 'State'],
                         right_on=['County', 'State'])

    # Only keep the columns that are needed:
    df_output = df_merged[
        ['Data Year', 'state_name', 'state_FIPS', 'county_name', 'county_FIPS', 'BA_Number', 'BA Code']].copy()

    # Rename the columns using a consistent format:
    df_output.rename(columns={"Data Year": "Year",
                              "state_FIPS": "State_FIPS",
                              "state_name": "State_Name",
                              "county_FIPS": "County_FIPS",
                              "county_name": "County_Name",
                              "BA Code": "BA_Code"}, inplace=True)

    # Reorder the columns:
    df_output = df_output[
        ['Year', 'State_FIPS', 'State_Name', 'County_FIPS', 'County_Name', 'BA_Number', 'BA_Code']].copy(deep=False)

    # Sort the dataframe by BA number first and then county FIPS code:
    df_output = df_output.sort_values(by=["BA_Number", "County_FIPS"])

    # Write the spatial mapping output to a .csv file:
    output_file = os.path.join(output_dir, f'ba_service_territory_{target_year}.csv')
    df_output.to_csv(output_file, sep=',', index=False)


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
    output_dir = os.path.join(data_input_dir, r'tell_quickstarter_data', r'outputs', r'ba_service_territory')

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
