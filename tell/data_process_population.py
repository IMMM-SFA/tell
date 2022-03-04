import os

import numpy as np
import pandas as pd

from pandas import DataFrame
from glob import glob
from datetime import datetime
from .metadata_eia import metadata_eia


def fips_pop_yearly(pop_input_dir: str, start_year: int, end_year: int) -> DataFrame:
    """Read in the raw population data, format columns, and return single dataframe for all years

    :param pop_input_dir:               Directory where raw county population data is stored
    :type pop_input_dir:                str

    :param start_year:                  Year to start process; four digit year (e.g., 1990)
    :type start_year:                   int

    :param end_year:                    Year to end process; four digit year (e.g., 1990)
    :type end_year:                     int

    :return:                            DataFrame

    """

    # Read in the raw county-level population .csv file from the U.S. Census Bureau:
    df_pop = pd.read_csv(pop_input_dir + '/county_populations_2000_to_2019.csv')

    # Loop over the range of years defined by the 'start_year' and 'end_year' variables:
    for y in range(start_year, end_year + 1):

        # Only keep columns that are needed:
        key = [f'pop_{y}', 'county_FIPS']

        # Change the variable name for population for the year:
        df_pop_yr = df_pop[key].copy()

        # Assign a new variable to indicate the year:
        df_pop_yr['year'] = y

        # Rename some columns for consistency:
        df_pop_yr.rename(columns={f'pop_{y}': 'population'}, inplace=True)

        # Concatenate all the years into a single dataframe:
        if y == start_year:
            df = df_pop_yr.copy()
        else:
            df = pd.concat([df, df_pop_yr])

    return df


def merge_mapping_data(map_input_dir: str, pop_input_dir: str, start_year: int, end_year: int) -> DataFrame:
    """Merge the BA mapping files and historical population data based on FIPS codes

    :param map_input_dir:               Directory where the BA-to-county mapping is stored
    :type map_input_dir:                str

    :param pop_input_dir:               Directory where raw county population data is stored
    :type pop_input_dir:                str

    :param start_year:                  Year to start process; four digit year (e.g., 1990)
    :type start_year:                   int

    :param end_year:                    Year to end process; four digit year (e.g., 1990)
    :type end_year:                     int

    :return:                                   DataFrame

    """

    # Load in the BA-to-county mapping files produced by the 'spatial_mapping.py' functions:
    for idx, file in enumerate(glob(f'{map_input_dir}/*.csv')):

        # Read in the .csv file:
        dfx = pd.read_csv(os.path.join(map_input_dir, file))

        # Concatenate the BA-to-county mapping files across years:
        if idx == 0:
            df = dfx.copy()
        else:
            df = pd.concat([df, dfx])

    # Only keep the columns that are needed:
    df = df[['Year', 'County_FIPS', 'BA_Number']].copy()

    # Fill in missing values and reassign the variables as integers:
    df['BA_Number'] = df['BA_Number'].fillna(0).astype(np.int64)
    df['County_FIPS'] = df['County_FIPS'].fillna(0).astype(np.int64)

    # Select for valid (and unique) BA numbers using the 'metadata_eia.py' functions:
    num = df['BA_Number'].tolist()
    unique_num = np.unique(num).tolist()
    metadata_df = metadata_eia(unique_num)

    # Merge the mapping dataframe to the the metadata dataframe based on BA number:
    df_map = df.merge(metadata_df, on=['BA_Number'])

    # Rename some columns for consistency:
    df_map.rename(columns={"County_FIPS": "county_FIPS"}, inplace=True)
    df_map.rename(columns={"Year": "year"}, inplace=True)

    # Get sum of population by FIPS code (e.g., counties) using the 'fips_pop_yearly' function:
    df_pop = fips_pop_yearly(pop_input_dir, start_year, end_year)

    # Merge the dataframes based on county FIPS code and year:
    df_combine = pd.merge(df_pop, df_map, how='left', left_on=['county_FIPS', 'year'], right_on=['county_FIPS', 'year'])

    return df_combine


def ba_pop_sum(map_input_dir: str, pop_input_dir: str, start_year: int, end_year: int) -> DataFrame:
    """Sum the total population within a BA's service territory in a given year

    :param map_input_dir:               Directory where the BA-to-county mapping is stored
    :type map_input_dir:                str

    :param pop_input_dir:               Directory where raw county population data is stored
    :type pop_input_dir:                str

    :param start_year:                  Year to start process; four digit year (e.g., 1990)
    :type start_year:                   int

    :param end_year:                    Year to end process; four digit year (e.g., 1990)
    :type end_year:                     int

    :return:                            DataFrame

    """

    # Get population from the 'merge_mapping_data' function:
    df_pop = merge_mapping_data(map_input_dir, pop_input_dir, start_year, end_year)

    # Sum the population for each BA by year:
    df = df_pop.groupby(['BA_Name', 'year'])['population'].sum().reset_index()

    return df


def process_ba_population_data(start_year: int, end_year: int, data_input_dir: str):
    """Calculate a time-series of the total population living with a BAs service territory

    :param start_year:                         Year to start process; four digit year (e.g., 1990)
    :type start_year:                          int

    :param end_year:                           Year to end process; four digit year (e.g., 1990)
    :type end_year:                            int

    :param data_input_dir:                     Top-level data directory for TELL
    :type data_input_dir:                      str

    """

    # Set the output directory based on the "data_input_dir" variable:
    output_dir = os.path.join(data_input_dir, r'outputs', r'historical_population')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set the input directories based on the "data_input_dir" variable:
    map_input_dir = os.path.join(data_input_dir, r'outputs', r'ba_service_territory')
    pop_input_dir = os.path.join(data_input_dir, r'tell_raw_data', r'Population')

    # Sum the populations using the 'ba_pop_sum' function:
    df = ba_pop_sum(map_input_dir, pop_input_dir, start_year, end_year)

    # Convert the year to a datetime variable:
    df['year'] = pd.to_datetime(df['year'], format='%Y')

    # Rename some columns for consistency:
    df.rename(columns={"population": "pop"}, inplace=True)
    df.rename(columns={'BA_Name': 'name'}, inplace=True)

    # Reshape the dataframe so that the interpolation will work:
    df = df.pivot(index='name', columns='year', values='pop')

    # Set the start and end times for the interpolation:
    rng_start = f'{start_year}-01-01'
    rng_end = f'{end_year}-12-31'
    datetime.strptime(rng_start, "%Y-%m-%d")
    datetime.strptime(rng_end, "%Y-%m-%d")

    # Get a range of dates to interpolate to:
    rng = pd.date_range(rng_start, rng_end, freq='H')

    # Reindex the dataframe and linearly interpolate from an annual to an hourly resolution:
    df_interp = df.reindex(rng, axis=1).interpolate(axis=1)

    # Transpose the interpolated dataframe:
    df_interp = df_interp.T

    # Resent the index variable:
    df_interp.reset_index(level=0, inplace=True)

    # Extract the year, month, day, and hour for each date:
    df_interp['Year'] = df_interp['index'].dt.strftime('%Y')
    df_interp['Month'] = df_interp['index'].dt.strftime('%m')
    df_interp['Day'] = df_interp['index'].dt.strftime('%d')
    df_interp['Hour'] = df_interp['index'].dt.strftime('%H')

    # Reorder the columns and remove the datestring variable:
    col = df_interp.pop("Year")
    df_interp.insert(0, col.name, col)

    col = df_interp.pop("Month")
    df_interp.insert(1, col.name, col)

    col = df_interp.pop("Day")
    df_interp.insert(2, col.name, col)

    col = df_interp.pop("Hour")
    df_interp.insert(3, col.name, col)

    # Drop the index variable:
    df_interp = df_interp.drop(columns='index')

    # Get list of BA names from the column headers:
    df_names = df_interp.loc[:, ~df_interp.columns.isin(['Year', 'Month', 'Day', 'Hour'])]
    BA_name = list(df_names)

    # Loop over BA names to write each BA's population time-series to a .csv file:
    for name in BA_name:
        df_interp.to_csv(os.path.join(output_dir, f'{name}_hourly_population_data.csv'),
                         index=False,
                         columns=['Year', 'Month', 'Day', 'Hour', f'{name}'],
                         header=['Year', 'Month', 'Day', 'Hour', 'Total_Population'])