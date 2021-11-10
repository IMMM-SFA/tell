import pandas as pd
import numpy as np
import os
from datetime import datetime

import tell.metadata_eia as metadata_eia

def fips_pop_yearly(pop_input_dir, start_year, end_year):
    """Read in population data, format columns and return single df for all years
    :param pop_input_dir:                      Directory where county population is stored
    :type pop_input_dir:                       dir
    :param start_year:                         Year to start model ; four digit year (e.g., 1990)
    :type start_year:                          int
    :param end_year:                           Year to start model ; four digit year (e.g., 1990)
    :type end_year:                            int
    :return:                                   Dataframe of valid population data for select timeframe
    """

    # get population from merged mapping data
    df_pop = pd.read_csv(pop_input_dir + '/county_populations_2000_to_2019.csv')

    # loop over years for pop data
    df = pd.DataFrame([])
    for y in range(start_year, end_year + 1):
        # only keep columns that are needed
        key = [f'pop_{y}', 'county_FIPS']

        # change pop yr name for later merging
        df_pop_yr = df_pop[key].copy()

        df_pop_yr['year'] = y
        df_pop_yr.rename(columns={f'pop_{y}': 'population'}, inplace=True)

        # combine all years for one dataset
        df = df.append(df_pop_yr)

    return df


def merge_mapping_data(map_input_dir, pop_input_dir, start_year, end_year):
    """Merge the fips county data and population data from FIPS code
     :param map_input_dir:                      Directory where fips county data is stored
     :type map_input_dir:                       dir
     :param pop_input_dir:                      Directory where county population is stored
     :type pop_input_dir:                       dir
     :param start_year:                         Year to start model ; four digit year (e.g., 1990)
     :type start_year:                          int
     :param end_year:                           Year to start model ; four digit year (e.g., 1990)
     :type end_year:                            int
     :return:                                   Dataframe of population data with FIPS and BA name
     """
    # load FIPS county data for BA number and FIPs code matching for later population sum by BA
    df = pd.DataFrame()
    for file in os.listdir(map_input_dir):
        df = df.append(pd.read_csv(os.path.join(map_input_dir, file)), ignore_index=True)

    # only keep columns that are needed
    col_names = ['Year', 'County_FIPS', 'BA_Number']
    df = df[col_names].copy()
    df['BA_Number'] = df['BA_Number'].fillna(0).astype(np.int64)
    df['County_FIPS'] = df['County_FIPS'].fillna(0).astype(np.int64)

    # select for valid (and unique) BA numbers (using metadata_eia.py)
    num = df['BA_Number'].tolist()
    unique_num = np.unique(num).tolist()
    metadata_df = metadata_eia(unique_num)

    # merge mapping df to the the metadata
    df_map = df.merge(metadata_df, on=['BA_Number'])
    df_map.rename(columns={"County_FIPS": "county_FIPS"}, inplace=True)
    df_map.rename(columns={"Year": "year"}, inplace=True)

    # get sum of population by FIPS and merge to mapping file
    df_pop = fips_pop_yearly(pop_input_dir, start_year, end_year)

    df_combine = pd.merge(df_pop, df_map, how='left', left_on=['county_FIPS', 'year'], right_on=['county_FIPS', 'year'])

    return df_combine


def ba_pop_sum(map_input_dir, pop_input_dir, start_year, end_year):
    """Sum the population by BA number and year
     :param mapping_input_dir:                  Directory where fips county data is stored
     :type mapping_input_dir:                   dir
     :param population_input_dir:               Directory where county population is stored
     :type population_input_dir:                dir
     :param start_year:                         Year to start model ; four digit year (e.g., 1990)
     :type start_year:                          int
     :param end_year:                           Year to start model ; four digit year (e.g., 1990)
     :type end_year:                            int
     :return:                                   Dataframe of total population by BA name and year
     """
    # get population from merged mapping data
    df_pop = merge_mapping_data(map_input_dir, pop_input_dir, start_year, end_year)

    # sum population by year
    df = df_pop.groupby(['BA_Name', 'year'])['population'].sum().reset_index()

    return df


def ba_pop_interpolate(map_input_dir, pop_input_dir, start_year, end_year):
    """Interpolate the population from yearly to hourly timeseries to match EIA 930 hourly data
     :param mapping_input_dir:                  Directory where fips county data is stored
     :type mapping_input_dir:                   dir
     :param population_input_dir:               Directory where county population is stored
     :type population_input_dir:                dir
     :param output_dir:                         Directory where to store the hourly population data
     :type output_dir:                          dir
     :param start_year:                         Year to start model ; four digit year (e.g., 1990)
     :type start_year:                          int
     :param end_year:                           Year to start model ; four digit year (e.g., 1990)
     :type end_year:                            int
     :return:                                   Dataframe of hourly population timeseries for each BA name
     """
    df = ba_pop_sum(map_input_dir, pop_input_dir, start_year, end_year)
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    df.rename(columns={"population": "pop"}, inplace=True)
    df.rename(columns={'BA_Name': 'name'}, inplace=True)
    # Reshape
    df = df.pivot(index='name', columns='year', values='pop')

    rng_start = f'{start_year}-01-01'
    rng_end = f'{end_year}-12-31'
    datetime.strptime(rng_start, "%Y-%m-%d")
    datetime.strptime(rng_end, "%Y-%m-%d")
    # Get range of dates to interpolate from
    rng = pd.date_range(rng_start, rng_end, freq='H')

    # Reindex and interpolate with linear interpolation
    df_interp = df.reindex(rng, axis=1).interpolate(axis=1)

    # Make dates rows and BA names columns
    df_interp = df_interp.T

    # Make BA name column
    df_interp.reset_index(level=0, inplace=True)

    # Extract year, month, day ,hour for each date
    df_interp['Year'] = df_interp['index'].dt.strftime('%Y')
    df_interp['Month'] = df_interp['index'].dt.strftime('%m')
    df_interp['Day'] = df_interp['index'].dt.strftime('%d')
    df_interp['Hour'] = df_interp['index'].dt.strftime('%H')

    # Reorder columns and remove datestring
    col = df_interp.pop("Year")
    df_interp.insert(0, col.name, col)

    col = df_interp.pop("Month")
    df_interp.insert(1, col.name, col)

    col = df_interp.pop("Day")
    df_interp.insert(2, col.name, col)

    col = df_interp.pop("Hour")
    df_interp.insert(3, col.name, col)

    df_interp = df_interp.drop('index', 1)




    #f = lambda x: x.to_csv(pop_output_dir + "hourly_pop_{}.csv".format(x.name.lower()), index=False)
    #df.groupby('BA_name').apply(f)
    #df_interp.to_csv(os.path.join(output_dir, f'{BA_name}_hourly_load_data.csv'), index=False, header=True)

    return df_interp


