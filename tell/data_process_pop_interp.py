import glob
import pandas as pd
import numpy as np
import tell
from datetime import date


def fips_pop_yearly(population_input_dir, start_year, end_year):
    """Make a list of all of the files xlsx in the data_input_dir

    :return:            List of input files to process

    """
    # get population from merged mapping data
    df_pop = pd.read_csv(population_input_dir + '/county_populations_2000_to_2019.csv')

    # loop over years to sum population by year
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


def merge_mapping_data(mapping_input_dir, population_input_dir, start_year, end_year):
    """Make a list of all of the files xlsx in the data_input_dir

    :return:            List of input files to process

    """
    # load FIPS county data for BA number and FIPs code matching for later population sum by BA
    all_files = glob.glob(mapping_input_dir + "/*.csv")

    list = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        list.append(df)

    frame = pd.concat(list, axis=0, ignore_index=True)
    col_names = ['year', 'county_fips', 'ba_number']

    # only keep columns that are needed
    frame = frame[col_names].copy()
    frame['ba_number'] = frame['ba_number'].fillna(0).astype(np.int64)
    frame['county_fips'] = frame['county_fips'].fillna(0).astype(np.int64)

    # select for valid BA numbers (from BA metadata)
    metadata = metadata_eia()
    metadata.rename(columns={"EIA_BA_Number": "ba_number"}, inplace=True)

    # merge mapping df to the the metadata
    df_map = frame.merge(metadata, on=['ba_number'])
    df_map.rename(columns={"county_fips": "county_FIPS"}, inplace=True)

    # get sum of population by FIPS and merge to mapping file
    df_pop = fips_pop_yearly(population_input_dir, start_year, end_year)

    df = pd.merge(df_pop, df_map, how='left', left_on=['county_FIPS', 'year'], right_on=['county_FIPS', 'year'])

    return df


def ba_pop_sum(mapping_input_dir, population_input_dir, start_year, end_year):
    """Make a list of all of the files xlsx in the data_input_dir

    :return:            List of input files to process

    """
    # get population from merged mapping data
    df_pop = merge_mapping_data(mapping_input_dir, population_input_dir, start_year, end_year)

    # loop over years to sum population by year
        # sum population by BA
    df = df_pop.groupby(['BA_Short_Name'], ['year'])['population'].sum().reset_index()

    return df


def ba_pop_interpolate(mapping_input_dir, population_input_dir, start_year, end_year):
    df = ba_pop_sum(mapping_input_dir, population_input_dir, start_year, end_year)
    pd.to_datetime(df['year'], format='%Y')
    df.rename(columns={"population": "pop"}, inplace=True)
    df.rename(columns={'BA_Short_Name': 'name'}, inplace=True)
    # Reshape
    df = df.pivot(index='year', columns='name', values='pop')

    # Build an hourly DatetimeIndex
    idx = pd.date_range(df.index.min(), df.index.max(), freq='H')

    # Reindex and interpolate with cubicspline as an example
    res = df.reindex(idx).interpolate('linear')

    return res
