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
    metadata = tell.metadata_eia()
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
    df = pd.DataFrame([])
    for y in range(start_year, end_year + 1):

        # sum population by BA
        pop_sum_yr = df_pop.groupby(['BA_Short_Name','year'])['population'].sum().reset_index()

        # combine all years for one dataset
        df = df.append(pop_sum_yr)

    return df



def ba_pop_sum(start_year, end_year):

    ba_names = df_pop['BA_Short_Name'].unique()

    df = pd.DataFrame([])
    for i in ba_names):

        t = df_pop['year']
        y1 = df_pop['population']
        x = pd.DataFrame({'Hours': pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31',
                                                 freq='1H', closed='left')})

        BA_interp = interp1(t, y1, x, 'linear');

    df = BA_interp.append(pop_sum_yr)

    return df

    return df



names = df_pop['name'].unique()
df = pd.DataFrame(columns = ['name', 'function'])

for i in names:
    condition = df_pop['name'].str.match(i) # define condition where name is i
    mini_df = df_pop[condition] # all rows where condition is met
    t = mini_df['year']
    y1 = mini_df['population']
    x = pd.DataFrame({'Hours': pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='1H', closed='left')})

    pop_interp = interp1d(t, y1, x, 'linear')
    new_row = {name: i, function: pop_interp} # make a new row to append
    df = df.append(new_row, ignore_index = True) # append it