import glob
import pandas as pd
import numpy as np

def concat_data(input_dir):
    """Make a list of all the filenames for EIA 930 hourly load data (xlsx)
    :param input_dir:               Directory where EIA 930 hourly load data
    :type input_dir:                dir
    :return:                        List of EIA 930 hourly load files by BA short name
    """

    list_dfs = []
    for path_file in input_dir.glob('*.csv'):
        df_small = pd.read_csv(path_file)
        list_dfs.append(df_small)

    df = pd.concat(list_dfs, axis=0)

    return df


def compile_data(eia_930_output_dir, pop_output_dir, wrf_output_dir):
    """Read in population data, format columns and return single df for all years
    :param population_input_dir:               Directory where county population is stored
    :type population_input_dir:                dir
    :param start_year:                         Year to start model ; four digit year (e.g., 1990)
    :type start_year:                          int
    :param end_year:                           Year to start model ; four digit year (e.g., 1990)
    :type end_year:                            int
    :return:                                   Dataframe of valid population data for select timeframe
    """

    eia_df = concat_data(eia_930_output_dir)
    pop_df = concat_data(pop_output_dir)
    wrf_df = concat_data(wrf_output_dir)


