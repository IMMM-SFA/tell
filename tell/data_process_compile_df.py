import os

import pandas as pd

from .package_data import get_ba_abbreviations


def compile_data(eia_dir, pop_dir, wrf_dir, target_yr, compile_output_dir):
    """Read in population data, format columns and return single df for all years
    :param population_input_dir:               Directory where county population is stored
    :type population_input_dir:                dir
    :param start_year:                         Year to start model ; four digit year (e.g., 1990)
    :type start_year:                          int
    :param end_year:                           Year to start model ; four digit year (e.g., 1990)
    :type end_year:                            int
    :return:                                   Dataframe of valid population data for select timeframe
    """


    # get a list of BA abbreviations to process
    ba_name = get_ba_abbreviations()

    for i in ba_name:
        # get the paths for th EIA, population and WRF data
        eia_path = os.path.join(eia_dir, f"{i}_hourly_load_data.csv")
        pop_path = os.path.join(pop_dir, f"{i}_hourly_population.csv")
        wrf_path = os.path.join(wrf_dir, f"{i}_WRF_Hourly_Mean_Meteorology_{target_yr}_hourly_wrf_data.csv")

        if os.path.isfile(eia_path) is True:
            eia_df = pd.read_csv(eia_path)
        if os.path.isfile(pop_path) is True:
            pop_df = pd.read_csv(pop_path)
        if os.path.isfile(wrf_path) is True:
            wrf_df = pd.read_csv(wrf_path)

        # merge the EIA 930, population and WRF data by date
        if os.path.isfile(eia_path) is True:
            if os.path.isfile(pop_path) is True:
                merged_first = pd.merge(eia_df, pop_df, how='inner', on =['Year', 'Month', 'Day', 'Hour'])

        if os.path.isfile(wrf_path) is True:
            merged = pd.merge(merged_first, wrf_df, how='inner', on=['Year', 'Month', 'Day', 'Hour'])

        # write the merged dataframe to a csv
        merged.to_csv(os.path.join(compile_output_dir, f'{i}_hourly_compiled_data.csv'), index=False, header=True)

    return merged
