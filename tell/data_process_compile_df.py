import os

import pandas as pd
from pandas import DataFrame

from .package_data import get_ba_abbreviations


def compile_data(eia_dir: str, pop_dir: str, wrf_dir: str, target_yr: int, compile_output_dir: str) -> DataFrame:
    """Read in EIA, population, and WRF data and compile into a merged DataFrame

    :param eia_dir:               Directory where modified EIA 930 hourly load data is stored
    :type eia_dir:                str

    :param pop_dir:               Directory where modified county population is stored
    :type pop_dir:                str

    :param wrf_dir:               Directory where modified wrf data is stored
    :type wrf_dir:                str

    :param target_yr:             Target year for analysis
    :type target_yr:              int

    :param compile_output_dir:    Directory where modified wrf data set is stored
    :type compile_output_dir:     str


    :return:                      DataFrame

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
                merged_first = pd.merge(eia_df, pop_df, how='inner', on=['Year', 'Month', 'Day', 'Hour'])

        if os.path.isfile(wrf_path) is True:
            merged = pd.merge(merged_first, wrf_df, how='inner', on=['Year', 'Month', 'Day', 'Hour'])

        # write the merged DataFrame to a csv
        merged.to_csv(os.path.join(compile_output_dir, f'{i}_hourly_compiled_data.csv'), index=False, header=True)

    return merged
