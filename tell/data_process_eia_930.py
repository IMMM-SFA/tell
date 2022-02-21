import os

import pandas as pd
from pandas import DataFrame
from joblib import Parallel, delayed


from .package_data import get_ba_abbreviations


def list_EIA_930_files(input_dir: str) -> list:
    """Make a list of all the file names for EIA 930 hourly load data

    :param input_dir:               Directory where raw EIA 930 hourly load data is stored
    :type input_dir:                str

    :return:                        list

    """

    # get a list of BA abbreviations to process
    ba_name = get_ba_abbreviations()

    path_list = []
    for i in ba_name:
        path_to_check = os.path.join(input_dir, f'{i}.xlsx')
        path_list.append(path_to_check)

    return path_list


def eia_data_subset(file_string: str, output_dir: str):
    """Select wanted columns in each file

    :param file_string:            File name of EIA 930 hourly load data by BA
    :type file_string:             str

    :param output_dir:             Directory to store the modified EIA 930 hourly load data
    :type output_dir:              str

    :return:                       DataFrame

     """
    # read in the Published Hourly Data
    df = pd.read_excel(file_string, sheet_name='Published Hourly Data')

    # use datetime string to get Year, Month, Day, Hour
    df['Year'] = df['UTC time'].dt.strftime('%Y')
    df['Month'] = df['UTC time'].dt.strftime('%m')
    df['Day'] = df['UTC time'].dt.strftime('%d')
    df['Hour'] = df['UTC time'].dt.strftime('%H')

    # only keep columns that are needed
    col_names = ['Year', 'Month', 'Day', 'Hour', 'DF', 'Adjusted D', 'Adjusted NG', 'Adjusted TI']
    df = df[col_names].copy()

    # extract date (Year, Month, Day, Hour), 'Forecast_Demand_MWh', 'Adjusted_Demand_MWh', 'Adjusted_Generation_MWh',
    # 'Adjusted_Interchange_MWh'
    df.rename(columns={"DF": "Forecast_Demand_MWh'",
                       "Adjusted D": "Adjusted_Demand_MWh",
                       "Adjusted NG": "Adjusted_Generation_MWh",
                       "Adjusted TI": "Adjusted_Interchange_MWh"}, inplace=True)

    BA_name = os.path.splitext(os.path.basename(file_string))[0]
    df.to_csv(os.path.join(output_dir, f'{BA_name}_hourly_load_data.csv'), index=False, header=True)


def process_eia_930(input_dir: str, output_dir: str, n_jobs: int):
    """Read in list of EIA 930 files, subset files and save as csv in new file name

    :param input_dir:              Directory where raw EIA 930 hourly load data is stored
    :type input_dir:               DataFrame

    :param output_dir:             Directory to store the modified EIA 930 hourly load data
    :type output_dir:              DataFrame

    :param n_jobs:                 Number of jobs to process
    :type n_jobs:                  int

    :return:                       DataFrame

     """
    # run the list function for the EIA files
    list_of_files = list_EIA_930_files(input_dir)

    # run all files in parallel
    Parallel(n_jobs=n_jobs)(
        delayed(eia_data_subset)(
            file_string=i,
            output_dir=output_dir
        ) for i in list_of_files
    )
