import os

import pandas as pd

from joblib import Parallel, delayed
from pandas import DataFrame
from .package_data import get_ba_abbreviations


def list_wrf_files(input_dir: str, target_year: int) -> list:
    """Make a list of all the file names for WRF data

    :param input_dir:               Directory where WRF data is stored from im3_components library (wrf_to_tell)
    :type input_dir:                str

    :param target_year:             Year of which wrf sample data is needed (Zenodo package includes 2019, 2059, 2099)
    :type target_year:              int

    :return:                        list

    """

    # get a list of BA abbreviations to process
    ba_name = get_ba_abbreviations()

    path_list = []
    for i in ba_name:
        path_to_check = os.path.join(input_dir, f'{i}_WRF_Hourly_Mean_Meteorology_{target_year}.csv')
        if os.path.isfile(path_to_check) is True:
            path_list.append(path_to_check)

    return path_list


def wrf_data_date(file_string: str, output_dir: str):
    """Select wanted columns in each file

    :param file_string:           File name of WRF data frame from list
    :type file_string:            str

    :param output_dir:            Directory to store the modified WRF data
    :type output_dir:             str

    :return:                      DataFrame

     """
    # read in the Published Hourly Data
    df = pd.read_csv(file_string)

    df['Time_UTC'] = pd.to_datetime(df['Time_UTC'])
    # use datetime string to get Year, Month, Day, Hour
    df['Year'] = df['Time_UTC'].dt.strftime('%Y')
    df['Month'] = df['Time_UTC'].dt.strftime('%m')
    df['Day'] = df['Time_UTC'].dt.strftime('%d')
    df['Hour'] = df['Time_UTC'].dt.strftime('%H')

    # only keep columns that are needed
    col_names = ['Year', 'Month', 'Day', 'Hour', 'T2', 'Q2', 'SWDOWN', 'GLW', 'WSPD']
    df = df[col_names].copy()

    # write to csv
    BA_name = os.path.splitext(os.path.basename(file_string))[0]
    df.to_csv(os.path.join(output_dir, f'{BA_name}_hourly_wrf_data.csv'), index=False, header=True)


def process_wrf(data_input_dir: str, n_jobs: int, process_historical=False, process_future=False):
    """Process the sample weather dataset into cleaned .csv files for input to the MLP model training

    :param process_historical:    If 'process_historical' = True process the historical weather data
    :type process_historical:     bool

    :param process_future:        If 'process_future' = True process the future weather data
    :type process_future:         bool

    :param data_input_dir:        Top-level data directory for TELL
    :type data_input_dir:         str

    :param n_jobs:                Number of jobs to process
    :type n_jobs:                 int

    """
    if process_historical:
       input_dir = os.path.join(data_input_dir, r'sample_weather_data', r'historical_weather')
    elif process_future:
       input_dir = os.path.join(data_input_dir, r'sample_weather_data', r'future_weather')

    print(input_dir)

    # Create a list of all of the WRF output files in the 'input_dir':
    # list_of_files = list_wrf_files(input_dir)

    # Process the list of WRF output files in parallel using the
    #Parallel(n_jobs=n_jobs)(
        #delayed(wrf_data_date)(
            #file_string=i,
            #output_dir=output_dir
        #) for i in list_of_files
    #)
