import os

import pandas as pd
from joblib import Parallel, delayed

from .package_data import get_ba_abbreviations


def list_wrf_files(input_dir, target_yr):
    """Make a list of all the filenames for EIA 930 hourly load data (xlsx)

    :param input_dir:               Directory where EIA 930 hourly load data
    :type input_dir:                str

    :param year:                    Year of which wrf sample data is needed (Zenodo package includes 2019, 2059, 2099)
    :type year:                     int

    :return:                        List of EIA 930 hourly load files by BA short name

    """


    # get a list of BA abbreviations to process
    ba_name = get_ba_abbreviations()

    path_list = []
    for i in ba_name:
        path_to_check = os.path.join(input_dir, f'{i}_WRF_Hourly_Mean_Meteorology_{target_yr}.csv')
        if os.path.isfile(path_to_check) is True:
            path_list.append(path_to_check)

    return path_list

def wrf_data_date(file_string, output_dir):
    """Select wanted columns in each file
    :param file_string:            File name of EIA 930 hourly load data by BA
    :type file_string:             str
    :param output_dir:             Directory to store the EIA 930 hourly load data as a csv
    :type output__dir:             dir
    :return:                       Subsetted dataframe of EIA 930 hourly data
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


def process_wrf(input_dir, output_dir, target_yr, n_jobs=-1):
    """Read in list of EIA 930 files, subset files and save as csv in new file name

    :param input_dir:              Directory where EIA 930 hourly load data
    :type input_dir:               dir

    :param output_dir:             Directory to store the EIA 930 hourly load data as a csv
    :type output__dir:             dir

    :return:                       Subsetted dataframe of EIA 930 hourly data by BA short name

     """
    # run the list function for the EIA files
    list_of_files = list_wrf_files(input_dir, target_yr)

    # run all files in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(wrf_data_date)(
            file_string=i,
            output_dir=output_dir
        ) for i in list_of_files
    )


