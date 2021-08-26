"""
MATLAB to Python conversion from: Process_Raw_EIA_930_BA_Hourly_Load_Data_into_Matlab_Files.m
MATLAB: Casey D. Burleyson
Python: Casey R. McGrath

This script takes the raw EIA-930 hourly load data by balancing authority
and convert it from .xlsx files into .mat and .csv files. The output
file format is given below. All times are in UTC. Missing values are reported as -9999
in the .csv output files and are reported as NaN in the .mat output files.
This script corresponds to needed functionality 1.2 on this Confluence page:
https://immm-sfa.atlassian.net/wiki/spaces/IP/pages/1732050973/2021-02-22+TELL+Meeting+Notes.
The raw data used as input to this script is stored on PIC at /projects/im3/tell/raw_data/EIA_930/.
"""

import os
import glob
import pandas as pd
import datetime

# set the data input and output directories:
data_input_dir = '//connie-1/im3fs/tell/inputs/raw_data/EIA_930/Balancing_Authority';
csv_data_output_dir = 'C:/Users/mcgr323/projects/tell/BA_hourly_inputs/BA_Hourly_Load';


def list_files(input_list):
    """Make a list of all of the files in the data_input_dir

    :return:            List of input files to process

    """
    #create list of files from EIA 930 BA directory
    path_to_check = os.path.join(input_list, '*.xlsx')
    list_of_files = sorted(glob.glob(path_to_check))

    return list_of_files


def data_subset(file_string):
    """Select wanted columns in each file

     :return:            Subsetted dataframe

     """
    #read in the Published Hourly Data
    df = pd.read_excel(file_string, sheet_name='Published Hourly Data')

    #use datetime string to get Year, Month, Day, Hour
    df['Year'] = df['UTC time'].dt.strftime('%Y')
    df['Month'] = df['UTC time'].dt.strftime('%m')
    df['Day'] = df['UTC time'].dt.strftime('%d')
    df['Hour'] = df['UTC time'].dt.strftime('%d')

    # only keep columns that are needed
    col_names = ['Year', 'Month', 'Day', 'Hour', 'DF', 'Adjusted D', 'Adjusted NG', 'Adjusted TI']
    df = df[col_names].copy()

    # extract date (Year, Month, Day, Hour), 'Forecast_Demand_MWh', 'Adjusted_Demand_MWh', 'Adjusted_Generation_MWh',
    # 'Adjusted_Interchange_MWh'
    df.rename(columns={"DF": "Forecast_Demand_MWh'",
                       "Adjusted D": "Adjusted_Demand_MWh",
                       "Adjusted NG": "Adjusted_Generation_MWh",
                       "Adjusted TI": "Adjusted_Interchange_MWh"}, inplace=True)

    return df


list_of_files = list_files(data_input_dir)
BA_list = []
    for filename in list_of_files:
        main_str = filename.split("\\")[-1]
        main_str = main_str.split("_")[0]  # get the BA name
        BA_list.append(main_str)
    return BA_list

for i in list_of_files:
    data_subset(i)
    output_file = os.path.join(csv_data_output_dir, f'_Hourly_Load_Data.csv')


