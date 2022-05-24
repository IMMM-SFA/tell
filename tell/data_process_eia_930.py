import os

import pandas as pd

from joblib import Parallel, delayed
from .package_data import get_ba_abbreviations


def list_EIA_930_files(data_input_dir: str) -> list:
    """Make a list of all the file names for the EIA-930 hourly load dataset

    :param data_input_dir:         Top-level data directory for TELL
    :type data_input_dir:          str

    :return:                       list

    """

    # Get a list of BA abbreviations to process:
    ba_name = get_ba_abbreviations()

    # Initiate an empty list:
    path_list = []

    # Loop over the list and find the path for each BA in the list:
    for i in ba_name:
        path_to_check = os.path.join(data_input_dir, r'tell_raw_data', r'EIA_930', r'Balancing_Authority', f'{i}.xlsx')
        path_list.append(path_to_check)

    # Return the list:
    return path_list


def eia_data_subset(file_string: str, data_input_dir: str):
    """Extract only the columns TELL needs from the EIA-930 Excel files

    :param file_string:            File name of EIA-930 hourly load data by BA
    :type file_string:             str

    :param data_input_dir:         Top-level data directory for TELL
    :type data_input_dir:          str

    """

    # Set the output directory based on the "data_input_dir" variable:
    output_dir = os.path.join(data_input_dir, r'tell_quickstarter_data', r'outputs', r'historical_ba_load')

    # If the output directory doesn't exist then create it:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read in the data from the "Published Hourly Data" sheet:
    df = pd.read_excel(file_string, sheet_name='Published Hourly Data')

    # Use datetime string to get the year, month, day, and hour:
    df['Year'] = df['UTC time'].dt.strftime('%Y')
    df['Month'] = df['UTC time'].dt.strftime('%m')
    df['Day'] = df['UTC time'].dt.strftime('%d')
    df['Hour'] = df['UTC time'].dt.strftime('%H')

    # Only keep the columns that are needed:
    col_names = ['Year', 'Month', 'Day', 'Hour', 'DF', 'Adjusted D', 'Adjusted NG', 'Adjusted TI']
    df = df[col_names].copy()

    # Rename the columns to add the units to each variable:
    df.rename(columns={"DF": "Forecast_Demand_MWh",
                       "Adjusted D": "Adjusted_Demand_MWh",
                       "Adjusted NG": "Adjusted_Generation_MWh",
                       "Adjusted TI": "Adjusted_Interchange_MWh"}, inplace=True)

    # Extract the BA name from the "file_string" variable:
    BA_name = os.path.splitext(os.path.basename(file_string))[0]

    # Write the output to a .csv file:
    df.to_csv(os.path.join(output_dir, f'{BA_name}_hourly_load_data.csv'), index=False, header=True)


def process_eia_930_data(data_input_dir: str, n_jobs: int):
    """Read in list of EIA 930 files, subset the data, and save the output as a .csv file

    :param data_input_dir:         Top-level data directory for TELL
    :type data_input_dir:          str

    :param n_jobs:                 The maximum number of concurrently running jobs, such as the number of Python
                                   worker processes when backend=”multiprocessing” or the size of the thread-pool
                                   when backend=”threading”. If -1 all CPUs are used. If 1 is given, no parallel
                                   computing code is used at all, which is useful for debugging. For n_jobs
                                   below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs
                                   but one are used. None is a marker for ‘unset’ that will be interpreted as
                                   n_jobs=1 (sequential execution) unless the call is performed under a
                                   parallel_backend context manager that sets another value for n_jobs.
    :type n_jobs:                  int

    """

    # Create the list of EIA-930 Excel files:
    list_of_files = list_EIA_930_files(data_input_dir)

    # Process each file in the list in parallel:
    Parallel(n_jobs=n_jobs)(
        delayed(eia_data_subset)(
            file_string=i,
            data_input_dir=data_input_dir
        ) for i in list_of_files
    )
