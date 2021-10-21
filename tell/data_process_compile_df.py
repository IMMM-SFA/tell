import glob
import pandas as pd
import numpy as np

def list_csv_files(input_dir):
    """Make a list of all the filenames for EIA 930 hourly load data (xlsx)
    :param input_dir:               Directory where EIA 930 hourly load data
    :type input_dir:                dir
    :return:                        List of EIA 930 hourly load files by BA short name
    """

    path_to_check = os.path.join(input_dir, '*.csv')

    return sorted(glob.glob(path_to_check))

def merge_by_ba(file_string, output_dir):
    """Select wanted columns in each file
    :param file_string:            File name of EIA 930 hourly load data by BA
    :type file_string:             str
    :param output_dir:             Directory to store the EIA 930 hourly load data as a csv
    :type output__dir:             dir
    :return:                       Subsetted dataframe of EIA 930 hourly data
     """
    # read in the Published Hourly Data
    df = pd.read_csv(file_string)

    BA_name = os.path.splitext(os.path.basename(file_string))[0]

    df.to_csv(os.path.join(output_dir, f'{BA_name}_compiled_data.csv'), index=False, header=True)

# def concat_data(input_dir):
#     """Make a list of all the filenames for EIA 930 hourly load data (xlsx)
#     :param input_dir:               Directory where EIA 930 hourly load data
#     :type input_dir:                dir
#     :return:                        List of EIA 930 hourly load files by BA short name
#     """
#
#     list_dfs = []
#     for path_file in input_dir.glob('*.csv'):
#         df_small = pd.read_csv(path_file)
#         list_dfs.append(df_small)
#
#
#     df = pd.concat(list_dfs, axis=0)

#    return df




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

    eia_df = list_csv_files(eia_930_output_dir)
    pop_df = list_csv_files(pop_output_dir)
    wrf_df = list_csv_files(wrf_output_dir)





