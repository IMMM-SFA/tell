import os
import glob
import pandas as pd


def list_EIA_930_files(input_dir):
    """Make a list of all of the files xlsx in the data_input_dir

    :return:            List of input files to process

    """
    path_to_check = os.path.join(input_dir, '*.xlsx')
    list_of_files = sorted(glob.glob(path_to_check))

    return (list_of_files)


def EIA_data_subset(file_string, output_dir):
    """Select wanted columns in each file

     :return:            Subsetted dataframe

     """
    # read in the Published Hourly Data
    df = pd.read_excel(file_string, sheet_name='Published Hourly Data')

    # use datetime string to get Year, Month, Day, Hour
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

    BA_name = os.path.splitext(os.path.basename(file_string))[0]
    df.to_csv(os.path.join(output_dir, f'{BA_name}_Hourly_Load_Data.csv'), index=None, header=True)


def process_eia_930(input_dir, output_dir):
    """Read in list of EIA 930 files, subset files and save as csv in new file name

    :return:            List of input files to process
    """
    # run the list function for the EIA files
    list_of_files = list_EIA_930_files(input_dir)

    # run the data suset function for EIA files
    for file_string in list_of_files:
        EIA_data_subset(file_string, output_dir)
