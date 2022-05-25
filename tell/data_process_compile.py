import os

import pandas as pd
import numpy as np

from .package_data import get_ba_abbreviations


def compile_data(start_year: int, end_year: int, data_input_dir: str):
    """Merge the load, population, and climate data into a single .csv file for each BA

    :param start_year:                         Year to start process; four digit year (e.g., 1990)
    :type start_year:                          int

    :param end_year:                           Year to end process; four digit year (e.g., 1990)
    :type end_year:                            int

    :param data_input_dir:                     Top-level data directory for TELL
    :type data_input_dir:                      str

    """

    # Get a list of BA abbreviations to process:
    ba_name = get_ba_abbreviations()

    # Set the input directories for each variable:
    load_dir = os.path.join(data_input_dir, r'tell_quickstarter_data', r'outputs', r'historical_ba_load')
    population_dir = os.path.join(data_input_dir, r'tell_quickstarter_data', r'outputs', r'historical_population')
    weather_dir = os.path.join(data_input_dir, r'sample_forcing_data', r'historical_weather')

    # Set the output directory based on the "data_input_dir" variable:
    output_dir = os.path.join(data_input_dir, r'tell_quickstarter_data', r'outputs', r'compiled_historical_data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over the list of BAs to process:
    for i in ba_name:

        # Check to make sure all of the requisite data exist for that BA:
        all_data_present = False
        if os.path.isfile(os.path.join(load_dir, f"{i}_hourly_load_data.csv")) is True:
            if os.path.isfile(os.path.join(population_dir, f"{i}_hourly_population_data.csv")) is True:
                if os.path.isfile(os.path.join(weather_dir, f"{i}_WRF_Hourly_Mean_Meteorology_2019.csv")) is True:
                    all_data_present = True

        if all_data_present is True:
            # Read in the historical load and population data for that BA:
            load_df = pd.read_csv(os.path.join(load_dir, f"{i}_hourly_load_data.csv"))
            population_df = pd.read_csv(os.path.join(population_dir, f"{i}_hourly_population_data.csv"))

            # Loop over the range of years defined by the 'start_year' and 'end_year' variables:
            for year in range(start_year, end_year + 1):
                # Read in the annual historical weather for that BA:
                temp_weather_df = pd.read_csv(os.path.join(weather_dir, f"{i}_WRF_Hourly_Mean_Meteorology_{year}.csv"))

                # Convert the time stamp to a datetime variable and then extract the year, month, day, and hour variables:
                temp_weather_df['Time_UTC'] = pd.to_datetime(temp_weather_df['Time_UTC'])
                temp_weather_df['Year'] = temp_weather_df['Time_UTC'].dt.strftime('%Y').astype(np.int64)
                temp_weather_df['Month'] = temp_weather_df['Time_UTC'].dt.strftime('%m').astype(np.int64)
                temp_weather_df['Day'] = temp_weather_df['Time_UTC'].dt.strftime('%d').astype(np.int64)
                temp_weather_df['Hour'] = temp_weather_df['Time_UTC'].dt.strftime('%H').astype(np.int64)

                # Only keep the columns that are needed:
                temp_weather_df = temp_weather_df[['Year', 'Month', 'Day', 'Hour', 'T2', 'Q2', 'SWDOWN', 'GLW', 'WSPD']].copy()

                # Concatenate all the years into a single dataframe:
                if year == start_year:
                    weather_df = temp_weather_df.copy()
                else:
                    weather_df = pd.concat([weather_df, temp_weather_df])

            # Merge the historical load and population dataframes together by date:
            merged_first = pd.merge(load_df, population_df, how='inner', on=['Year', 'Month', 'Day', 'Hour'])

            # Merge in the historical weather by date:
            merged_second = pd.merge(merged_first, weather_df, how='inner', on=['Year', 'Month', 'Day', 'Hour'])

            # Round the population to 2 decimal places:
            merged_second['Total_Population'] = merged_second['Total_Population'].round(2)

            # Write the merged dataframe to a .csv file
            merged_second.to_csv(os.path.join(output_dir, f"{i}_historical_data.csv"), index=False, header=True)

            # Clean up the variables and move to the next BA in the loop:
            del temp_weather_df, weather_df, load_df, population_df, merged_first, merged_second, all_data_present