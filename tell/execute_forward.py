import os
import glob
import datetime

import numpy as np
import pandas as pd

from pandas import DataFrame
from scipy import interpolate
from .states_fips_function import state_metadata_from_state_abbreviation


def extract_gcam_usa_loads(scenario_to_process: str, gcam_usa_input_dir:str) -> DataFrame:
    """Extracts the state-level annual loads from a GCAM-USA output file.

    :param scenario_to_process: Scenario to process
    :type scenario_to_process:  str

    :param gcam_usa_input_dir:  Path to where the GCAM-USA data are stored
    :type gcam_usa_input_dir:   str

    :return:                    DataFrame of state-level annual total electricity loads

    """

    # Cluge the scenario for historical runs:
    if scenario_to_process == 'historic':
        scenario_to_process_gcam = 'rcp45cooler_ssp3'
    else:
        scenario_to_process_gcam = scenario_to_process

    # Create the filename for the needed GCAM run:
    filename = (os.path.join(gcam_usa_input_dir, ('electricity_demand_' + scenario_to_process_gcam + '.csv')))

    # Load in the raw GCAM-USA output file:
    gcam_usa_df = pd.read_csv(filename, index_col=None, header=0)

    # Make a list of all of the states in the "gcam_usa_df":
    states = gcam_usa_df['subRegion'].unique()

    # Loop over the states and interpolate their loads to an annual time step:
    for i in range(len(states)):

        # Subset to just the data for the state being processed:
        subset_df = gcam_usa_df[gcam_usa_df['subRegion'].isin([states[i]])].copy()

        # Retrieve the state metadata:
        (state_fips, state_name) = state_metadata_from_state_abbreviation(states[i])

        # Linearly interpolate the 5-year loads from GCAM-USA to an annual time step:
        annual_time_vector = pd.Series(range(subset_df['x'].min(), subset_df['x'].max()))
        interpolation_function = interpolate.interp1d(subset_df['x'], subset_df['value'], kind='linear')
        annual_loads = interpolation_function(annual_time_vector)

        # Create an empty dataframe and store the results:
        state_df = pd.DataFrame()
        state_df['Year'] = annual_time_vector.tolist()
        state_df['GCAM_USA_State_Annual_Load_TWh'] = annual_loads
        state_df['State_FIPS'] = state_fips
        state_df['State_Name'] = state_name
        state_df['State_Abbreviation'] = states[i]

        # Aggregate the output into a new dataframe:
        if i == 0:
            gcam_usa_output_df = state_df
        else:
            gcam_usa_output_df = pd.concat([gcam_usa_output_df, state_df])

    return gcam_usa_output_df


def process_population_scenario(scenario_to_process: str, population_data_input_dir: str) -> DataFrame:
    """Read in a future population file and interpolate the data to an annual resolution.

    :param scenario_to_process:         Scenario to process
    :type scenario_to_process:          str

    :param population_data_input_dir:   Path to where the sample population projections are stored
    :type population_data_input_dir:    str

    :return:                            DataFrame of population projections at an annual resolution

    """

    if scenario_to_process == 'historic':
        # Read in the raw county-level population .csv file from the U.S. Census Bureau:
        df_pop = pd.read_csv(population_data_input_dir + '/county_populations_2000_to_2020.csv')

        # Loop over the range of years defined by the 'start_year' and 'end_year' variables:
        for y in range(2000, 2020 + 1):

            # Only keep columns that are needed:
            key = [f'pop_{y}', 'county_FIPS']

            # Change the variable name for population for the year:
            df_pop_yr = df_pop[key].copy()

            # Assign a new variable to indicate the year:
            df_pop_yr['year'] = y

            # Rename some columns for consistency:
            df_pop_yr.rename(columns={f'pop_{y}': 'population'}, inplace=True)

            # Concatenate all the years into a single dataframe:
            if y == 2000:
                population_interp_df = df_pop_yr.copy()
            else:
                population_interp_df = pd.concat([population_interp_df, df_pop_yr])

        # Rename the some variables for consistency:
        population_interp_df.rename(columns={'population': 'Population',
                                             'year': 'Year',
                                             'county_FIPS': 'County_FIPS'}, inplace=True)

        # Reorder the columns, convert the FIPS values from strings to integers, round the population projections
        # to whole numbers, and sort the dataframe by FIPS code then year:
        population_interp_df = population_interp_df[['County_FIPS', 'Year', 'Population']]
        population_interp_df['County_FIPS'] = population_interp_df['County_FIPS'].astype(int)
        population_interp_df['Population'] = population_interp_df['Population'].round(0).astype(int)
        population_interp_df = population_interp_df.sort_values(['County_FIPS', 'Year'])

    else:
        if scenario_to_process == 'rcp45cooler_ssp3':
            scenario_string = 'ssp3'
        if scenario_to_process == 'rcp45cooler_ssp5':
            scenario_string = 'ssp5'
        if scenario_to_process == 'rcp45hotter_ssp3':
            scenario_string = 'ssp3'
        if scenario_to_process == 'rcp45hotter_ssp5':
            scenario_string = 'ssp5'
        if scenario_to_process == 'rcp85cooler_ssp3':
            scenario_string = 'ssp3'
        if scenario_to_process == 'rcp85cooler_ssp5':
            scenario_string = 'ssp5'
        if scenario_to_process == 'rcp85hotter_ssp3':
            scenario_string = 'ssp3'
        if scenario_to_process == 'rcp85hotter_ssp5':
            scenario_string = 'ssp5'

        # Read in the raw file:
        population_df = pd.read_csv(os.path.join(population_data_input_dir, f"{scenario_string}_county_population.csv"),
                                    dtype={'FIPS': str})

        # Drop the 'state_name' column and rename the "FIPS" column:
        population_df.drop(columns=['state_name'], inplace=True)
        population_df.rename(columns={'FIPS': 'County_FIPS'}, inplace=True)

        # Set county FIPS code as the index variable:
        population_df.set_index('County_FIPS', inplace=True)

        # Transpose the dataframe:
        population_dft = population_df.T

        # Bring the index back into the dataframe:
        population_dft.reset_index(inplace=True)

        # Rename the index column as "yr":
        population_dft.rename(columns={'index': 'yr'}, inplace=True)

        # Convert the year to a datetime variable:
        population_dft['yr'] = pd.to_datetime(population_dft['yr'])

        # Set the year as the index variable:
        population_dft.set_index('yr', inplace=True)

        # Interpolate the populations to an annual time-step and transpose the results:
        population_interp_df = population_dft.resample('1YE').mean().interpolate('linear').T

        # Convert the dataframe from a wide format to a long format and name the population variable:
        population_interp_df = population_interp_df.stack().reset_index()
        population_interp_df.rename(columns={0: 'Population'}, inplace=True)

        # Change the time variable to only the year value:
        population_interp_df['Year'] = population_interp_df['yr'].dt.year
        population_interp_df.drop(columns=['yr'], inplace=True)

        # Reorder the columns, convert the FIPS values from strings to integers, round the population projections
        # to whole numbers, and sort the dataframe by FIPS code then year:
        population_interp_df = population_interp_df[['County_FIPS', 'Year', 'Population']]
        population_interp_df['County_FIPS'] = population_interp_df['County_FIPS'].astype(int)
        population_interp_df['Population'] = population_interp_df['Population'].round(0).astype(int)
        population_interp_df = population_interp_df.sort_values(['County_FIPS', 'Year'])

    return population_interp_df


def aggregate_mlp_output_files(list_of_files: list) -> DataFrame:
    """Aggregates a list of MLP output files into a dataframe.

    :param list_of_files:           List of MLP output files
    :type list_of_files:            list

    :return:                        DataFrame of all MLP output concatenated together

    """

    # Loop over the list of MLP output files:
    for file in range(len(list_of_files)):

        # Read in the .csv file and replace missing values with nan:
        mlp_data = pd.read_csv(list_of_files[file]).replace(-9999, np.nan)

        # Rename the "BA" variable:
        mlp_data.rename(columns={'BA': 'BA_Code'}, inplace=True)

        # Rename the "Load" variable:
        mlp_data.rename(columns={'Load': 'Total_BA_Load_MWh'}, inplace=True)

        # Replacing missing or negative loads with NaN:
        mlp_data.loc[~(mlp_data['Total_BA_Load_MWh'] > 0), 'Total_BA_Load_MWh'] = np.nan

        # Aggregate the output into a new dataframe:
        if file == 0:
            mlp_output_df = mlp_data
        else:
            mlp_output_df = pd.concat([mlp_output_df, mlp_data])
    return mlp_output_df


def output_tell_summary_data(joint_mlp_df: DataFrame, year_to_process: str, gcam_target_year: str, data_output_dir: str) -> DataFrame:
    """Writes a summary file describing state-level annual total loads from TELL and GCAM-USA.

    :param joint_mlp_df:            DataFrame of processed TELL loads
    :type joint_mlp_df:             DataFrame

    :param year_to_process:         Year to process
    :type year_to_process:          str

    :param gcam_target_year:        Year to scale against the GCAM-USA annual loads
    :type gcam_target_year:         str

    :param data_output_dir:         Data output directory
    :type data_output_dir:          str

    :return:                        DataFrame of summary statistics

    """

    # Make a copy of the necessary variables, drop the duplicates, and add in the "year_to_process":
    output_df = joint_mlp_df[
        ['State_FIPS', 'State_Name', 'TELL_State_Annual_Load_TWh', 'GCAM_USA_State_Annual_Load_TWh',
         'State_Scaling_Factor']].copy(deep=False)

    output_df = output_df.drop_duplicates()

    output_df['Year'] = year_to_process

    # Rename the columns to make them more readable:
    output_df.rename(columns={'TELL_State_Annual_Load_TWh': 'Raw_TELL_Load_TWh',
                              'GCAM_USA_State_Annual_Load_TWh': 'GCAM_USA_Load_TWh',
                              'State_Scaling_Factor': 'Scaling_Factor'}, inplace=True)

    # Calculate the scaled TELL loads:
    output_df['Scaled_TELL_Load_TWh'] = output_df['Raw_TELL_Load_TWh'].mul(output_df['Scaling_Factor'])

    # Round off the values to make the output file more readable:
    output_df['State_FIPS'] = output_df['State_FIPS'].round(0)
    output_df['Raw_TELL_Load_TWh'] = output_df['Raw_TELL_Load_TWh'].round(5)
    output_df['GCAM_USA_Load_TWh'] = output_df['GCAM_USA_Load_TWh'].round(5)
    output_df['Scaled_TELL_Load_TWh'] = output_df['Scaled_TELL_Load_TWh'].round(5)
    output_df['Scaling_Factor'] = output_df['Scaling_Factor'].round(5)

    # Reorder the columns, fill in missing values, and sort alphabetically by state name:
    output_df = output_df[
        ['Year', 'State_Name', 'State_FIPS', 'Scaling_Factor', 'GCAM_USA_Load_TWh', 'Raw_TELL_Load_TWh',
         'Scaled_TELL_Load_TWh']]
    output_df = output_df.fillna(-9999)
    output_df = output_df.sort_values('State_Name')

    # Generate the .csv output file name:
    csv_output_filename = os.path.join(data_output_dir, 'TELL_State_Summary_Data_' + year_to_process
                                       + '_Scaled_' + gcam_target_year + '.csv')

    # Write out the dataframe to a .csv file:
    output_df.to_csv(csv_output_filename, sep=',', index=False)

    return output_df


def output_tell_ba_data(joint_mlp_df: DataFrame, year_to_process: str, gcam_target_year: str, data_output_dir: str) -> DataFrame:
    """Writes a file of the time-series of hourly loads for each BA.

    :param joint_mlp_df:            DataFrame of processed TELL loads
    :type joint_mlp_df:             DataFrame

    :param year_to_process:         Year to process
    :type year_to_process:          str

    :param gcam_target_year:        Year to scale against the GCAM-USA annual loads
    :type gcam_target_year:         str

    :param data_output_dir:         Data output directory
    :type data_output_dir:          str

    :return:                        DataFrame of BA-level total load time-series

    """

    # Make a copy of the necessary variables:
    ba_output_df = joint_mlp_df[
        ['Time_UTC', 'BA_Code', 'BA_Number', 'County_BA_Load_MWh', 'County_BA_Load_MWh_Scaled']].copy(deep=False)

    # Make a list of all of the BAs in "ba_output_df":
    bas = ba_output_df['BA_Code'].unique()

    # Loop over the BAs and process their data into an output file:
    for i in range(len(bas)):

        # Subset to just the data for the BA being processed:
        output_df = ba_output_df[ba_output_df['BA_Code'].isin([bas[i]])].copy()

        # Sum the county loads as a function of time:
        output_df['Raw_TELL_BA_Load_MWh'] = output_df.groupby('Time_UTC')['County_BA_Load_MWh'].transform('sum')
        output_df['Scaled_TELL_BA_Load_MWh'] = output_df.groupby('Time_UTC')['County_BA_Load_MWh_Scaled'].transform('sum')

        # Subset and reorder the columns, drop duplicates, fill in missing values, and sort chronologically:
        output_df = output_df[['BA_Code', 'BA_Number', 'Time_UTC', 'Raw_TELL_BA_Load_MWh', 'Scaled_TELL_BA_Load_MWh']]
        output_df = output_df.drop_duplicates()
        output_df = output_df.fillna(-9999)
        output_df = output_df.sort_values("Time_UTC")

        # Round off the values to make the output file more readable:
        output_df['BA_Number'] = output_df['BA_Number'].round(0)
        output_df['Raw_TELL_BA_Load_MWh'] = output_df['Raw_TELL_BA_Load_MWh'].round(2)
        output_df['Scaled_TELL_BA_Load_MWh'] = output_df['Scaled_TELL_BA_Load_MWh'].round(2)

        # Aggregate the output into a new dataframe:
        if i == 0:
            aggregate_output_df = output_df
        else:
            aggregate_output_df = pd.concat([aggregate_output_df, output_df])


    # Sort the data alphabetically by BA:
    aggregate_output_df = aggregate_output_df.sort_values(by=["BA_Code", "Time_UTC"])

    # Generate the .csv output file name:
    csv_output_filename = os.path.join(data_output_dir,
                                       'TELL_Balancing_Authority_Hourly_Load_Data_' + year_to_process
                                       + '_Scaled_' + gcam_target_year + '.csv')

    # Write out the dataframe to a .csv file:
    aggregate_output_df.to_csv(csv_output_filename, sep=',', index=False)

    return aggregate_output_df


def output_tell_state_data(joint_mlp_df: DataFrame, year_to_process: str, gcam_target_year: str, data_output_dir: str) -> DataFrame:
    """Writes a file of the time-series of hourly loads for each state.

    :param joint_mlp_df:            DataFrame of processed TELL loads
    :type joint_mlp_df:             DataFrame

    :param year_to_process:         Year to process
    :type year_to_process:          str

    :param gcam_target_year:        Year to scale against the GCAM-USA annual loads
    :type gcam_target_year:         str

    :param data_output_dir:         Data output directory
    :type data_output_dir:          str

    :return:                        DataFrame of state-level total load time-series

    """

    # Make a copy of the necessary variables:
    state_output_df = joint_mlp_df[
        ['Time_UTC', 'State_FIPS', 'State_Name', 'State_Scaling_Factor', 'County_BA_Load_MWh',
         'County_BA_Load_MWh_Scaled']].copy(deep=False)

    # Make a list of all of the states in "state_output_df":
    states = state_output_df['State_Name'].unique()

    # Loop over the states and process their data:
    for i in range(len(states)):

        # Subset to just the data for the state being processed:
        output_df = state_output_df[state_output_df['State_Name'].isin([states[i]])].copy()

        # Sum the county loads as a function of time:
        output_df['Raw_TELL_State_Load_MWh'] = output_df.groupby('Time_UTC')['County_BA_Load_MWh'].transform('sum')
        output_df['Scaled_TELL_State_Load_MWh'] = output_df.groupby('Time_UTC')['County_BA_Load_MWh_Scaled'].transform(
            'sum')

        # Subset and reorder the columns, drop duplicates, fill in missing values, and sort chronologically:
        output_df = output_df[
            ['State_Name', 'State_FIPS', 'Time_UTC', 'Raw_TELL_State_Load_MWh', 'Scaled_TELL_State_Load_MWh']]
        output_df = output_df.drop_duplicates()
        output_df = output_df.fillna(-9999)
        output_df = output_df.sort_values('Time_UTC')

        # Round off the values to make the output file more readable:
        output_df['State_FIPS'] = output_df['State_FIPS'].round(0)
        output_df['Raw_TELL_State_Load_MWh'] = output_df['Raw_TELL_State_Load_MWh'].round(2)
        output_df['Scaled_TELL_State_Load_MWh'] = output_df['Scaled_TELL_State_Load_MWh'].round(2)

        # Aggregate the output into a new dataframe:
        if i == 0:
            aggregate_output_df = output_df
        else:
            aggregate_output_df = pd.concat([aggregate_output_df, output_df])

    # Sort the data alphabetically by state:
    aggregate_output_df = aggregate_output_df.sort_values(by=['State_Name', 'Time_UTC'])

    # Generate the .csv output file name:
    csv_output_filename = os.path.join(data_output_dir, 'TELL_State_Hourly_Load_Data_' + year_to_process
                                       + '_Scaled_' + gcam_target_year + '.csv')

    # Write out the dataframe to a .csv file:
    aggregate_output_df.to_csv(csv_output_filename, sep=',', index=False)

    return aggregate_output_df


def output_tell_county_data(joint_mlp_df: DataFrame, year_to_process: str, gcam_target_year: str, data_output_dir: str):
    """Writes a file of the time-series of hourly loads for each county.

    :param joint_mlp_df:            DataFrame of processed TELL loads
    :type joint_mlp_df:             DataFrame

    :param year_to_process:         Year to process
    :type year_to_process:          str

    :param gcam_target_year:        Year to scale against the GCAM-USA annual loads
    :type gcam_target_year:         str

    :param data_output_dir:         Data output directory
    :type data_output_dir:          str

    """

    # Make a copy of the necessary variables:
    county_output_df = joint_mlp_df[
        ['Time_UTC', 'County_FIPS', 'County_Name', 'State_Name', 'State_FIPS', 'County_BA_Load_MWh',
         'County_BA_Load_MWh_Scaled']].copy(deep=False)

    # Make a list of all of the counties in "county_output_df":
    counties = county_output_df['County_FIPS'].unique()

    # Loop over the counties and process their data:
    for i in range(len(counties)):

        # Subset to just the data for the county being processed:
        output_df = county_output_df[county_output_df['County_FIPS'] == counties[i]].copy()

        # Sum the county loads as a function of time:
        output_df['Raw_TELL_County_Load_MWh'] = output_df.groupby('Time_UTC')['County_BA_Load_MWh'].transform('sum')
        output_df['Scaled_TELL_County_Load_MWh'] = output_df.groupby('Time_UTC')['County_BA_Load_MWh_Scaled'].transform(
            'sum')

        # Subset and reorder the columns, drop duplicates, fill in missing values, and sort chronologically:
        output_df = output_df[
            ['County_Name', 'County_FIPS', 'State_Name', 'State_FIPS', 'Time_UTC', 'Raw_TELL_County_Load_MWh',
             'Scaled_TELL_County_Load_MWh']]
        output_df = output_df.drop_duplicates()
        output_df = output_df.fillna(-9999)
        output_df = output_df.sort_values("Time_UTC")

        # Round off the values to make the output file more readable:
        output_df['County_FIPS'] = output_df['County_FIPS'].round(0)
        output_df['State_FIPS'] = output_df['State_FIPS'].round(0)
        output_df['Raw_TELL_County_Load_MWh'] = output_df['Raw_TELL_County_Load_MWh'].round(2)
        output_df['Scaled_TELL_County_Load_MWh'] = output_df['Scaled_TELL_County_Load_MWh'].round(2)

        # Generate the .csv output file name:
        county_name = output_df['County_Name'].unique()[0]
        county_name = county_name.replace(" ", "_")
        county_name = county_name.replace(",", "_")
        state_name = output_df['State_Name'].unique()[0]
        state_name = state_name.replace(" ", "_")
        state_name = state_name.replace(",", "_")

        csv_output_filename = os.path.join(
            data_output_dir + '/County_Level_Data/' + year_to_process + '/TELL_' + state_name + '_' + county_name + '_Hourly_Load_Data_' +
            year_to_process + '_Scaled_' + gcam_target_year + '.csv')

        # Write out the dataframe to a .csv file:
        output_df.to_csv(csv_output_filename, sep=',', index=False)


def execute_forward(year_to_process: str, gcam_target_year: str, scenario_to_process: str, data_output_dir: str,
                    gcam_usa_input_dir: str, map_input_dir: str, mlp_input_dir: str, pop_input_dir: str,
                    save_county_data=False):
    """Takes the .csv files produced by the TELL MLP model and distributes
    the predicted load to the counties that each balancing authority (BA) operates
    in. The county-level hourly loads are then summed to the state-level and scaled
    to match the state-level annual loads produced by GCAM-USA. Three sets of output
    files are generated: county-level hourly loads, state-level hourly loads, and
    hourly loads for each BA. There is one additional summary output file that includes
    state-level annual loads from TELL and GCAM-USA as well as the scaling factors.

    :param year_to_process:             Year to process
    :type year_to_process:              str

    :param gcam_target_year:            Year to scale against the GCAM-USA annual loads
    :type gcam_target_year:             str

    :param scenario_to_process:         Scenario to process
    :type scenario_to_process:          str

    :param data_output_dir:              Top-level data directory for TELL output
    :type data_output_dir:               str

    :param gcam_usa_input_dir:          Path to where the GCAM-USA data is stored
    :type gcam_usa_input_dir:           str

    :param map_input_dir:               Path to where the BA-to-county mapping data are stored
    :type map_input_dir:                str

    :param mlp_input_dir:               Path to where the TELL MLP output data are stored
    :type mlp_input_dir:                str

    :param pop_input_dir:               Path to where the population projection data are stored
    :type pop_input_dir:                str

    :param save_county_data:            Set to True if you want to save the time-series of load for each county
    :type save_county_data:             bool

    :return:                            [0] DataFrame of summary statistics
                                        [1] DataFrame of BA-level total load time-series
                                        [2] DataFrame of state-level total load time-series

    """

    # Print the start time and set a time variable to benchmark the run time:
    begin_time = datetime.datetime.now()
    print('Scenario = ', scenario_to_process, ', Year = ', year_to_process)

    # Set the data output directory:
    data_output_dir_full = os.path.join(data_output_dir, scenario_to_process, gcam_target_year)

    # Check if the data output directory exists and if not then create it:
    if os.path.exists(data_output_dir_full) is False:
        os.makedirs(data_output_dir_full)
    if save_county_data:
        if os.path.exists(os.path.join(data_output_dir_full, 'County_Level_Data', year_to_process)) is False:
            os.mkdir(os.path.join(data_output_dir_full, 'County_Level_Data', year_to_process))

    # Load in the sample GCAM-USA output file and subset the data to only the "year_to_process":
    gcam_usa_df = extract_gcam_usa_loads(scenario_to_process = scenario_to_process,
                                         gcam_usa_input_dir = gcam_usa_input_dir)
    gcam_usa_df = gcam_usa_df[gcam_usa_df['Year'] == int(gcam_target_year)]

    # Load in the most recent (i.e., 2019) BA service territory mapping file:
    ba_mapping_df = pd.read_csv((os.path.join(map_input_dir, 'ba_service_territory_2019.csv')), index_col=None, header=0)

    # Read in the sample population projection dataset for the scenario being processed:
    population_df = process_population_scenario(scenario_to_process, pop_input_dir)

    # Subset to only the year being processed:
    if (scenario_to_process == 'historic') and (int(year_to_process) < 2000):
       population_df = population_df.loc[(population_df['Year'] == 2000)]
    else:
       population_df = population_df.loc[(population_df['Year'] == int(year_to_process))]

    # Only keep the columns that are needed:
    population_df = population_df[['County_FIPS', 'Population']].copy()

    # Merge the ba_mapping_df and population_df dataframes. Compute the fraction of the total population in each BA
    # that lives in a given county:
    mapping_df = ba_mapping_df.merge(population_df, on=['County_FIPS'])
    mapping_df = mapping_df.sort_values('BA_Number')
    mapping_df['BA_Population_Sum'] = mapping_df.groupby('BA_Code')['Population'].transform('sum')
    mapping_df['BA_Population_Fraction'] = mapping_df['Population'] / mapping_df['BA_Population_Sum']
    mapping_df = mapping_df.dropna()

    # Create a list of all of the MLP output files in the "mlp_input_dir" and aggregate the files in that list:
    mlp_filelist = sorted(glob.glob(os.path.join(mlp_input_dir, scenario_to_process, year_to_process, '*_mlp_output.csv')))
    mlp_output_df = aggregate_mlp_output_files(mlp_filelist)

    # Merge the "mapping_df" with "mlp_output_df" using BA codes to merge on:
    joint_mlp_df = pd.merge(mlp_output_df, mapping_df, on='BA_Code')

    # Scale the BA loads in each county by the fraction of the BA's total population that lives there:
    joint_mlp_df['County_BA_Load_MWh'] = joint_mlp_df['Total_BA_Load_MWh'].mul(joint_mlp_df['BA_Population_Fraction'])

    # Sum the county-level hourly loads into annual state-level total loads and convert that value from MWh to TWh:
    joint_mlp_df['TELL_State_Annual_Load_TWh'] = (joint_mlp_df.groupby('State_FIPS')['County_BA_Load_MWh'].transform('sum')) / 1000000

    # Add a column with the state-level annual total loads from GCAM-USA:
    joint_mlp_df = pd.merge(joint_mlp_df, gcam_usa_df[['State_FIPS', 'GCAM_USA_State_Annual_Load_TWh']], on='State_FIPS', how='left')

    # Compute the state-level scaling factors that force TELL annual loads to match GCAM-USA annual loads:
    joint_mlp_df['State_Scaling_Factor'] = joint_mlp_df['GCAM_USA_State_Annual_Load_TWh'].div(joint_mlp_df['TELL_State_Annual_Load_TWh'])

    # Apply those scaling factors to the "County_BA_Load_MWh" variable:
    joint_mlp_df['County_BA_Load_MWh_Scaled'] = joint_mlp_df['County_BA_Load_MWh'].mul(joint_mlp_df['State_Scaling_Factor'])

    # Output the resulting projections using the output functions:
    summary_df = output_tell_summary_data(joint_mlp_df, year_to_process, gcam_target_year, data_output_dir_full)
    ba_time_series_df = output_tell_ba_data(joint_mlp_df, year_to_process, gcam_target_year, data_output_dir_full)
    state_time_series_df = output_tell_state_data(joint_mlp_df, year_to_process, gcam_target_year, data_output_dir_full)

    # If the "save_county_data" flag is set to true then save the time-series of demand for each county:
    if save_county_data:
        output_tell_county_data(joint_mlp_df, year_to_process, gcam_target_year, data_output_dir_full)

    # Output the end time and elapsed time in order to benchmark the run time:
    print('Elapsed time = ', datetime.datetime.now() - begin_time)

    return summary_df, ba_time_series_df, state_time_series_df
