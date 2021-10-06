# execute_tell.py
# Casey D. Burleyson
# Pacific Northwest National Laboratory
# 17-Sep 2021

# This script takes the .csv files produced by the TELL MLP model and distributes
# the predicted load to the counties that each balancing authority (BA) operates
# in. The county-level hourly loads are then summed to the state-level and scaled
# to match the state-level annual loads produced by GCAM-USA. Three sets of output
# files are generated: county-level hourly loads, state-level hourly loads, and
# hourly loads for each BA. There is one additional summary output file that includes
# state-level annual loads from TELL and GCAM-USA as well as the scaling factors.
# All times are in UTC. Missing values are reported as -9999.

# Import all of the required libraries and packages:
import numpy as np
import pandas as pd
import os
from scipy import interpolate
from state_fips_functions import state_metadata_from_state_abbreviation

# Check if the nested data output directories exists and if not create them:
if os.path.exists(data_output_dir) == False:
   os.mkdir(data_output_dir)
if os.path.exists(data_output_dir + 'County_Level_Data/') == False:
   os.mkdir(data_output_dir + 'County_Level_Data/')

# Create a function to extract the state-level annual loads from a given GCAM-USA output file:
def extract_gcam_usa_loads(filename):
    """
    Extracts the state-level annual loads from a given GCAM-USA output file.
    :param filename: str -> name of the GCAM-USA output file
    :return: gcam_usa_output_df: df -> dataframe of state-level annual total electricity loads
    """
    # Load in the raw GCAM-USA output file:
    gcam_usa_df = pd.read_csv(filename, index_col=None, header=0)
    # Subset the data to only the total annual consumption of electricity by state:
    gcam_usa_df = gcam_usa_df[gcam_usa_df['param'].isin(['elecFinalBySecTWh'])]
    # Make a list of all of the states in the "gcam_usa_df":
    states = gcam_usa_df['region'].unique()
    # Loop over the states and interpolate their loads to an annual time step:
    for i in range(len(states)):
        # Subset to just the data for the state being processed:
        subset_df = gcam_usa_df[gcam_usa_df['region'].isin([states[i]])]
        # Retrieve the state metadata:
        (state_fips, state_name) = state_metadata_from_state_abbreviation(states[i])
        # Linearly interpolate the 5-year loads from GCAM-USA to an annual time step:
        annual_time_vector = pd.Series(range(subset_df['origX'].min(), subset_df['origX'].max()))
        interpolation_function = interpolate.interp1d(subset_df['origX'], subset_df['value'], kind='linear')
        annual_loads = interpolation_function(annual_time_vector)
        # Create an empty dataframe and store the results:
        state_df = pd.DataFrame()
        state_df['Year'] = pd.DataFrame(annual_time_vector)
        state_df['GCAM_USA_State_Annual_Load_TWh'] = annual_loads
        state_df['State_FIPS'] = state_fips
        state_df['State_Name'] = state_name
        state_df['State_Abbreviation'] = states[i]
        # Aggregate the output into a new dataframe:
        if i == 0:
            gcam_usa_output_df = state_df
        else:
            gcam_usa_output_df = gcam_usa_output_df.append(state_df)
    return gcam_usa_output_df


# Create a function to extract the balancing authority code from an MLP output filename:
def extract_ba_code(filename):
    """
    Extracts the balancing authority code from an MLP output filename.
    :param filename: str -> name of the MLP output file
    :return: ba_code: str -> alphanumeric code of the balancing authority
    """
    ba_code = filename[filename.rindex('/')+1:].rstrip('_mlp_predictions.csv')
    return ba_code

# Create a function to aggregate MLP output files:
def aggregate_mlp_output_files(list_of_files):
    """
    Aggregates a series of MLP output files into a dataframe.
    :param list_of_files: list -> list of MLP output files
    :return: mlp_output_df: df -> dataframe of all MLP output concatenated together
    """
    # Loop over the list of MLP output files:
    for file in range(len(list_of_files)):
        # Read in the .csv file and replace missing values with nan:
        mlp_data = pd.read_csv(list_of_files[file]).replace(-9999,np.nan)
        # Pull out the BA code from the filename:
        mlp_data['BA_Code'] = extract_ba_code(list_of_files[file])
        # Rename the "Predictions" variable:
        mlp_data.rename(columns={"Predictions":"Total_BA_Load_MWh"}, inplace=True)
        # Aggregate the output into a new dataframe:
        if file == 0:
           mlp_output_df = mlp_data
        else:
           mlp_output_df = mlp_output_df.append(mlp_data)
    return mlp_output_df


# Create a function to write a summary file describing state-level annual total loads from TELL and GCAM-USA:
def output_tell_summary_data(joint_mlp_df,data_output_dir):
    """
    Writes a summary file describing state-level annual total loads from TELL and GCAM-USA.
    :param joint_mlp_df: df -> dataframe of processed TELL loads
    :param data_output_dir: dir -> data output directory
    """
    # Make a copy of the necessary variables, drop the duplicates, and add in the "year_to_process":
    output_df = joint_mlp_df[{'State_FIPS', 'State_Name', 'TELL_State_Annual_Load_TWh', 'GCAM_USA_State_Annual_Load_TWh', 'State_Scaling_Factor'}].copy(deep=False)
    output_df = output_df.drop_duplicates()
    output_df['Year'] = year_to_process
    # Rename the columns to make them more readable:
    output_df.rename(columns={"TELL_State_Annual_Load_TWh":"Raw_TELL_Load_TWh",
                              "GCAM_USA_State_Annual_Load_TWh":"GCAM_USA_Load_TWh",
                              "State_Scaling_Factor":"Scaling_Factor"}, inplace=True)
    # Calculate the scaled TELL loads:
    output_df['Scaled_TELL_Load_TWh'] = output_df['Raw_TELL_Load_TWh'].mul(output_df['Scaling_Factor'])
    # Round off the values to make the output file more readable:
    output_df['State_FIPS'] = output_df['State_FIPS'].round(0)
    output_df['Raw_TELL_Load_TWh'] = output_df['Raw_TELL_Load_TWh'].round(5)
    output_df['GCAM_USA_Load_TWh'] = output_df['GCAM_USA_Load_TWh'].round(5)
    output_df['Scaled_TELL_Load_TWh'] = output_df['Scaled_TELL_Load_TWh'].round(5)
    output_df['Scaling_Factor'] = output_df['Scaling_Factor'].round(5)
    # Reorder the columns, fill in missing values, and sort alphabetically by state name:
    output_df = output_df[['Year', 'State_Name', 'State_FIPS', 'Scaling_Factor', 'GCAM_USA_Load_TWh', 'Raw_TELL_Load_TWh', 'Scaled_TELL_Load_TWh']]
    output_df = output_df.fillna(-9999)
    output_df = output_df.sort_values("State_Name")
    # Generate the .csv output file name:
    csv_output_filename = os.path.join(data_output_dir,'TELL_State_Summary_Data_' + year_to_process + '.csv')
    # Write out the dataframe to a .csv file:
    output_df.to_csv(csv_output_filename, sep=',',index=False)
    return

# Create a function to write a file of hourly loads for each BA:
def output_tell_ba_data(joint_mlp_df,data_output_dir):
    """
    Writes a file of hourly loads for each BA.
    :param joint_mlp_df: df -> dataframe of processed TELL loads
    :param data_output_dir: dir -> data output directory
    """
    # Make a copy of the necessary variables:
    ba_output_df = joint_mlp_df[{'Datetime', 'BA_Code', 'BA_Number', 'County_BA_Load_MWh', 'County_BA_Load_MWh_Scaled'}].copy(deep=False)
    # Make a list of all of the BAs in "ba_output_df":
    bas = ba_output_df['BA_Code'].unique()
    # Loop over the BAs and process their data into an output file:
    for i in range(len(bas)):
        # Subset to just the data for the BA being processed:
        output_df = ba_output_df[ba_output_df['BA_Code'].isin([bas[i]])]
        # Sum the county loads as a function of time:
        output_df['Raw_TELL_BA_Load_MWh'] = output_df.groupby('Datetime')['County_BA_Load_MWh'].transform('sum')
        output_df['Scaled_TELL_BA_Load_MWh'] = output_df.groupby('Datetime')['County_BA_Load_MWh_Scaled'].transform('sum')
        # Subset and reorder the columns, drop duplicates, fill in missing values, and sort chronologically:
        output_df = output_df[['BA_Code', 'BA_Number', 'Datetime', 'Raw_TELL_BA_Load_MWh', 'Scaled_TELL_BA_Load_MWh']]
        output_df = output_df.drop_duplicates()
        output_df = output_df.fillna(-9999)
        output_df = output_df.sort_values("Datetime")
        # Rename the "Datetime" variable:
        output_df.rename(columns={"Datetime":"Time_UTC"}, inplace=True)
        # Round off the values to make the output file more readable:
        output_df['BA_Number'] = output_df['BA_Number'].round(0)
        output_df['Raw_TELL_BA_Load_MWh'] = output_df['Raw_TELL_BA_Load_MWh'].round(3)
        output_df['Scaled_TELL_BA_Load_MWh'] = output_df['Scaled_TELL_BA_Load_MWh'].round(3)
        # Aggregate the output into a new dataframe:
        if i == 0:
            aggregate_output_df = output_df
        else:
            aggregate_output_df = aggregate_output_df.append(output_df)
    # Sort the data alphabetically by BA:
    aggregate_output_df = aggregate_output_df.sort_values(by=["BA_Code", "Time_UTC"])
    # Generate the .csv output file name:
    csv_output_filename = os.path.join(data_output_dir,'TELL_Balancing_Authority_Hourly_Load_Data_' + year_to_process + '.csv')
    # Write out the dataframe to a .csv file:
    aggregate_output_df.to_csv(csv_output_filename, sep=',', index=False)
    return

# Create a function to write out a file of hourly loads for each state:
def output_tell_state_data(joint_mlp_df,data_output_dir):
    """
    Writes a file of hourly loads for each state.
    :param joint_mlp_df: df -> dataframe of processed TELL loads
    :param data_output_dir: dir -> data output directory
    """
    # Make a copy of the necessary variables:
    state_output_df = joint_mlp_df[{'Datetime', 'State_FIPS', 'State_Name', 'State_Scaling_Factor', 'County_BA_Load_MWh', 'County_BA_Load_MWh_Scaled'}].copy(deep=False)
    # Make a list of all of the states in "state_output_df":
    states = state_output_df['State_Name'].unique()
    # Loop over the states and process their data:
    for i in range(len(states)):
        # Subset to just the data for the state being processed:
        output_df = state_output_df[state_output_df['State_Name'].isin([states[i]])]
        # Sum the county loads as a function of time:
        output_df['Raw_TELL_State_Load_MWh'] = output_df.groupby('Datetime')['County_BA_Load_MWh'].transform('sum')
        output_df['Scaled_TELL_State_Load_MWh'] = output_df.groupby('Datetime')['County_BA_Load_MWh_Scaled'].transform('sum')
        # Subset and reorder the columns, drop duplicates, fill in missing values, and sort chronologically:
        output_df = output_df[['State_Name', 'State_FIPS', 'Datetime', 'Raw_TELL_State_Load_MWh', 'Scaled_TELL_State_Load_MWh']]
        output_df = output_df.drop_duplicates()
        output_df = output_df.fillna(-9999)
        output_df = output_df.sort_values("Datetime")
        # Rename the "Datetime" variable:
        output_df.rename(columns={"Datetime": "Time_UTC"}, inplace=True)
        # Round off the values to make the output file more readable:
        output_df['State_FIPS'] = output_df['State_FIPS'].round(0)
        output_df['Raw_TELL_State_Load_MWh'] = output_df['Raw_TELL_State_Load_MWh'].round(5)
        output_df['Scaled_TELL_State_Load_MWh'] = output_df['Scaled_TELL_State_Load_MWh'].round(5)
        # Aggregate the output into a new dataframe:
        if i == 0:
           aggregate_output_df = output_df
        else:
           aggregate_output_df = aggregate_output_df.append(output_df)
    # Sort the data alphabetically by state:
    aggregate_output_df = aggregate_output_df.sort_values(by=["State_Name","Time_UTC"])
    # Generate the .csv output file name:
    csv_output_filename = os.path.join(data_output_dir,'TELL_State_Hourly_Load_Data_' + year_to_process + '.csv')
    # Write out the dataframe to a .csv file:
    aggregate_output_df.to_csv(csv_output_filename, sep=',',index=False)
    return

# Create a function to write out a file of hourly loads for each county:
def output_tell_county_data(joint_mlp_df,data_output_dir):
    """
    Writes a file of hourly loads for each county.
    :param joint_mlp_df: df -> dataframe of processed TELL loads
    :param data_output_dir: dir -> data output directory
    """
    # Make a copy of the necessary variables:
    county_output_df = joint_mlp_df[{'Datetime', 'County_FIPS', 'County_Name', 'State_Name', 'State_FIPS', 'County_BA_Load_MWh', 'County_BA_Load_MWh_Scaled'}].copy(deep=False)
    # Make a list of all of the counties in "county_output_df":
    counties = county_output_df['County_FIPS'].unique()
    # Loop over the counties and process their data:
    for i in range(len(counties)):
        # Subset to just the data for the county being processed:
        output_df = county_output_df[county_output_df['County_FIPS'] == counties[i]]
        # Sum the county loads as a function of time:
        output_df['Raw_TELL_County_Load_MWh'] = output_df.groupby('Datetime')['County_BA_Load_MWh'].transform('sum')
        output_df['Scaled_TELL_County_Load_MWh'] = output_df.groupby('Datetime')['County_BA_Load_MWh_Scaled'].transform('sum')
        # Subset and reorder the columns, drop duplicates, fill in missing values, and sort chronologically:
        output_df = output_df[['County_Name', 'County_FIPS', 'State_Name', 'State_FIPS', 'Datetime', 'Raw_TELL_County_Load_MWh', 'Scaled_TELL_County_Load_MWh']]
        output_df = output_df.drop_duplicates()
        output_df = output_df.fillna(-9999)
        output_df = output_df.sort_values("Datetime")
        # Rename the "Datetime" variable:
        output_df.rename(columns={"Datetime": "Time_UTC"}, inplace=True)
        # Round off the values to make the output file more readable:
        output_df['County_FIPS'] = output_df['County_FIPS'].round(0)
        output_df['State_FIPS'] = output_df['State_FIPS'].round(0)
        output_df['Raw_TELL_County_Load_MWh'] = output_df['Raw_TELL_County_Load_MWh'].round(5)
        output_df['Scaled_TELL_County_Load_MWh'] = output_df['Scaled_TELL_County_Load_MWh'].round(5)
        # Generate the .csv output file name:
        county_name = output_df['County_Name'].unique()[0]
        county_name = county_name.replace(" ", "_")
        county_name = county_name.replace(",", "_")
        state_name = output_df['State_Name'].unique()[0]
        state_name = state_name.replace(" ", "_")
        state_name = state_name.replace(",", "_")
        csv_output_filename = os.path.join(data_output_dir + 'County_Level_Data/TELL_' + state_name + '_' + county_name + '_Hourly_Load_Data_' + year_to_process + '.csv')
        # Write out the dataframe to a .csv file:
        output_df.to_csv(csv_output_filename, sep=',', index=False)
    return
