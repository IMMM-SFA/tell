import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_ba_service_territory(ba_to_plot: str, year_to_plot: str, data_input_dir: str, image_output_dir: str,
                              image_resolution: int, save_images=False):
    """Plot maps of the service territory for a given BA in a given year

    :param ba_to_plot:          Code for the BA you want to plot
    :type ba_to_plot:           str

    :param year_to_plot:        Year you want to plot (valid 2015-2019)
    :type year_to_plot:         str

    :param data_input_dir:      Top-level data directory for TELL
    :type data_input_dir:       str

    :param image_output_dir:    Directory to store the images
    :type image_output_dir:     str

    :param image_resolution:    Resolution at which you want to save the images in DPI
    :type image_resolution:     int

    :param save_images:         Set to True if you want to save the images after they're generated
    :type save_images:          bool

    """

    # Set the input directories based on the 'data_input_dir' variable:
    shapefile_input_dir = os.path.join(data_input_dir, r'tell_raw_data', r'County_Shapefiles')
    population_input_dir = os.path.join(data_input_dir, r'tell_raw_data', r'Population')
    ba_service_territory_input_dir = os.path.join(data_input_dir, r'outputs', r'ba_service_territory')

    # Read in the county shapefile and reassign the 'FIPS' variable as integers:
    counties_df = gpd.read_file(os.path.join(shapefile_input_dir, r'tl_2020_us_county.shp')).rename(columns={'GEOID': 'County_FIPS'})
    counties_df['County_FIPS'] = counties_df['County_FIPS'].astype(int)

    # Read in county populations file:
    population_df = pd.read_csv(os.path.join(population_input_dir, r'county_populations_2000_to_2020.csv'))

    # Keep only the columns we need:
    population_df = population_df[['county_FIPS', ('pop_' + year_to_plot)]].copy(deep=False)

    # Rename the columns:
    population_df.rename(columns={"county_FIPS": "County_FIPS", ('pop_' + year_to_plot): "Population"}, inplace=True)

    # Read in the BA mapping file:
    ba_mapping_df = pd.read_csv((os.path.join(ba_service_territory_input_dir, f'ba_service_territory_{str(year_to_plot)}.csv')), index_col=None, header=0)

    # Merge the ba_mapping_df and population_df together using county FIPS codes to join them:
    ba_mapping_df = ba_mapping_df.merge(population_df, on='County_FIPS', how='left')

    # Merge the ba_mapping_df and counties_df together using county FIPS codes to join them:
    counties_df = counties_df.merge(ba_mapping_df, on='County_FIPS', how='left')

    # Subset to only the BA you want to plot:
    counties_subset_df = counties_df.loc[counties_df['BA_Code'] == ba_to_plot]

    # Create the figure:
    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    ax1 = counties_subset_df.plot(column='Population',
                                  cmap='GnBu',
                                  ax=ax,
                                  cax=cax,
                                  edgecolor='grey',
                                  linewidth=0.5,
                                  legend=True,
                                  legend_kwds={'label': ('County Population in ' + year_to_plot), 'orientation': 'vertical'})
    ax1.set_title((ba_to_plot + ' Service Territory in ' + year_to_plot))

    # If the "save_images" flag is set to true then save the plot to a .png file:
    if save_images == True:
       filename = (ba_to_plot + '_Service_Territory_' + year_to_plot + '.png')
       plt.savefig(os.path.join(image_output_dir, filename), dpi=image_resolution, bbox_inches='tight')


def plot_mlp_summary_statistics(year_to_plot: str, data_input_dir: str, image_output_dir: str,
                                image_resolution: int, save_images=False):
    """Plot the summary statistics of the MLP evaluation data across BAs

    :param year_to_plot:        Year you want to plot (valid 2015-2019)
    :type year_to_plot:         str

    :param data_input_dir:      Top-level data directory for TELL
    :type data_input_dir:       str

    :param image_output_dir:    Directory to store the images
    :type image_output_dir:     str

    :param image_resolution:    Resolution at which you want to save the images in DPI
    :type image_resolution:     int

    :param save_images:         Set to True if you want to save the images after they're generated
    :type save_images:          bool

    """

    # Set the input directory based on the 'data_input_dir' and 'year_to_plot' variables:
    mlp_input_dir = os.path.join(data_input_dir, r'outputs', r'mlp_output', year_to_plot)

    # Read in summary statistics file:
    statistics_df = pd.read_csv(os.path.join(mlp_input_dir, r'summary.csv'))

    # Sort the statistics by R2 value:
    statistics_df_sorted = statistics_df.sort_values(by=['R2'], ascending=True)

    # Create an x-axis the length of the dataframe to be used in plotting:
    x_axis = np.arange(len(statistics_df_sorted))

    # Make the plot:
    plt.figure(figsize=(25, 10))
    plt.bar(x_axis, statistics_df_sorted['R2'], 0.75, label='Correlation')
    plt.xticks(x_axis, statistics_df_sorted['BA'])
    plt.xticks(rotation=90)
    plt.ylim([0, 1])
    plt.xlabel("Balancing Authority")
    plt.ylabel("Correlation with Observed Loads")
    plt.title(('Correlation Between Observed and MLP Predicted Loads in ' + year_to_plot))

    # If the "save_images" flag is set to true then save the plot to a .png file:
    if save_images == True:
       filename = ('MLP_Correlations_by_BA_' + year_to_plot + '.png')
       plt.savefig(os.path.join(image_output_dir, filename), dpi=image_resolution, bbox_inches='tight')

    # Multiply the MAPE values by 100 to convert them to percentages:
    statistics_df['MAPE'] = statistics_df['MAPE'] * 100

    # Sort the statistics by MAPE value:
    statistics_df_sorted = statistics_df.sort_values(by=['MAPE'], ascending=True)

    # Create an x-axis the length of the dataframe to be used in plotting:
    x_axis = np.arange(len(statistics_df_sorted))

    # Make the plot:
    plt.figure(figsize=(25, 10))
    plt.bar(x_axis, statistics_df_sorted['MAPE'], 0.75, label='MAPE')
    plt.xticks(x_axis, statistics_df_sorted['BA'])
    plt.xticks(rotation=90)
    plt.xlabel("Balancing Authority")
    plt.ylabel("Mean Absolute Percentage Error [%]")
    plt.title(('Mean Absolute Percentage Error Between Observed and MLP Predicted Loads in ' + year_to_plot))

    # If the "save_images" flag is set to true then save the plot to a .png file:
    if save_images == True:
       filename = ('MLP_MAPE_by_BA_' + year_to_plot + '.png')
       plt.savefig(os.path.join(image_output_dir, filename), dpi=image_resolution, bbox_inches='tight')


def plot_state_scaling_factors(year_to_plot: str, data_input_dir: str, image_output_dir: str,
                               image_resolution: int, save_images=False):
    """Plot the scaling factor that force TELL annual total state loads to agree with GCAM-USA

    :param year_to_plot:        Year you want to plot (valid 2039, 2059, 2079, 2099)
    :type year_to_plot:         str

    :param data_input_dir:      Top-level data directory for TELL
    :type data_input_dir:       str

    :param image_output_dir:    Directory to store the images
    :type image_output_dir:     str

    :param image_resolution:    Resolution at which you want to save the images in DPI
    :type image_resolution:     int

    :param save_images:         Set to True if you want to save the images after they're generated
    :type save_images:          bool

    """

    # Set the data input directories for the various variables you need:
    sample_data_input_dir = os.path.join(data_input_dir, r'sample_output_data', year_to_plot)

    # Read in the states shapefile and change the geolocation variable name to state FIPS code:
    states_df = gpd.read_file(os.path.join(data_input_dir, r'tell_raw_data', r'State_Shapefiles', r'tl_2020_us_state.shp')).rename(columns={'GEOID': 'State_FIPS'})

    # Convert the state FIPS code to an integer and multiply it by 1000:
    states_df['State_FIPS'] = states_df['State_FIPS'].astype(int) * 1000

    # Read in the 'TELL_State_Summary_Data' .csv file and reassign the 'State_FIPS' code as an integer:
    state_summary_df = pd.read_csv((sample_data_input_dir + '/' + 'TELL_State_Summary_Data_' + year_to_plot + '.csv'), dtype={'State_FIPS': int})

    # Merge the two dataframes together using state FIPS codes to join them:
    states_df = states_df.merge(state_summary_df, on='State_FIPS', how='left')

    # Make the plot:
    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    ax1 = states_df.plot(column='Scaling_Factor',
                         cmap='RdBu_r',
                         ax=ax,
                         cax=cax,
                         edgecolor='grey',
                         vmin=0.5,
                         vmax=1.5,
                         linewidth=0.5,
                         legend=True,
                         legend_kwds={'label': 'TELL Scaling Factor', 'orientation': 'vertical'})
    ax1.set_title(('State-Level Scaling Factors in ' + year_to_plot))

    # If the "save_images" flag is set to true then save the plot to a .png file:
    if save_images == True:
        filename = ('TELL_State_Scaling_Factors_' + year_to_plot + '.png')
        plt.savefig(os.path.join(image_output_dir, filename), dpi=image_resolution, bbox_inches='tight')


def plot_state_annual_total_loads(year_to_plot: str, data_input_dir: str, image_output_dir: str,
                                  image_resolution: int, save_images=False):
    """Plot annual total loads from both GCAM-USA and TELL

    :param year_to_plot:        Year you want to plot (valid 2039, 2059, 2079, 2099)
    :type year_to_plot:         str

    :param data_input_dir:      Top-level data directory for TELL
    :type data_input_dir:       str

    :param image_output_dir:    Directory to store the images
    :type image_output_dir:     str

    :param image_resolution:    Resolution at which you want to save the images in DPI
    :type image_resolution:     int

    :param save_images:         Set to True if you want to save the images after they're generated
    :type save_images:          bool

    """

    # Set the data input directories for the various variables you need:
    sample_data_input_dir = os.path.join(data_input_dir, r'sample_output_data', year_to_plot)

    # Read in the 'TELL_State_Summary_Data' .csv file and reassign the 'State_FIPS' code as an integer:
    state_summary_df = pd.read_csv((sample_data_input_dir + '/' + 'TELL_State_Summary_Data_' + year_to_plot + '.csv'), dtype={'State_FIPS': int})

    # Create an x-axis the length of the dataframe to be used in plotting:
    x_axis = np.arange(len(state_summary_df))

    # Make the plot:
    plt.figure(figsize=(25, 10))
    plt.bar(x_axis - 0.2, state_summary_df['GCAM_USA_Load_TWh'], 0.4, label='GCAM-USA Loads')
    plt.bar(x_axis + 0.2, state_summary_df['Raw_TELL_Load_TWh'], 0.4, label='Unscaled TELL Loads')
    plt.xticks(x_axis, state_summary_df['State_Name'])
    plt.xticks(rotation=90)
    plt.legend()
    plt.ylabel("Annual Total Load [TWh]")
    plt.title(('Annual Total Loads from GCAM-USA and TELL in ' + year_to_plot))

    # If the "save_images" flag is set to true then save the plot to a .png file:
    if save_images == True:
        filename = ('TELL_State_Annual_Total_Loads_' + year_to_plot + '.png')
        plt.savefig(os.path.join(image_output_dir, filename), dpi=image_resolution, bbox_inches='tight')


def plot_state_load_time_series(state_to_plot: str, year_to_plot: str, data_input_dir: str, image_output_dir: str,
                                image_resolution: int, save_images=False):
    """Plot the time series of load for a given state

    :param state_to_plot:       State you want to plot
    :type state_to_plot:        str

    :param year_to_plot:        Year you want to plot (valid 2039, 2059, 2079, 2099)
    :type year_to_plot:         str

    :param data_input_dir:      Top-level data directory for TELL
    :type data_input_dir:       str

    :param image_output_dir:    Directory to store the images
    :type image_output_dir:     str

    :param image_resolution:    Resolution at which you want to save the images in DPI
    :type image_resolution:     int

    :param save_images:         Set to True if you want to save the images after they're generated
    :type save_images:          bool

    """

    # Set the data input directories for the various variables you need:
    sample_data_input_dir = os.path.join(data_input_dir, r'sample_output_data', year_to_plot)

    # Read in the 'TELL_State_Summary_Data' .csv file parse the time variable:
    state_hourly_load_df = pd.read_csv((sample_data_input_dir + '/' + 'TELL_State_Hourly_Load_Data_' + year_to_plot + '.csv'), parse_dates=["Time_UTC"])

    # Subset the dataframe to only the state you want to plot:
    state_subset_df = state_hourly_load_df.loc[state_hourly_load_df['State_Name'] == state_to_plot]

    # Make the plot:
    fig, ax = plt.subplots(2, figsize=(25, 10), sharex=True, sharey=True)
    ax[0].plot(state_subset_df['Time_UTC'], state_subset_df['Raw_TELL_State_Load_MWh'], 'k-', label='Raw Load',
               linewidth=0.5)
    ax[1].plot(state_subset_df['Time_UTC'], state_subset_df['Scaled_TELL_State_Load_MWh'], 'k-', label='Scaled Load',
               linewidth=0.5)
    ax[0].set_title((state_subset_df['State_Name'].iloc[0] + ' Raw TELL Loads in ' + year_to_plot))
    ax[1].set_title((state_subset_df['State_Name'].iloc[0] + ' Scaled TELL Loads in ' + year_to_plot))
    ax[0].set_ylabel('Hourly Load [MWh]')
    ax[1].set_ylabel('Hourly Load [MWh]')

    # If the "save_images" flag is set to true then save the plot to a .png file:
    if save_images == True:
        state_name = state_subset_df['State_Name'].iloc[0]
        state_name = state_name.replace(" ", "_")
        filename = ('TELL_State_Hourly_Loads_' + state_name + '_' + year_to_plot + '.png')
        plt.savefig(os.path.join(image_output_dir, filename), dpi=image_resolution, bbox_inches='tight')


def plot_state_load_duration_curve(state_to_plot: str, year_to_plot: str, data_input_dir: str, image_output_dir: str,
                                   image_resolution: int, save_images=False):
    """Plot the load duration curve for a given state

    :param state_to_plot:       State you want to plot
    :type state_to_plot:        str

    :param year_to_plot:        Year you want to plot (valid 2039, 2059, 2079, 2099)
    :type year_to_plot:         str

    :param data_input_dir:      Top-level data directory for TELL
    :type data_input_dir:       str

    :param image_output_dir:    Directory to store the images
    :type image_output_dir:     str

    :param image_resolution:    Resolution at which you want to save the images in DPI
    :type image_resolution:     int

    :param save_images:         Set to True if you want to save the images after they're generated
    :type save_images:          bool

    """

    # Set the data input directories for the various variables you need:
    sample_data_input_dir = os.path.join(data_input_dir, r'sample_output_data', year_to_plot)

    # Read in the 'TELL_State_Summary_Data' .csv file and parse the time variable:
    state_hourly_load_df = pd.read_csv((sample_data_input_dir + '/' + 'TELL_State_Hourly_Load_Data_' + year_to_plot + '.csv'), parse_dates=["Time_UTC"])

    # Subset the dataframe to only the state you want to plot:
    state_subset_df = state_hourly_load_df.loc[state_hourly_load_df['State_Name'] == state_to_plot]

    # Sort the hourly load values from largest to smallest and compute the hourly duration for each value:
    load_df_sorted = state_subset_df.sort_values(by=['Scaled_TELL_State_Load_MWh'], ascending=False)
    load_df_sorted['Interval'] = 1
    load_df_sorted['Duration'] = load_df_sorted['Interval'].cumsum()

    # Make the plot:
    plt.figure(figsize=(25, 10))
    plt.plot(load_df_sorted['Duration'], load_df_sorted['Raw_TELL_State_Load_MWh'], 'k-', label='Raw Load', linewidth=0.5)
    plt.xlabel("Duration [h]")
    plt.ylabel("Scaled State Hourly Load [MWh]")
    plt.title((state_subset_df['State_Name'].iloc[0] + ' Load Duration Curve in ' + year_to_plot))

    # If the "save_images" flag is set to true then save the plot to a .png file:
    if save_images == True:
        state_name = state_subset_df['State_Name'].iloc[0]
        state_name = state_name.replace(" ", "_")
        filename = ('TELL_State_Load_Duration_Curve_' + state_name + '_' + year_to_plot + '.png')
        plt.savefig(os.path.join(image_output_dir, filename), dpi=image_resolution, bbox_inches='tight')


def plot_ba_load_time_series(ba_to_plot: str, year_to_plot: str, data_input_dir: str, image_output_dir: str,
                             image_resolution: int, save_images=False):
    """Plot the time series of load for a given Balancing Authority

    :param ba_to_plot:          Balancing Authority code for the BA you want to plot
    :type ba_to_plot:           str

    :param year_to_plot:        Year you want to plot (valid 2039, 2059, 2079, 2099)
    :type year_to_plot:         str

    :param data_input_dir:      Top-level data directory for TELL
    :type data_input_dir:       str

    :param image_output_dir:    Directory to store the images
    :type image_output_dir:     str

    :param image_resolution:    Resolution at which you want to save the images in DPI
    :type image_resolution:     int

    :param save_images:         Set to True if you want to save the images after they're generated
    :type save_images:          bool

    """

    # Set the data input directories for the various variables you need:
    sample_data_input_dir = os.path.join(data_input_dir, r'sample_output_data', year_to_plot)

    # Read in the 'TELL_Balancing_Authority_Hourly_Load_Data' .csv file and parse the time variable:
    ba_hourly_load_df = pd.read_csv((sample_data_input_dir + '/' + 'TELL_Balancing_Authority_Hourly_Load_Data_' + year_to_plot + '.csv'),
                                    parse_dates=["Time_UTC"])

    # Subset the dataframe to only the BA you want to plot:
    ba_subset_df = ba_hourly_load_df.loc[ba_hourly_load_df['BA_Code'] == ba_to_plot]

    # Make the plot:
    fig, ax = plt.subplots(2, figsize=(25, 10), sharex=True, sharey=True)
    ax[0].plot(ba_subset_df['Time_UTC'], ba_subset_df['Raw_TELL_BA_Load_MWh'], 'k-', label='Raw Load', linewidth=0.5)
    ax[1].plot(ba_subset_df['Time_UTC'], ba_subset_df['Scaled_TELL_BA_Load_MWh'], 'k-', label='Scaled Load', linewidth=0.5)
    ax[0].set_title((ba_to_plot + ' Raw TELL Loads in ' + year_to_plot))
    ax[1].set_title((ba_to_plot + ' Scaled TELL Loads in ' + year_to_plot))
    ax[0].set_ylabel('Hourly Load [MWh]')
    ax[1].set_ylabel('Hourly Load [MWh]')

    # If the "save_images" flag is set to true then save the plot to a .png file:
    if save_images == True:
        filename = ('TELL_BA_Hourly_Loads_' + ba_to_plot + '_' + year_to_plot + '.png')
        plt.savefig(os.path.join(image_output_dir, filename), dpi=image_resolution, bbox_inches='tight')


def plot_ba_variable_correlations(ba_to_plot: str, data_input_dir: str, image_output_dir: str, image_resolution: int,
                                  save_images=False):
    """Plot the correlation matrix between predictive variables and observed demand for individual or all BAs.

    :param ba_to_plot:          BA code for the BA you want to plot. Set to "All" to plot the average
                                correlation across all BAs.
    :type ba_to_plot:           str

    :param data_input_dir:      Top-level data directory for TELL
    :type data_input_dir:       str

    :param image_output_dir:    Directory to store the images
    :type image_output_dir:     str

    :param image_resolution:    Resolution at which you want to save the images in DPI
    :type image_resolution:     int

    :param save_images:         Set to True if you want to save the images after they're generated
    :type save_images:          bool

    """

    # Set the input directory based on the 'data_input_dir' variable:
    compiled_data_input_dir = os.path.join(data_input_dir, r'outputs', r'compiled_historical_data')

    if ba_to_plot != 'All':
        # Read in compiled historical data file for the BA you want to plot:
        df = pd.read_csv(os.path.join(compiled_data_input_dir, f'{ba_to_plot}_historical_data.csv'))

        # Rename the a few columns for simplicity:
        df.rename(columns={"Adjusted_Demand_MWh": "Demand"}, inplace=True)
        df.rename(columns={"Total_Population": "Population"}, inplace=True)

        # Drop out the columns we don't need anymore:
        df.drop(['Forecast_Demand_MWh', 'Adjusted_Generation_MWh', 'Adjusted_Interchange_MWh'], axis=1, inplace=True)

        # Calculate the correlation matrix of the dataframe:
        corr = df.corr()
    else:
        # Loop over the compiled historical data files in the input directory:
        for idx, file in enumerate(glob(f'{compiled_data_input_dir}/*.csv')):

            # Read in the .csv file:
            dfx = pd.read_csv(os.path.join(compiled_data_input_dir, file))

            # Rename the a few columns for simplicity:
            dfx.rename(columns={"Adjusted_Demand_MWh": "Demand"}, inplace=True)
            dfx.rename(columns={"Total_Population": "Population"}, inplace=True)

            # Drop out the columns we don't need anymore:
            dfx.drop(['Forecast_Demand_MWh', 'Adjusted_Generation_MWh', 'Adjusted_Interchange_MWh'], axis=1,
                     inplace=True)

            # Calculate the correlation matrix of the dataframe:
            corrx = dfx.corr()

            # Concatenate the correlation matrix across BAs:
            if idx == 0:
                corrall = corrx.copy()
            else:
                corrall = np.dstack((corrall, corrx))

            del dfx, corrx

        # Calculate the average correlation matrix across all BAs and convert that value to a pd dataframe for plotting:
        corr = pd.DataFrame(np.nanmean(corrall, axis=2),
                            columns=['Year', 'Month', 'Day', 'Hour', 'Demand', 'Population', 'T2', 'Q2', 'SWDOWN',
                                     'GLW', 'WSPD'])

    # Fill diagonal and upper half with NaNs
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    corr[mask] = np.nan

    f = plt.figure(figsize=(25, 10))
    plt.matshow(corr,
                fignum=f.number,
                cmap='RdBu_r',
                vmin=-1,
                vmax=1)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    if ba_to_plot != 'All':
        plt.title('Correlation Matrix in the ' + ba_to_plot + ' Balancing Authority', fontsize=16);
    else:
        plt.title('Average Correlation Matrix Across All Balancing Authorities in TELL', fontsize=16);
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)

    # If the "save_images" flag is set to true then save the plot to a .png file:
    if save_images == True:
        filename = (ba_to_plot + '_Correlation_Matrix.png')
        plt.savefig(os.path.join(image_output_dir, filename), dpi=image_resolution, bbox_inches='tight')
