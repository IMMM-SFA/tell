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
    ba_service_territory_input_dir = os.path.join(data_input_dir, r'tell_quickstarter_data', r'outputs', r'ba_service_territory')

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


def plot_mlp_summary_statistics(validation_df, image_output_dir: str, image_resolution: int, save_images=False):
    """Plot the summary statistics of the MLP evaluation data across BAs

    :param validation_df:       Validation dataframe produced by the batch training of MLP models for all BAs
    :type validation_dft:       df

    :param image_output_dir:    Directory to store the images
    :type image_output_dir:     str

    :param image_resolution:    Resolution at which you want to save the images in DPI
    :type image_resolution:     int

    :param save_images:         Set to True if you want to save the images after they're generated
    :type save_images:          bool

    """

    # Create an x-axis the length of the dataframe to be used in plotting:
    x_axis = np.arange(len(validation_df))

    # Make the plot:
    plt.figure(figsize=(25, 10))
    plt.subplot(221)
    plt.bar(x_axis, validation_df.sort_values(by=['R2'], ascending=True)['R2'], 0.75)
    plt.xticks(x_axis, validation_df.sort_values(by=['R2'], ascending=True)['BA'], rotation=90)
    plt.grid()
    plt.xlabel('Balancing Authority')
    plt.ylabel('R2 Score')
    plt.title('Coefficient of Determination')

    plt.subplot(222)
    plt.bar(x_axis, validation_df.sort_values(by=['MAPE'], ascending=True)['MAPE'], 0.75)
    plt.xticks(x_axis, validation_df.sort_values(by=['MAPE'], ascending=True)['BA'], rotation=90)
    plt.grid()
    plt.xlabel('Balancing Authority')
    plt.ylabel('MAPE')
    plt.title('Mean Absolute Percentage Error')

    plt.subplot(223)
    plt.bar(x_axis, validation_df.sort_values(by=['RMS_ABS'], ascending=True)['RMS_ABS'], 0.75)
    plt.xticks(x_axis, validation_df.sort_values(by=['RMS_ABS'], ascending=True)['BA'], rotation=90)
    plt.grid()
    plt.xlabel('Balancing Authority')
    plt.ylabel('Absolute RMS Error [MWh]')
    plt.title('Absolute Root-Mean-Squared Error')

    plt.subplot(224)
    plt.bar(x_axis, validation_df.sort_values(by=['RMS_NORM'], ascending=True)['RMS_NORM'], 0.75)
    plt.xticks(x_axis, validation_df.sort_values(by=['RMS_NORM'], ascending=True)['BA'], rotation=90)
    plt.grid()
    plt.xlabel('Balancing Authority')
    plt.ylabel('Normalized RMS Error')
    plt.title('Normalized Root-Mean-Squared Error')

    plt.subplots_adjust(wspace=0.15, hspace=0.4)

    # If the "save_images" flag is set to true then save the plot to a .png file:
    if save_images:
        plt.savefig(os.path.join(image_output_dir, 'MLP_Summary_Statistics.png'), dpi=image_resolution,
                    bbox_inches='tight', facecolor='white')


def plot_mlp_errors_vs_load(prediction_df, validation_df, image_output_dir: str, image_resolution: int, save_images=False):
    """Plot the summary statistics of the MLP evaluation data as a function of mean load

    :param prediction_df:       Prediction dataframe produced by the batch training of MLP models for all BAs
    :type prediction_df:        df

    :param validation_df:       Validation dataframe produced by the batch training of MLP models for all BAs
    :type validation_df:        df

    :param image_output_dir:    Directory to store the images
    :type image_output_dir:     str

    :param image_resolution:    Resolution at which you want to save the images in DPI
    :type image_resolution:     int

    :param save_images:         Set to True if you want to save the images after they're generated
    :type save_images:          bool

    """

    # Compute the mean hourly load for each BA:
    prediction_df['Mean_Load_MWh'] = prediction_df.groupby('region')['predictions'].transform('mean')

    # Rename the region variable:
    prediction_df.rename(columns={'region': 'BA'}, inplace=True)

    # Keep on the variables we need:
    mean_load_df = prediction_df[['BA', 'Mean_Load_MWh']].copy().drop_duplicates()

    # Merge the mean load data into the validation dataframe:
    validation_df = validation_df.merge(mean_load_df, on=['BA'])

    # Make the plot:
    plt.figure(figsize=(25, 10))
    plt.subplot(221)
    plt.scatter(validation_df['Mean_Load_MWh'], validation_df['R2'], s=15, c='blue')
    plt.grid()
    plt.xlabel('Mean Hourly Load [MWh]')
    plt.ylabel('R2 Score')
    plt.title('Coefficient of Determination')

    plt.subplot(222)
    plt.scatter(validation_df['Mean_Load_MWh'], validation_df['MAPE'], s=15, c='blue')
    plt.grid()
    plt.xlabel('Mean Hourly Load [MWh]')
    plt.ylabel('MAPE')
    plt.title('Mean Absolute Percentage Error')

    plt.subplot(223)
    plt.scatter(validation_df['Mean_Load_MWh'], validation_df['RMS_ABS'], s=15, c='blue')
    plt.grid()
    plt.xlabel('Mean Hourly Load [MWh]')
    plt.ylabel('Absolute RMS Error [MWh]')
    plt.title('Absolute Root-Mean-Squared Error')

    plt.subplot(224)
    plt.scatter(validation_df['Mean_Load_MWh'], validation_df['RMS_NORM'], s=15, c='blue')
    plt.grid()
    plt.xlabel('Mean Hourly Load [MWh]')
    plt.ylabel('Normalized RMS Error')
    plt.title('Normalized Root-Mean-Squared Error')

    plt.subplots_adjust(wspace=0.15, hspace=0.4)

    # If the "save_images" flag is set to true then save the plot to a .png file:
    if save_images:
        plt.savefig(os.path.join(image_output_dir, 'MLP_Summary_Statistics_vs_Load.png'), dpi=image_resolution,
                    bbox_inches='tight', facecolor='white')

    return validation_df


def plot_mlp_ba_time_series(prediction_df, ba_to_plot: str,
                            image_output_dir: str, image_resolution: int, save_images=False):
    """Plot the performance metrics for an individual BA

    :param prediction_df:       Prediction dataframe produced by the batch training of MLP models for all BAs
    :type prediction_df:        df

    :param ba_to_plot:          Code for the BA you want to plot
    :type ba_to_plot:           str

    :param image_output_dir:    Directory to store the images
    :type image_output_dir:     str

    :param image_resolution:    Resolution at which you want to save the images in DPI
    :type image_resolution:     int

    :param save_images:         Set to True if you want to save the images after they're generated
    :type save_images:          bool

    """

    # Rename the region variable:
    prediction_df.rename(columns={'region': 'BA'}, inplace=True)

    # Subset to just the data for the BA you want to plot
    subset_df = prediction_df[prediction_df['BA'].isin([ba_to_plot])]

    one_to_one = np.arange(0, 200000, 1000)

    # Make the plot:
    plt.figure(figsize=(25, 10))
    plt.subplot(211)
    plt.plot(subset_df['datetime'], subset_df['ground_truth'], 'r', linewidth=0.5, label='Observed')
    plt.plot(subset_df['datetime'], subset_df['predictions'], 'b', linewidth=0.5, label='Predicted')
    plt.xlim(subset_df['datetime'].dropna().min(), subset_df['datetime'].dropna().max())
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Demand [MWh]')
    plt.title('Hourly Demand Time Series in ' + ba_to_plot)

    plt.subplot(223)
    plt.hist(subset_df['ground_truth'], bins=40, density=True, histtype='step', edgecolor = 'r', label='Observed', linewidth=3)
    plt.hist(subset_df['predictions'], bins=40, density=True, histtype='step', edgecolor = 'b', label='Predicted', linewidth=3)
    plt.legend()
    plt.xlabel('Demand [MWh]')
    plt.ylabel('Frequency')
    plt.title('Hourly Demand Distribution in ' + ba_to_plot)

    plt.subplot(224)
    plt.scatter(subset_df['ground_truth'], subset_df['predictions'], s=15, c='blue', label='Hourly Sample')
    plt.plot(one_to_one,one_to_one,'k', linewidth=3, label = '1:1')
    plt.plot(one_to_one, (one_to_one*1.1), 'k', linewidth=3, linestyle='--', label = '1:1 - 10%')
    plt.plot(one_to_one, (one_to_one*0.9), 'k', linewidth=3, linestyle='--', label = '1:1 + 10%')
    plt.legend()
    plt.xlim(0.98*subset_df[['ground_truth', 'predictions']].min().min(), 1.02*subset_df[['ground_truth', 'predictions']].max().max())
    plt.ylim(0.98*subset_df[['ground_truth', 'predictions']].min().min(), 1.02*subset_df[['ground_truth', 'predictions']].max().max())
    plt.xlabel('Observed Hourly Demand [MWh]')
    plt.ylabel('Predicted Hourly Demand [MWh]')
    plt.title('Hourly Demand Relationship in ' + ba_to_plot)

    plt.subplots_adjust(wspace=0.15, hspace=0.4)

    # If the "save_images" flag is set to true then save the plot to a .png file:
    if save_images:
        plt.savefig(os.path.join(image_output_dir, ba_to_plot + '_Time_Series.png'), dpi=image_resolution,
                    bbox_inches='tight', facecolor='white')


def plot_mlp_ba_peak_week(prediction_df, ba_to_plot: str,
                          image_output_dir: str, image_resolution: int, save_images=False):
    """Plot the time-series of load during the peak week of the year for a given BA.

    :param prediction_df:       Prediction dataframe produced by the batch training of MLP models for all BAs
    :type prediction_df:        df

    :param ba_to_plot:          Code for the BA you want to plot
    :type ba_to_plot:           str

    :param image_output_dir:    Directory to store the images
    :type image_output_dir:     str

    :param image_resolution:    Resolution at which you want to save the images in DPI
    :type image_resolution:     int

    :param save_images:         Set to True if you want to save the images after they're generated
    :type save_images:          bool

    """

    # Rename the region variable:
    prediction_df.rename(columns={'region': 'BA'}, inplace=True)

    # Subset to just the data for the BA you want to plot
    subset_df = prediction_df[prediction_df['BA'].isin([ba_to_plot])].copy()

    # Smooth the predictions using exponentially-weighted windows:
    subset_df['Rolling_Mean'] = subset_df['predictions'].ewm(span=168).mean()

    # Find the index of the maximum value of the rolling mean:
    index = subset_df['Rolling_Mean'].idxmax(axis=0)
    if index > 84:
       start = (index -84)
    else:
       start = 0

    if index < (len(subset_df)-84):
       end = (index + 84)
    else:
       end = len(subset_df)

    peak_df = subset_df[start:end]

    # Make the plot:
    plt.figure(figsize=(25, 10))
    plt.plot(peak_df['datetime'], peak_df['ground_truth'], 'r', linewidth=3, label='Observed')
    plt.plot(peak_df['datetime'], peak_df['predictions'], 'b', linewidth=3, label='Predicted')
    plt.xlim(peak_df['datetime'].dropna().min(), peak_df['datetime'].dropna().max())
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Demand [MWh]')
    plt.title('Peak Demand Week in ' + ba_to_plot)

    # If the "save_images" flag is set to true then save the plot to a .png file:
    if save_images:
        plt.savefig(os.path.join(image_output_dir, ba_to_plot + '_Peak_Week.png'), dpi=image_resolution,
                    bbox_inches='tight', facecolor='white')


def plot_state_scaling_factors(year_to_plot: str, gcam_target_year: str, scenario_to_plot: str,
                               data_input_dir: str, image_output_dir: str, image_resolution: int, save_images=False):
    """Plot the scaling factor that force TELL annual total state loads to agree with GCAM-USA

    :param year_to_plot:        Year you want to plot (valid 2039, 2059, 2079, 2099)
    :type year_to_plot:         str

    :param gcam_target_year:    Year to scale against the GCAM-USA annual loads
    :type gcam_target_year:     str

    :param scenario_to_plot:    Scenario you want to plot
    :type scenario_to_plot:     str

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
    tell_data_input_dir = os.path.join(data_input_dir, r'tell_quickstarter_data', r'outputs', r'tell_output', scenario_to_plot, year_to_plot)

    # Read in the states shapefile and change the geolocation variable name to state FIPS code:
    states_df = gpd.read_file(os.path.join(data_input_dir, r'tell_raw_data', r'State_Shapefiles', r'tl_2020_us_state.shp')).rename(columns={'GEOID': 'State_FIPS'})

    # Convert the state FIPS code to an integer and multiply it by 1000:
    states_df['State_FIPS'] = states_df['State_FIPS'].astype(int) * 1000

    # Read in the 'TELL_State_Summary_Data' .csv file and reassign the 'State_FIPS' code as an integer:
    state_summary_df = pd.read_csv((tell_data_input_dir + '/' + 'TELL_State_Summary_Data_' + year_to_plot
                                    + '_Scaled_' + gcam_target_year + '.csv'), dtype={'State_FIPS': int})

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


def plot_state_annual_total_loads(year_to_plot: str, gcam_target_year: str, scenario_to_plot: str, data_input_dir: str,
                                  image_output_dir: str, image_resolution: int, save_images=False):
    """Plot annual total loads from both GCAM-USA and TELL

    :param year_to_plot:        Year you want to plot (valid 2039, 2059, 2079, 2099)
    :type year_to_plot:         str

    :param gcam_target_year:    Year to scale against the GCAM-USA annual loads
    :type gcam_target_year:     str

    :param scenario_to_plot:    Scenario you want to plot
    :type scenario_to_plot:     str

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
    tell_data_input_dir = os.path.join(data_input_dir, r'tell_quickstarter_data', r'outputs', r'tell_output', scenario_to_plot, year_to_plot)

    # Read in the 'TELL_State_Summary_Data' .csv file and reassign the 'State_FIPS' code as an integer:
    state_summary_df = pd.read_csv((tell_data_input_dir + '/' + 'TELL_State_Summary_Data_' + year_to_plot
                                    + '_Scaled_' + gcam_target_year + '.csv'), dtype={'State_FIPS': int})

    # Create an x-axis the length of the dataframe to be used in plotting:
    x_axis = np.arange(len(state_summary_df))

    # Make the plot:
    plt.figure(figsize=(25, 10))
    plt.bar(x_axis - 0.2, state_summary_df['GCAM_USA_Load_TWh'], 0.4, label=('GCAM-USA Loads: Year = ' + gcam_target_year))
    plt.bar(x_axis + 0.2, state_summary_df['Raw_TELL_Load_TWh'], 0.4, label=('Unscaled TELL Loads: Year = ' + year_to_plot))
    plt.xticks(x_axis, state_summary_df['State_Name'])
    plt.xticks(rotation=90)
    plt.legend()
    plt.ylabel("Annual Total Load [TWh]")
    plt.title(('Annual Total Loads from GCAM-USA and TELL in ' + year_to_plot))

    # If the "save_images" flag is set to true then save the plot to a .png file:
    if save_images == True:
        filename = ('TELL_State_Annual_Total_Loads_' + year_to_plot + '.png')
        plt.savefig(os.path.join(image_output_dir, filename), dpi=image_resolution, bbox_inches='tight')


def plot_state_load_time_series(state_to_plot: str, year_to_plot: str, gcam_target_year: str, scenario_to_plot: str,
                                data_input_dir: str, image_output_dir: str, image_resolution: int, save_images=False):
    """Plot the time series of load for a given state

    :param state_to_plot:       State you want to plot
    :type state_to_plot:        str

    :param year_to_plot:        Year you want to plot (valid 2039, 2059, 2079, 2099)
    :type year_to_plot:         str

    :param gcam_target_year:    Year to scale against the GCAM-USA annual loads
    :type gcam_target_year:     str

    :param scenario_to_plot:    Scenario you want to plot
    :type scenario_to_plot:     str

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
    tell_data_input_dir = os.path.join(data_input_dir, r'tell_quickstarter_data', r'outputs', r'tell_output',
                                       scenario_to_plot, year_to_plot)

    # Read in the 'TELL_State_Summary_Data' .csv file parse the time variable:
    state_hourly_load_df = pd.read_csv((tell_data_input_dir + '/' + 'TELL_State_Hourly_Load_Data_' + year_to_plot
                                        + '_Scaled_' + gcam_target_year + '.csv'), parse_dates=["Time_UTC"])

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


def plot_state_load_duration_curve(state_to_plot: str, year_to_plot: str, gcam_target_year: str, scenario_to_plot: str,
                                   data_input_dir: str, image_output_dir: str, image_resolution: int,
                                   save_images=False):
    """Plot the load duration curve for a given state

    :param state_to_plot:       State you want to plot
    :type state_to_plot:        str

    :param year_to_plot:        Year you want to plot (valid 2039, 2059, 2079, 2099)
    :type year_to_plot:         str

    :param gcam_target_year:    Year to scale against the GCAM-USA annual loads
    :type gcam_target_year:     str

    :param scenario_to_plot:    Scenario you want to plot
    :type scenario_to_plot:     str

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
    tell_data_input_dir = os.path.join(data_input_dir, r'tell_quickstarter_data', r'outputs', r'tell_output', scenario_to_plot, year_to_plot)

    # Read in the 'TELL_State_Summary_Data' .csv file and parse the time variable:
    state_hourly_load_df = pd.read_csv((tell_data_input_dir + '/' + 'TELL_State_Hourly_Load_Data_' + year_to_plot
                                        + '_Scaled_' + gcam_target_year + '.csv'), parse_dates=["Time_UTC"])

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


def plot_ba_load_time_series(ba_to_plot: str, year_to_plot: str, gcam_target_year: str, scenario_to_plot: str,
                             data_input_dir: str, image_output_dir: str, image_resolution: int, save_images=False):
    """Plot the time series of load for a given Balancing Authority

    :param ba_to_plot:          Balancing Authority code for the BA you want to plot
    :type ba_to_plot:           str

    :param year_to_plot:        Year you want to plot (valid 2039, 2059, 2079, 2099)
    :type year_to_plot:         str

    :param gcam_target_year:    Year to scale against the GCAM-USA annual loads
    :type gcam_target_year:     str

    :param scenario_to_plot:    Scenario you want to plot
    :type scenario_to_plot:     str

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
    tell_data_input_dir = os.path.join(data_input_dir, r'tell_quickstarter_data', r'outputs', r'tell_output',
                                       scenario_to_plot, year_to_plot)

    # Read in the 'TELL_Balancing_Authority_Hourly_Load_Data' .csv file and parse the time variable:
    ba_hourly_load_df = pd.read_csv((tell_data_input_dir + '/' + 'TELL_Balancing_Authority_Hourly_Load_Data_'
                                     + year_to_plot + '_Scaled_' + gcam_target_year + '.csv'), parse_dates=["Time_UTC"])

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
    compiled_data_input_dir = os.path.join(data_input_dir, r'tell_quickstarter_data', r'outputs', r'compiled_historical_data')

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
