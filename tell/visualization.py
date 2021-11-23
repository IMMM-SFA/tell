# Import all of the required libraries and packages:
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_state_scaling_factors(shapefile_input_dir, data_input_dir, year_to_plot, save_images, image_resolution,
                               image_output_dir):
    """Select wanted columns in each file
     :param shapefile_input_dir:        Directory where the Census TL shapefile is stored
     :type shapefile_input_dir:         str
     :param data_input_dir:             Directory to store the EIA 930 hourly load data as a csv
     :type data_input_dir:              dir
     :param year_to_plot:               Directory to store the EIA 930 hourly load data as a csv
     :type year_to_plot:                dir
     :param save_images:                Directory to store the EIA 930 hourly load data as a csv
     :type save_images:                 dir
     :param image_resolution:           Directory to store the EIA 930 hourly load data as a csv
     :type image_resolution:            dir
     :param image_output_dir:           Directory to store the EIA 930 hourly load data as a csv
     :type image_output_dir:            dir
     :return:                           Subsetted dataframe of EIA 930 hourly data
      """
    states_df = gpd.read_file((shapefile_input_dir + 'tl_2020_us_state.shp')).rename(columns={'GEOID': 'State_FIPS', })
    states_df['State_FIPS'] = states_df['State_FIPS'].astype(int) * 1000

    # Read in the 'TELL_State_Summary_Data' .csv file and reassign the 'State_FIPS' code as an integer:
    state_summary_df = pd.read_csv((data_input_dir + year_to_plot + '/' + 'TELL_State_Summary_Data_' + year_to_plot + '.csv'),
        dtype={'State_FIPS': int})

    # Merge the two dataframes together using state FIPS codes to join them and display the merged datafram:
    states_df = states_df.merge(state_summary_df, on='State_FIPS', how='left')

    fig, ax = plt.subplots(1, 1, figsize=(25,10))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    ax1 = states_df.plot(column='Scaling_Factor',
                         cmap='RdBu_r',
                         ax=ax,
                         cax=cax,
                         edgecolor='grey',
                         vmin = 0.5,
                         vmax = 1.5,
                         linewidth=0.1,
                         legend=True,
                         legend_kwds={'label': 'TELL Scaling Factor','orientation': 'vertical'})
    ax1.set_title(('State-Level Scaling Factors in ' + year_to_plot))
    if save_images == 1:
       plt.savefig((image_output_dir + year_to_plot + '/' + 'TELL_State_Scaling_Factors_' + year_to_plot + '.png'), dpi=image_resolution, bbox_inches='tight')

def plot_state_annual_total_loads(data_input_dir, year_to_plot, save_images, image_resolution, image_output_dir):
    """Select wanted columns in each file
       :param data_input_dir:             Directory to store the EIA 930 hourly load data as a csv
       :type data_input_dir:              dir
       :param year_to_plot:               Directory to store the EIA 930 hourly load data as a csv
       :type year_to_plot:                dir
       :param save_images:                Directory to store the EIA 930 hourly load data as a csv
       :type save_images:                 dir
       :param image_resolution:           Directory to store the EIA 930 hourly load data as a csv
       :type image_resolution:            dir
       :param image_output_dir:           Directory to store the EIA 930 hourly load data as a csv
       :type image_output_dir:            dir
       :return:                           Subsetted dataframe of EIA 930 hourly data
        """
     state_summary_df = pd.read_csv((data_input_dir + year_to_plot + '/' + 'TELL_State_Summary_Data_' + year_to_plot + '.csv'),
        dtype={'State_FIPS': int})
        x_axis = np.arange(len(state_summary_df))

        plt.figure(figsize=(22, 10))
        plt.bar(x_axis - 0.2, state_summary_df['GCAM_USA_Load_TWh'], 0.4, label='GCAM-USA Loads')
        plt.bar(x_axis + 0.2, state_summary_df['Raw_TELL_Load_TWh'], 0.4, label='Unscaled TELL Loads')
        plt.xticks(x_axis, state_summary_df['State_Name'])
        plt.xticks(rotation=90)
        plt.legend()
        plt.ylabel("Annual Total Load [TWh]")
        plt.title(('Annual Total Loads from GCAM-USA and TELL in ' + year_to_plot))

        if save_images == 1:
            plt.savefig((image_output_dir + year_to_plot + '/' + 'TELL_State_Annual_Total_Loads_' + year_to_plot + '.png'),
                dpi=image_resolution, bbox_inches='tight')


def plot_state_load_time_series(state, data_input_dir, year_to_plot, save_images, image_resolution,
                                image_output_dir):
    """Select wanted columns in each file
       :param state:                      Directory where the Census TL shapefile is stored
       :type state:                       str
       :param data_input_dir:             Directory to store the EIA 930 hourly load data as a csv
       :type data_input_dir:              dir
       :param year_to_plot:               Directory to store the EIA 930 hourly load data as a csv
       :type year_to_plot:                dir
       :param save_images:                Directory to store the EIA 930 hourly load data as a csv
       :type save_images:                 dir
       :param image_resolution:           Directory to store the EIA 930 hourly load data as a csv
       :type image_resolution:            dir
       :param image_output_dir:           Directory to store the EIA 930 hourly load data as a csv
       :type image_output_dir:            dir
       :return:                           Subsetted dataframe of EIA 930 hourly data
        """
    state_hourly_load_df = pd.read_csv(
        (data_input_dir + year_to_plot + '/' + 'TELL_State_Hourly_Load_Data_' + year_to_plot + '.csv'),
        parse_dates=["Time_UTC"])

    if (type(state)) == str:
        state_subset_df = state_hourly_load_df.loc[state_hourly_load_df['State_Name'] == state]
    if (type(state)) == int:
        state_subset_df = state_hourly_load_df.loc[state_hourly_load_df['State_FIPS'] == state]

    fig, ax = plt.subplots(2, figsize=(22, 10), sharex=True, sharey=True)
    ax[0].plot(state_subset_df['Time_UTC'], state_subset_df['Raw_TELL_State_Load_MWh'], 'k-', label='Raw Load',
               linewidth=0.5)
    ax[1].plot(state_subset_df['Time_UTC'], state_subset_df['Scaled_TELL_State_Load_MWh'], 'k-', label='Scaled Load',
               linewidth=0.5)
    ax[0].set_title((state_subset_df['State_Name'].iloc[0] + ' Raw TELL Loads in ' + year_to_plot))
    ax[1].set_title((state_subset_df['State_Name'].iloc[0] + ' Scaled TELL Loads in ' + year_to_plot))
    ax[0].set_ylabel('Hourly Load [MWh]')
    ax[1].set_ylabel('Hourly Load [MWh]')

    state_name = state_subset_df['State_Name'].iloc[0]
    state_name = state_name.replace(" ", "_")

    if save_images == 1:
        plt.savefig((image_output_dir + year_to_plot + '/' + 'TELL_State_Hourly_Loads_' + state_name + '_' + year_to_plot + '.png'),
                    dpi=image_resolution

        def plot_state_load_duration_curve(state, state_hourly_load_df, year_to_plot, save_images, image_resolution,
                                           image_output_dir):
    if (type(state)) == str:
        state_subset_df = state_hourly_load_df.loc[state_hourly_load_df['State_Name'] == state]
    if (type(state)) == int:
        state_subset_df = state_hourly_load_df.loc[state_hourly_load_df['State_FIPS'] == state]

    load_df_sorted = state_subset_df.sort_values(by=['Scaled_TELL_State_Load_MWh'], ascending=False)
    load_df_sorted['Interval'] = 1
    load_df_sorted['Duration'] = load_df_sorted['Interval'].cumsum()

    plt.figure(figsize=(22, 10))
    plt.plot(load_df_sorted['Duration'], load_df_sorted['Raw_TELL_State_Load_MWh'], 'k-', label='Raw Load',
             linewidth=0.5)
    plt.xlabel("Duration [h]")
    plt.ylabel("Scaled State Hourly Load [MWh]")
    plt.title((state_subset_df['State_Name'].iloc[0] + ' Load Duration Curve in ' + year_to_plot))

    state_name = state_subset_df['State_Name'].iloc[0]
    state_name = state_name.replace(" ", "_")
    if save_images == 1:
        plt.savefig((image_output_dir + year_to_plot + '/' + 'TELL_State_Load_Duration_Curve_' + state_name + '_' + year_to_plot + '.png'),
                    dpi=image_reso

        # Define a function to plot time-series of hourly loads for a given BA based on the BA abbreviation or BA number:


def plot_ba_load_time_series(BA, data_input_dir, year_to_plot, save_images, image_resolution, image_output_dir):
    """Select wanted columns in each file
       :param BA:                         Directory where the Census TL shapefile is stored
       :type BA:                          str
       :param data_input_dir:             Directory to store the EIA 930 hourly load data as a csv
       :type data_input_dir:              dir
       :param year_to_plot:               Directory to store the EIA 930 hourly load data as a csv
       :type year_to_plot:                dir
       :param save_images:                Directory to store the EIA 930 hourly load data as a csv
       :type save_images:                 dir
       :param image_resolution:           Directory to store the EIA 930 hourly load data as a csv
       :type image_resolution:            dir
       :param image_output_dir:           Directory to store the EIA 930 hourly load data as a csv
       :type image_output_dir:            dir
       :return:                           Subsetted dataframe of EIA 930 hourly data
        """
    # Read in the 'TELL_Balancing_Authority_Hourly_Load_Data' .csv file and display the dataframe:
    ba_hourly_load_df = pd.read_csv((data_input_dir + year_to_plot + '/' + 'TELL_Balancing_Authority_Hourly_Load_Data_' + year_to_plot + '.csv'),
        parse_dates=["Time_UTC"])

    if (type(BA)) == str:
        ba_subset_df = ba_hourly_load_df.loc[ba_hourly_load_df['BA_Code'] == BA]
    if (type(BA)) == int:
        ba_subset_df = ba_hourly_load_df.loc[ba_hourly_load_df['BA_Number'] == BA]

    fig, ax = plt.subplots(2, figsize=(22, 10), sharex=True, sharey=True)
    ax[0].plot(ba_subset_df['Time_UTC'], ba_subset_df['Raw_TELL_BA_Load_MWh'], 'k-', label='Raw Load', linewidth=0.5)
    ax[1].plot(ba_subset_df['Time_UTC'], ba_subset_df['Scaled_TELL_BA_Load_MWh'], 'k-', label='Scaled Load',
               linewidth=0.5)
    ax[0].set_title((ba_subset_df['BA_Code'].iloc[0] + ' Raw TELL Loads in ' + year_to_plot))
    ax[1].set_title((ba_subset_df['BA_Code'].iloc[0] + ' Scaled TELL Loads in ' + year_to_plot))
    ax[0].set_ylabel('Hourly Load [MWh]')
    ax[1].set_ylabel('Hourly Load [MWh]')

    if save_images == 1:
        plt.savefig((image_output_dir + year_to_plot + '/' + 'TELL_BA_Hourly_Loads_' + ba_subset_df['BA_Code'].iloc[
            0] + '_' + year_to_plot + '.png'), dpi=image_resolution, bbox_i