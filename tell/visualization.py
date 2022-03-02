import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_state_scaling_factors(shapefile_input_dir: str, data_input_dir: str, year_to_plot: str, image_resolution: int,
                               image_output_dir: str, save_images=0):
    """Create state scaling factors map and save image in image directory

    :param shapefile_input_dir:        Directory where the Census TL shapefile is stored
    :type shapefile_input_dir:         str

    :param data_input_dir:             Directory to store the EIA 930 hourly load data as a csv
    :type data_input_dir:              str

    :param year_to_plot:               What year to plot as a data visualization
    :type year_to_plot:                str

    :param save_images:                Save images to directory? 0 is no, 1 is yes (default = 0)
    :type save_images:                 int

    :param image_resolution:           Resolution in dpi to save images
    :type image_resolution:            int

    :param image_output_dir:           Directory to store the image outputs
    :type image_output_dir:            str

    :return:                           State scaling factors map and save image in image directory


    """

    states_df = gpd.read_file((shapefile_input_dir + '/tl_2020_us_state.shp')).rename(columns={'GEOID': 'State_FIPS', })
    states_df['State_FIPS'] = states_df['State_FIPS'].astype(int) * 1000

    # Read in the 'TELL_State_Summary_Data' .csv file and reassign the 'State_FIPS' code as an integer:
    state_summary_df = pd.read_csv((data_input_dir + '/' + year_to_plot + '/' + 'TELL_State_Summary_Data_' +
                                    year_to_plot + '.csv'), dtype={'State_FIPS': int})

    # Merge the two dataframes together using state FIPS codes to join them and display the merged data frame:
    states_df = states_df.merge(state_summary_df, on='State_FIPS', how='left')

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
                         linewidth=0.1,
                         legend=True,
                         legend_kwds={'label': 'TELL Scaling Factor', 'orientation': 'vertical'})
    ax1.set_title(('State-Level Scaling Factors in ' + year_to_plot))
    if save_images == 1: plt.savefig((image_output_dir + year_to_plot + '/' + 'TELL_State_Scaling_Factors_' +
                                      year_to_plot + '.png'), dpi=image_resolution, bbox_inches='tight')


def plot_state_annual_total_loads(data_input_dir: str, year_to_plot: str,  image_resolution: int, image_output_dir: str,
                                  save_images=0):
    """Plot the state annual total loads and save to image directory

    :param data_input_dir:             Directory to store the EIA 930 hourly load data as a csv
    :type data_input_dir:              str

    :param year_to_plot:               What year to plot as a data visualization
    :type year_to_plot:                str

    :param save_images:                Save images to directory? 0 is no, 1 is yes (default = 0)
    :type save_images:                 int

    :param image_resolution:           Resolution in dpi to save images
    :type image_resolution:            int

    :param image_output_dir:           Directory to store the image outputs
    :type image_output_dir:            str

    :return:                           Plot of state annual total loads and save to image directory

    """

    state_summary_df = pd.read_csv(
        (data_input_dir + year_to_plot + '/' + 'TELL_State_Summary_Data_' + year_to_plot + '.csv'),
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


def plot_state_load_time_series(state: str, data_input_dir: str, year_to_plot: str,  image_resolution: int,
                                image_output_dir: str, save_images=0):
    """Plot state load time series and save to image directory
    :param state:                      What state to plot state load time series
    :type state:                       str

    :param data_input_dir:             Directory to store the EIA 930 hourly load data as a csv
    :type data_input_dir:              str

    :param year_to_plot:               What year to plot as a data visualization
    :type year_to_plot:                str

    :param save_images:                Save images to directory? 0 is no, 1 is yes (default = 0)
    :type save_images:                 int

    :param image_resolution:           Resolution in dpi to save images
    :type image_resolution:            int

    :param image_output_dir:           Directory to store the image outputs
    :type image_output_dir:            str

    :return:                           Plot of state load time series and save to image directory

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
        plt.savefig((
                                image_output_dir + year_to_plot + '/' + 'TELL_State_Hourly_Loads_' + state_name + '_'
                                + year_to_plot + '.png'), dpi=image_resolution, bbox_inches='tight')

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
        plt.savefig((image_output_dir + year_to_plot + '/' + 'TELL_State_Load_Duration_Curve_' + state_name + '_' +
                     year_to_plot + '.png'), dpi=image_resolution, bbox_inches='tight')


def plot_ba_load_time_series(ba: str, data_input_dir: str, year_to_plot: str,  image_resolution: int,
                             image_output_dir: str, save_images=0):
    """Plot the BA load time series and save to image directory

    :param ba:                         What BA to plot load time series
    :type ba:                          str

    :param data_input_dir:             Directory to store the EIA 930 hourly load data as a csv
    :type data_input_dir:              str

    :param year_to_plot:               What year to plot as a data visualization
    :type year_to_plot:                str

    :param save_images:                Save images to directory? 0 is no, 1 is yes (default = 0)
    :type save_images:                 int

    :param image_resolution:           Resolution in dpi to save images
    :type image_resolution:            int

    :param image_output_dir:           Directory to store the image outputs
    :type image_output_dir:            str

    :return:                            BA load time series plot and save to image directory

    """

    # Read in the 'TELL_Balancing_Authority_Hourly_Load_Data' .csv file and display the dataframe:
    ba_hourly_load_df = pd.read_csv(
        (data_input_dir + year_to_plot + '/' + 'TELL_Balancing_Authority_Hourly_Load_Data_' + year_to_plot + '.csv'),
        parse_dates=["Time_UTC"])

    if (type(ba)) == str:
        ba_subset_df = ba_hourly_load_df.loc[ba_hourly_load_df['BA_Code'] == ba]
    if (type(ba)) == int:
        ba_subset_df = ba_hourly_load_df.loc[ba_hourly_load_df['BA_Number'] == ba]

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
            0] + '_' + year_to_plot + '.png'), dpi=image_resolution, bbox_inches='tight')
