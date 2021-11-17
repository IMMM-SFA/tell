import tell
import os

# Create directory to store raw data
current_dir =  os.path.dirname(os.getcwd())
raw_data_dir = os.path.join(current_dir, r'raw_data')
if not os.path.exists(raw_data_dir):
   os.makedirs(raw_data_dir)

# Download the raw data from the Zenodo package
tell.install_package_data(data_dir = raw_data_dir)

## FIPS to BA code mapping ##
#Set the start and end year for processing
start_year = 2015
end_year = 2019

tell.map_fips_codes(start_year, end_year,raw_data_dir, current_dir)

## Hourly load data (EIA 930) ##
# Set the data input and output directories:
eia_930_input_dir = raw_data_dir
eia_930_output_dir = os.path.join(current_dir, r'outputs', r'hourly_ba_load')
if not os.path.exists(eia_930_output_dir):
   os.makedirs(eia_930_output_dir)

# Process the hourly load data
tell.process_eia_930(eia_930_input_dir, eia_930_output_dir)

## County population data ##
# Set the data input and output directories:
pop_input_dir = raw_data_dir
map_input_dir = os.path.join(current_dir, r'outputs', r'fips_mapping_files')
pop_output_dir =  os.path.join(current_dir, r'outputs', r'hourly_population')
if not os.path.exists(pop_output_dir):
   os.makedirs(pop_output_dir)

tell.ba_pop_interpolate(map_input_dir, pop_input_dir, pop_output_dir, start_year, end_year)

## Meterology data ##

# Set the data input and output directories:
wrf_input_dir = current_dir
wrf_output_dir = f'{current_dir}/hourly_meterology'

# load the wrf to tell functions from im3components package
from im3components.wrf_tell.wrf_tell_counties import wrf_to_tell_counties
from im3components.wrf_tell.wrf_tell_balancing_authorities import wrf_to_tell_balancing_authorities

# average wrf meterolgoy by county
df = wrf_to_tell_counties(wrf_input_dir)

# aggregate wrf county averages to annual hourly times-series  of population-weighted meteorology for each balancing
# authority (BA).
wrf_to_tell_balancing_authorities(df, wrf_output_dir)

## Compile hourly load, hourly population and hourly WRF data for MLP model ##
compile_output_dir = f'{current_dir}/compiled_data'

tell.compile_data(eia_930_output_dir, pop_output_dir, wrf_output_dir, compile_output_dir)

import time

output_dir =  f'{current_dir}/mlp_output'
batch_run = True
target_ba_list = None
generate_plots = True
start_time = "2016-01-01 00:00:00"
end_time = "2019-12-31 23:00:00"
start_test_period = "2018-12-31 23:00:00"

t0 = time.time()

tell.predict(compile_output_dir ,
            out_dir,
            start_time = start_time,
            end_time = end_time
            start_test_period = start_test_period,
            batch_run = batch_run,
            target_ba_list = target_ba_list,
            generate_plots = generate_plots)

# Set the year and GCAM-USA scenario to process:
year_to_process = '2020'
gcam_usa_scenario = 'scenario_name'

# Set the data input and output directories:
mlp_input_dir = 'f'{current_dir}/'mlp_output'/{year_to_process}'
ba_geolocation_input_dir = current_dir
gcam_usa_input_dir = f'{current_dir}/{gcam_usa_scenario}'
data_output_dir = 'f'{current_dir}/'mlp_state_output'/{year_to_process}'

#Run the MLP model forward in time and
tell.execute_forward(year_to_process, mlp_input_dir, ba_geolocation_input_dir,
                     pop_input_dir, gcam_usa_input_dir, data_output_dir)

#Data visualization
# Set the data input and output directories:
data_input_dir = data_output_dir
image_output_dir = 'f'{current_dir}/'image_output'/
shapefile_input_dir = raw_data_dir

# Set the year of TELL output to visualize:
year_to_plot = '2020'

# Choose whether or not to save the images and set the image resolution:
save_images = 0 # (1 = Yes)
image_resolution = 150 # (dpi)
# If you want to save the images, check that the image output directory exist and if not then create it:
if save_images == 1:
   if os.path.exists((image_output_dir + year_to_plot)) == False:
      os.mkdir((image_output_dir + year_to_plot))
# Read in the state shapefile, rename the GEOID variable as 'State_FIPS', convert 'State_FIPS' to an integer then mulitply it by 1000:
states_df = gpd.read_file((shapefile_input_dir + 'tl_2020_us_state.shp')).rename(columns={'GEOID': 'State_FIPS',})
states_df['State_FIPS'] = states_df['State_FIPS'].astype(int) * 1000

# Read in the 'TELL_State_Summary_Data' .csv file and reassign the 'State_FIPS' code as an integer:
state_summary_df = pd.read_csv((data_input_dir + year_to_plot + '/' + 'TELL_State_Summary_Data_' + year_to_plot + '.csv'), dtype={'State_FIPS': int})

# Merge the two dataframes together using state FIPS codes to join them and display the merged datafram:
states_df = states_df.merge(state_summary_df, on='State_FIPS', how='left')

# Plot a map of the state scaling factors:
tell.plot_state_scaling_factors(states_df, year_to_plot, save_images, image_resolution, image_output_dir)

# Plot the state annual total loads from GCAM-USA and TELL:
tell.plot_state_annual_total_loads(state_summary_df, year_to_plot, save_images, image_resolution, image_output_dir)

# Read in the 'TELL_State_Hourly_Load_Data' .csv file and display the dataframe:
state_hourly_load_df = pd.read_csv((data_input_dir + year_to_plot + '/' + 'TELL_State_Hourly_Load_Data_' + year_to_plot + '.csv'), parse_dates=["Time_UTC"])
state_hourly_load_df

# Plot the time-series of total hourly loads for a given state by specifying either the state name or FIPS code:
tell.plot_state_load_time_series('California', state_hourly_load_df, year_to_plot, save_images, image_resolution, image_output_dir)

# Plot the load duration curve for a given state by specifying either the state name or FIPS code:
tell.plot_state_load_duration_curve('California', state_hourly_load_df, year_to_plot, save_images, image_resolution, image_output_dir)

# Read in the 'TELL_Balancing_Authority_Hourly_Load_Data' .csv file and display the dataframe:
ba_hourly_load_df = pd.read_csv((data_input_dir + year_to_plot + '/' + 'TELL_Balancing_Authority_Hourly_Load_Data_' + year_to_plot + '.csv'), parse_dates=["Time_UTC"])
ba_hourly_load_df

# Plot the time-series of total hourly loads for a given BA by specifying either the BA abbreviation or BA number:
tell.plot_ba_load_time_series('ERCO', ba_hourly_load_df, year_to_plot, save_images, image_resolution, image_output_dir)