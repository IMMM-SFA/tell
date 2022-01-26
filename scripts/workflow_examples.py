import tell
import os
import time

# Create directory to store raw data
current_dir =  os.path.dirname(os.getcwd())
current_dir =  os.path.join(os.path.dirname(os.getcwd()), r'tell_valid')
raw_data_dir = os.path.join(current_dir, r'raw_data', r'demand_and_population')
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
population_input_dir = raw_data_dir
map_input_dir = os.path.join(current_dir, r'outputs', r'fips_mapping_files')
population_output_dir =  os.path.join(current_dir, r'outputs', r'hourly_population')
if not os.path.exists(pop_output_dir):
   os.makedirs(pop_output_dir)

tell.ba_pop_interpolate(map_input_dir, pop_input_dir, pop_output_dir, start_year, end_year)

## Create input directory meteorology data ##
wrf_input_dir =  os.path.join(current_dir, r'raw_data', r'wrf')
if not os.path.exists(wrf_input_dir):
   os.makedirs(wrf_input_dir)

# Download the raw wrf data from the Zenodo package
tell.install_sample_data(data_dir = wrf_input_dir)

## Create output directory meteorology data ##
wrf_output_dir =  os.path.join(current_dir, r'outputs', r'hourly_meterology')
if not os.path.exists(wrf_output_dir):
   os.makedirs(wrf_output_dir)

# set the target year
target_yr = 2019

# Process wrf data to put into right date format
tell.process_wrf(wrf_input_dir, wrf_output_dir, target_yr, n_jobs=-1)

## Compile hourly load, hourly population and hourly WRF data for MLP model ##
# create directory to store the compiled data
compile_output_dir =  os.path.join(current_dir, r'outputs', r'compiled_data')
if not os.path.exists(compile_output_dir):
   os.makedirs(compile_output_dir)

#set target year for WRF data
target_yr = 2019

# compile the hourly load data, population data, and wrf climate data by date
tell.compile_data(eia_930_output_dir, pop_output_dir, wrf_output_dir, target_yr, compile_output_dir)

# create the directory for the mlp output
mlp_output_dir =  os.path.join(current_dir, r'outputs', r'mlp_output')
if not os.path.exists(mlp_output_dir):
   os.makedirs(mlp_output_dir)

# specify the parameters of the MLP model
batch_run = True
target_ba_list = None
generate_plots = True
start_time = "2019-01-01 00:00:00"
end_time = "2019-12-31 23:00:00"
split_time = "2019-06-01 00:00:00"

t0 = time.time()

tell.predict(compile_output_dir ,
            mlp_output_dir,
            start_time = start_time,
            end_time = end_time,
            split_time = split_time,
            batch_run = batch_run,
            target_ba_list = target_ba_list,
            generate_plots = generate_plots)

# Set the year to process:
year_to_process = '2020'

# Set the data input and output directories:
mlp_input_dir = os.path.join(current_dir, r'outputs', r'mlp_output')
ba_geolocation_input_dir = os.path.join(current_dir, r'outputs', r'fips_mapping_files')
gcam_usa_input_dir = raw_data_dir
data_output_dir = os.path.join(current_dir, r'outputs', r'forward_output', rf'{year_to_process}')
if not os.path.exists(data_output_dir):
   os.makedirs(data_output_dir)


#Run the MLP model forward in time and
tell.execute_forward(year_to_process, mlp_input_dir, ba_geolocation_input_dir,
                     population_input_dir, gcam_usa_input_dir, data_output_dir)

#Data visualization
# Set the data input and output directories:
data_input_dir = data_output_dir
image_output_dir = os.path.join(current_dir, r'outputs', r'image_output')
if not os.path.exists(image_output_dir):
   os.makedirs(image_output_dir)
shapefile_input_dir = raw_data_dir

# Set the year of TELL output to visualize:
year_to_plot = '2020'

# Choose whether or not to save the images and set the image resolution:
save_images = 0 # (1 = Yes)
image_resolution = 150 # (dpi)

# If you want to save the images, check that the image output directory exist and if not then create it:
if save_images == 1:
    image_output_dir = os.path.join(image_output_dir, rf'{year_to_process}')
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)

# Plot a map of the state scaling factors:
tell.plot_state_scaling_factors(shapefile_input_dir, data_input_dir, year_to_plot, save_images, image_resolution,
                                image_output_dir)

# Plot the state annual total loads from GCAM-USA and TELL:
tell.plot_state_annual_total_loads(state_summary_df, year_to_plot, save_images, image_resolution, image_output_dir)

# Plot the time-series of total hourly loads for a given state by specifying either the state name or FIPS code:
tell.plot_state_load_time_series('California', state_hourly_load_df, year_to_plot, save_images, image_resolution,
                                 image_output_dir)

# Plot the load duration curve for a given state by specifying either the state name or FIPS code:
tell.plot_state_load_duration_curve('California', state_hourly_load_df, year_to_plot, save_images, image_resolution,
                                    image_output_dir)

# Plot the time-series of total hourly loads for a given BA by specifying either the BA abbreviation or BA number:
tell.('ERCO', ba_hourly_load_df, year_to_plot, save_images, image_resolution, image_output_dir)