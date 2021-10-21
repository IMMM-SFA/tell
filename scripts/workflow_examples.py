import tell
import im3components

# Change to your local directory where you would like to store the data
tell_data_dir = 'C:/Users/mcgr323/projects/tell_valid/tell/tell_data'

# Download the raw data from the Zenodo package
tell.install_package_data(data_dir=data_dir)

## Hourly load data (EIA 930) ##
# Set the data input and output directories:
eia_930_input_dir = tell_data_dir
eia_930_output_dir = f'{tell_data_dir}/hourly_ba_load'

# Process the hourly load data
tell.process_eia_930(eia_930_input_dir, eia_930_output_dir)

## County population data ##
# Set the data input and output directories:
pop_input_dir = tell_data_dir
map_input_dir = tell_data_dir
pop_output_dir = f'{tell_data_dir}/hourly_population'

# Set some processing flags:
start_year = 2015;  # Starting year of time series
end_year = 2019;  # Ending year of time series

tell.ba_pop_interpolate(map_input_dir, pop_input_dir, pop_output_dir, start_year, end_year)

## Meterolgoy data ##

# Set the data input and output directories:
wrf_input_dir = tell_data_dir
wrf_output_dir = f'{tell_data_dir}/hourly_meterology'

# load the wrf to tell functions from im3components package
from im3components.wrf_tell.wrf_tell_counties import wrf_to_tell_counties
from im3components.wrf_tell.wrf_tell_balancing_authorities import wrf_to_tell_balancing_authorities

# average wrf meterolgoy by county
df = wrf_to_tell_counties(wrf_input_dir)

# aggregate wrf county averages to annual hourly times-series  of population-weighted meteorology for each balancing
# authority (BA).
wrf_to_tell_balancing_authorities(df, wrf_output_dir)

## Compile hourly load, hourly population and hourly WRF data for MLP model ##
compile_output_dir = f'{tell_data_dir}/compiled_data'

tell.compile_data(eia_930_output_dir, pop_output_dir, wrf_output_dir, compile_output_dir)

import time

output_dir =  f'{tell_data_dir}/mlp_output'
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
mlp_input_dir = 'f'{tell_data_dir}/'mlp_output'/{year_to_process}'
ba_geolocation_input_dir = tell_data_dir
gcam_usa_input_dir = f'{tell_data_dir}/{gcam_usa_scenario}'
data_output_dir = 'f'{tell_data_dir}/'mlp_state_output'/{year_to_process}'

#Run the MLP model forawrd in time and
tell.execute_forward(year_to_process, mlp_input_dir, ba_geolocation_input_dir,
                     pop_input_dir, gcam_usa_input_dir, data_output_dir)