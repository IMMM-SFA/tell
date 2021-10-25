import tell
import pandas as pd
import os

tell_data_dir = 'C:/Users/mcgr323/projects/tell_valid/tell/tell_raw_data'

# set the target year
target_year = 2015

# directory containing the outputs
output_dir = os.path.join(tell_data_dir, 'outputs')

# paths to files
fips_file = os.path.join(tell_data_dir, 'state_and_county_fips_codes.csv')
service_area_file = os.path.join(tell_data_dir, f'Service_Territory_{target_year}.xlsx')
sales_ult_file = os.path.join(tell_data_dir, f'Sales_Ult_Cust_{target_year}.xlsx')
bal_auth_file = os.path.join(tell_data_dir, f'Balancing_Authority_{target_year}.xlsx')

# prepare data
tell.process_data(target_year, fips_file, service_area_file, sales_ult_file, bal_auth_file, output_dir)

output = os.path.join(output_dir, f'fips_service_match_{target_year}.csv')
output = pd.read_csv(output)
output2 = output.drop_duplicates()
output_file = os.path.join(output_dir, f'fips_service_match_rm{target_year}.csv')
output2.to_csv(output_file, sep=',', index=False)

# Hourly load data (EIA 930):

# Set the data input and output directories:
EIA_930_input_dir = '//connie-1/im3fs/tell/inputs/raw_data/EIA_930/Balancing_Authority'
EIA_930_output_dir = 'C:/Users/mcgr323/projects/tell/BA_hourly_inputs/BA_Hourly_Load'

# Process the hourly load data
tell.process_eia_930(EIA_930_input_dir, EIA_930_output_dir)

# County population data:

# Set the data input and output directories:
population_input_dir = '//connie-1/im3fs/tell/inputs'
mapping_input_dir = '//connie-1/im3fs/tell/inputs/Utility_Mapping/CSV_Files'
pop_output_dir = 'C:/Users/mcgr323/projects/tell/BA_hourly_inputs/BA_population'

# Set some processing flags:
start_year = 2015;  # Starting year of time series
end_year = 2019;  # Ending year of time series

# Meteorlogy data: WRF

# TODO add im3components functions for wrf_to_tell_part_one.py and wrf_to_tell_part_two.py

WRF_output_dir = 'C:/Users/mcgr323/projects/tell/BA_hourly_inputs/BA_WRF'

# Compile hourly load, hourly population and hourly WRF data for MLP model

EIA_930_output_dir = 'C:/Users/mcgr323/projects/tell/BA_hourly_inputs/BA_Hourly_Load'
pop_output_dir = 'C:/Users/mcgr323/projects/tell/BA_hourly_inputs/BA_population'
WRF_output_dir = 'C:/Users/mcgr323/projects/tell/BA_hourly_inputs/BA_WRF'