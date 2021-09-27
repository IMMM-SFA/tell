import tell

# Hourly load data (EIA 930):

# Set the data input and output directories:
EIA_930_input_dir = '//connie-1/im3fs/tell/inputs/raw_data/EIA_930/Balancing_Authority'
EIA_930_output_dir = 'C:/Users/mcgr323/projects/tell/BA_hourly_inputs/BA_Hourly_Load'

# Process the hourly load data
tell.process_EIA_930(EIA_930_input_dir, EIA_930_output_dir)

# County population data:

# Set the data input and output directories:
population_input_dir = '//connie-1/im3fs/tell/inputs'
mapping_input_dir = '//connie-1/im3fs/tell/inputs/Utility_Mapping/CSV_Files'
pop_output_dir = 'C:/Users/mcgr323/OneDrive - PNNL/Desktop/inputs/BA_Hourly_Population/CSV_Files'

# Set some processing flags:
start_year = 2015;  # Starting year of time series
end_year = 2019;  # Ending year of time series
