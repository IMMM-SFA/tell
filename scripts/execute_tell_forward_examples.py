import tell

# Set the year and GCAM-USA scenario to process:
year_to_process = '2020'

# Set the data input and output directories:
mlp_input_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/outputs/MLP_Model_Output/' + year_to_process + '/'
ba_geolocation_input_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/inputs/Utility_Mapping/CSV_Files/'
population_input_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/inputs/'
gcam_usa_input_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/forward_execution/GCAM_USA_Forcing/Raw_Data/' + gcam_usa_scenario + '/'
data_output_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/outputs/' + year_to_process + '/'

#Run the MLP model forawrd in time and
tell.execute_forward(year_to_process, mlp_input_dir, ba_geolocation_input_dir,
                     population_input_dir, gcam_usa_input_dir, data_output_dir)
