# set the target year
from tell import prepare_data_yearly

target_year = 2019

# get the input directory as it currently exists within this repository
input_dir = os.path.join(os.path.dirname(__file__), 'inputs')

# get the path to the EIA raw data for the target year
ferc_data_dir = os.path.join(input_dir, 'FERC_714')

# directory containing the outputs
output_dir = os.path.join(os.path.dirname(__file__), 'FERC_outputs')

# paths to files
ferc_hourly_file = os.path.join(ferc_data_dir, 'FERC_hourly_gen.csv')
ferc_resp_eia_code = os.path.join(ferc_data_dir, 'Respondent_IDs_fix_mismatch.csv')
eia_operators_nerc_region_mapping = os.path.join(ferc_data_dir, 'eia_operators_nerc_region_mapping.csv')

df_ferc_hrly, df_ferc_resp, df_eia_mapping = prepare_data_yearly(ferc_hourly_file, ferc_resp_eia_code, eia_operators_nerc_region_mapping)
