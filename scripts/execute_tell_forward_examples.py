import tell
import datetime
import glob

# Set a time variable to benchmark the run time:
begin_time = datetime.datetime.now()

# Set the year and GCAM-USA scenario to process:
year_to_process = '2020'
gcam_usa_scenario = 'scenario_name'

# Set the data input and output directories:
mlp_input_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/outputs/MLP_Model_Output/' + year_to_process + '/'
ba_geolocation_input_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/inputs/Utility_Mapping/CSV_Files/'
population_input_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/inputs/'
gcam_usa_input_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/forward_execution/GCAM_USA_Forcing/Raw_Data/' + gcam_usa_scenario + '/'
data_output_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/outputs/' + year_to_process + '/'

# Load in the accompanying GCAM-USA output file and subset to the "year_to_process":
gcam_usa_df = tell.extract_gcam_usa_loads((gcam_usa_input_dir + 'gcamDataTable_aggParam.csv'))
gcam_usa_df = tell.gcam_usa_df[gcam_usa_df['Year'] == int(year_to_process)]

# Load in the most recent (e.g., 2019) BA service territory map and simplify the dataframe:
ba_mapping = pd.read_csv((ba_geolocation_input_dir + 'ba_service_territory_2019.csv'), index_col=None, header=0)

# Load in the population data and simplify the dataframe:
population = pd.read_csv(population_input_dir + '/county_populations_2000_to_2019.csv')
population = population[{'county_FIPS', 'pop_2019'}].copy(deep=False)
population.rename(columns={"county_FIPS":"County_FIPS",
                           "pop_2019":"Population"}, inplace=True)

# Merge the ba_mapping and population dataframes together. Compute the fraction of the
# total population in each BA that lives in a given county:
mapping_df = ba_mapping.merge(population, on=['County_FIPS'])
mapping_df = mapping_df.sort_values("BA_Number")
mapping_df['BA_Population_Sum'] = mapping_df.groupby('BA_Code')['Population'].transform('sum')
mapping_df['BA_Population_Fraction'] = mapping_df['Population'] / mapping_df['BA_Population_Sum']
mapping_df = mapping_df.dropna()
del population, ba_mapping

# Create a list of all of the MLP output files in the "data_input_dir" and aggregate the files
# in that list using the "aggregate_mlp_output_files" function:
mlp_output_df = tell.aggregate_mlp_output_files(sorted(glob.glob(os.path.join(mlp_input_dir + '*_mlp_predictions.csv'))))

# Merge the "mapping_df" with "mlp_output_df":
joint_mlp_df = pd.merge(mlp_output_df, mapping_df, on='BA_Code')

# Scale the BA loads in each county by the fraction of the BA's total population that lives there:
joint_mlp_df['County_BA_Load_MWh'] = joint_mlp_df['Total_BA_Load_MWh'].mul(joint_mlp_df['BA_Population_Fraction'])

# Sum the county-level hourly loads into annual state-level total loads and convert that value from MWh to TWh:
joint_mlp_df['TELL_State_Annual_Load_TWh'] = (joint_mlp_df.groupby('State_FIPS')['County_BA_Load_MWh'].transform('sum')) / 1000000

# Add a column with the state-level annual total loads from GCAM-USA:
joint_mlp_df = pd.merge(joint_mlp_df, gcam_usa_df[['State_FIPS', 'GCAM_USA_State_Annual_Load_TWh']], on='State_FIPS', how='left')

# Compute the state-level scaling factors that force TELL annual loads to match GCAM-USA annual loads:
joint_mlp_df['State_Scaling_Factor'] = joint_mlp_df['GCAM_USA_State_Annual_Load_TWh'].div(joint_mlp_df['TELL_State_Annual_Load_TWh'])

# Apply those scaling factors to the "County_BA_Load_MWh" value:
joint_mlp_df['County_BA_Load_MWh_Scaled'] = joint_mlp_df['County_BA_Load_MWh'].mul(joint_mlp_df['State_Scaling_Factor'])

# Output the data using the output functions:
tell.output_tell_summary_data(joint_mlp_df, data_output_dir)
tell.output_tell_ba_data(joint_mlp_df, data_output_dir)
tell.output_tell_state_data(joint_mlp_df, data_output_dir)
tell.output_tell_county_data(joint_mlp_df, data_output_dir)

# Output the elapsed time in order to benchmark the run time:
print('Elapsed time = ', datetime.datetime.now() - begin_time)
