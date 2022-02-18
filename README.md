![build](https://github.com/IMMM-SFA/tell/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/IMMM-SFA/tell/branch/package/graph/badge.svg?token=URP1KWRI6U)](https://codecov.io/gh/IMMM-SFA/tell)

# TELL

`tell` is an open-source Python package for predicting future electricty load in the Lower 48 United States.

## A little about `tell`

The Total ELectricity Load (TELL) model provides a framework that integrates aspects of both short- and long-term predictions of electricity demand in a coherent and scalable way. tell takes as input gridded hourly time-series of meteorology and uses the temporal variations in weather to predict hourly profiles of total electricity demand for every county in the l ower 48 United States using a multilayer perceptron (MLP) approach. Hourly predictions from tell are then scaled to match the annual state-level total electricity loads predicted by the U.S. version of the Global Change Analysis Model (GCAM-USA). GCAM-USA is designed to capture the long-term co-evolution of the human-Earth system. Using this unique approach allows tell to reflect both changes in the shape of the load profile due to variations in weather and climate and the long-term evolution of energy demand due to changes in population, technology, and economics. tell is unique from other probabilistic load forecasting models in that it features an explicit spatial component that allows us to relate predicted loads to where they would occur spatially within a grid operations model.

## Lets get started! 

In this quickstarter we will walk through a subset of the data used in `tell`, starting with importing the package and ending with data visualization. This allows the user to walk through the entire `tell` package in a matter of minutes. If you have more questions please feel free to visit the [Read the Docs](https://immm-sfa.github.io/tell/index.html) site for `tell`.

### Load necessary packages
```buildoutcfg
import tell
import os 
import time 
```

### Install package data

**NOTE: The package data will require approximately 1.4 GB of storage.**

Set the local directory where you would like to store the package data and run the function below:
```buildoutcfg
# Create directory to store raw data
current_dir =  os.path.dirname(os.getcwd())
current_dir =  os.path.join(os.path.dirname(os.getcwd()), r'tell_valid')
raw_data_dir = os.path.join(current_dir, r'raw_data', r'demand_and_population')
if not os.path.exists(raw_data_dir):
   os.makedirs(raw_data_dir)

# Download the raw data from the Zenodo package
tell.install_package_data(data_dir = raw_data_dir)
```
## 1. Data pre-processing for TELL

In the next few code blocks we will load and manipulate the nescessary data for the `tell`  package. This consists of hourly load, population and meteorology for the CONUS, which will be loaded in from raw data sources, manipulated and then compiled together to use as input for the MLP model. Please follow the steps below to produce the hourly input data, if you have already finished this step you can proceed to **2. Model training and prediction**
    
### 1.0 Spaitally mapping the Balancing Authorities (BAs)

The code chunk below brings in the unique spatial component of <tell>, where we map the Balancing Authorities (BAs) to the Federal Information Processing Standard Publication (FIPS) codes. This allows us to assign load where it occurs spatially within the CONUS.  
```buildoutcfg
## FIPS to BA code mapping ##
#Set the start and end year for processing
start_year = 2015
end_year = 2019

tell.map_fips_codes(start_year, end_year,raw_data_dir, current_dir)
```

### 1.1 Hourly load

Here we load in the raw EIA 930 hourly load profiles for all Balancing Authorities (BAs), subset for the wanted columns only and then output the hourly load as csvs to be compiled later with population, and meteorology to be fed to the MLP model downstream for predict future load.  
```buildoutcfg
# Set the data input and output directories:
eia_930_input_dir = raw_data_dir
eia_930_output_dir = os.path.join(current_dir, r'outputs', r'hourly_ba_load')
if not os.path.exists(eia_930_output_dir):
   os.makedirs(eia_930_output_dir)

# Process the hourly load data
tell.process_eia_930(eia_930_input_dir, eia_930_output_dir)
```

### 1.2 Population data

For this data processing step we will load in the annual population by FIPS code, merge by FIPS code to get the correspondng BA number, sum by  year and BA number and then interpolate the annual population to hourly population in order to feed it to the MLP model downstream. 

```buildoutcfg
# Set the data input and output directories:
population_input_dir = raw_data_dir
map_input_dir = os.path.join(current_dir, r'outputs', r'fips_mapping_files')
population_output_dir =  os.path.join(current_dir, r'outputs', r'hourly_population')
if not os.path.exists(pop_output_dir):
   os.makedirs(pop_output_dir)

tell.ba_pop_interpolate(map_input_dir, pop_input_dir, pop_output_dir, start_year, end_year)
```

### 1.3 Meteorology data

Here we use the <im3components> package to load in the WRF meterology data, average WRF meteorology by county and then aggregate them into annual hourly time-series of population-weighted meteorology for each balancing authority (BA). All times are in UTC. Missing values are reported as -9999. First we download a subset of the wrf data from the Zenodo package to work with in this quickstarter. For thi subset we choose the target year of 2019.

```buildoutcfg
# Create input directory meterology data #
wrf_input_dir =  os.path.join(current_dir, r'raw_data', r'wrf')
if not os.path.exists(wrf_input_dir):
   os.makedirs(wrf_input_dir)

# Download the raw sample wrf data from the Zenodo package
tell.install_sample_data(data_dir = wrf_input_dir)

# Create output directory meteorology data #
wrf_output_dir =  os.path.join(current_dir, r'outputs', r'hourly_meteorology')
if not os.path.exists(wrf_output_dir):
   os.makedirs(wrf_output_dir)

# set the target year 
target_yr = 2019

# Process wrf data to put into right date format
tell.process_wrf(wrf_input_dir, wrf_output_dir, target_yr, n_jobs=-1)
```

### 1.4 Compile hourly load, population and meterology data 

Here we compile all the data processing steps above for hourly load (EIA 930), population (county FIPS) and meteorology (WRF) to get a final cleaned up dataset to use as an input to the MLP model. 
```buildoutcfg
# create directory to store the compiled data
compile_output_dir =  os.path.join(current_dir, r'outputs', r'compiled_data')
if not os.path.exists(compile_output_dir):
   os.makedirs(compile_output_dir)

#set target year for WRF data
target_yr = 2019

# compile the hourly load data, population data, and wrf climate data by date
tell.compile_data(eia_930_output_dir, population_output_dir, wrf_output_dir, target_yr, compile_output_dir)
```

## 2. Model training and prediction

This step takes the data processed and compiled above and runs a multilayer perceptron (MLP) model to predict future hourly load. Start-time is the start-time for analysis, end-time is the end time for analysis and start_test_period is the timestamp splitting train and test data.
```buildoutcfg
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
start_test_period = "2019-06-01 00:00:00"

t0 = time.time()

tell.predict(compile_output_dir ,
            mlp_output_dir,
            start_time = start_time,
            end_time = end_time
            start_test_period = start_test_period,
            batch_run = batch_run,
            target_ba_list = target_ba_list,
            generate_plots = generate_plots)ompile_data(eia_930_output_dir, population_output_dir, wrf_output_dir, target_yr, compile_output_dir)
```

## 3. Model forward execution

This script takes the .csv files produced by the TELL MLP model and distributes the predicted load to the counties that each balancing authority (BA) operates in. The county-level hourly loads are then summed to the state-level and scaled to match the state-level annual loads produced by GCAM-USA. Three sets of output files are generated: county-level hourly loads, state-level hourly loads, and hourly loads for each BA. There is one additional summary output file that includes state-level annual loads from TELL and GCAM-USA as well as the scaling factors.

Please set the directories below to your local machine preferences and run the tell.execute_forward function. 

```buildoutcfg
# Set the year to process:
year_to_process = '2020'

# Set the data input and output directories:
mlp_input_dir = os.path.join(current_dir, r'outputs', r'mlp_output')
ba_geolocation_input_dir = os.path.join(current_dir, r'outputs', r'fips_mapping_files')
gcam_usa_input_dir = raw_data_dir
data_output_dir = os.path.join(current_dir, r'outputs', r'forward_output', rf'{year_to_process}')
if not os.path.exists(data_output_dir):
   os.makedirs(data_output_dir)


# Run the MLP model forward in time and
tell.execute_forward(year_to_process, mlp_input_dir, ba_geolocation_input_dir,
                     population_input_dir, gcam_usa_input_dir, data_output_dir)
```
4. Model visualization
Below are a few select model visualizations to check on model performance for select states and BAs

```buildoutcfg
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

sample_state = 'California'
sample_ba = 'ERCO'

# If you want to save the images, check that the image output directory exist and if not then create it:
if save_images == 1:
    image_output_dir = os.path.join(image_output_dir, rf'{year_to_plot}')
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)

# Plot a map of the state scaling factors:
tell.plot_state_scaling_factors(shapefile_input_dir, data_input_dir, year_to_plot, save_images, image_resolution,
                                image_output_dir)

# Plot the state annual total loads from GCAM-USA and TELL:
tell.plot_state_annual_total_loads(state_summary_df, year_to_plot, save_images, image_resolution, image_output_dir)

# Plot the time-series of total hourly loads for a given state by specifying either the state name or FIPS code:
tell.plot_state_load_time_series(state, state_hourly_load_df, year_to_plot, save_images, image_resolution,
                                 image_output_dir)

# Plot the load duration curve for a given state by specifying either the state name or FIPS code:
tell.plot_state_load_duration_curve(state, state_hourly_load_df, year_to_plot, save_images, image_resolution,
                                    image_output_dir)

# Plot the time-series of total hourly loads for a given BA by specifying either the BA abbreviation or BA number:
tell.plot_ba_load_time_series(sample_ba, ba_hourly_load_df, year_to_plot, save_images, image_resolution, image_output_dir)
```

