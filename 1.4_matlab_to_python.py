"""
MATLAB to Python conversion from: Process_Population_Time_Series_Within_BA_Domain.m
MATLAB: Casey D. Burleyson
Python: Casey R. McGrath

This script takes as .mat files containing the county mapping of
utilities and balancing authorities (BAs) and computes the annual
total population in counties covered by that BA. Those populations
are then interpolated to an hourly resolution in order to match the
temporal resolution of the load and meteorology time series.
The output file format is given below. The script takes as input the
years to process as well as paths to the relevant input and output directories.
All of the required input files are stored on PIC at /projects/im3/tell/inputs/.
The script relies on one function that provides BA metadata based on the
EIA BA number: EIA_930_BA_Information_From_BA_Number.m.

  .csv output file format:
  C1: Year
  C2: Month
  C3: Day
  C4: Hour
  C5: Total population within the BA domain
"""

import os
import glob
import pandas as pd

# Set some processing flags:
start_year = 2015; # Starting year of time series
end_year = 2019; # Ending year of time series

# Set the data input and output directories:
population_data_input_dir = 'C:/Users/mcgr323/OneDrive - PNNL/Desktop/raw_data/inputs';
service_territory_data_input_dir = 'C:/Users/mcgr323/OneDrive - PNNL/Desktop/raw_data/inputs/Utility_Mapping/CSV_Files';
csv_data_output_dir = 'C:/Users/mcgr323/OneDrive - PNNL/Desktop/raw_data/inputs/BA_Hourly_Population/CSV_Files';

# load population data
df_population = pd.read_csv(population_data_input_dir +'/county_populations_2000_to_2019.csv')

# load FIPS county data for BA number and FIPs code matching for later population sum by BA
all_files = glob.glob(service_territory_data_input_dir + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
col_names = ['year', 'county_fips', 'ba_number', 'ba_abbreviation','ba_name']

# only keep columns that are needed
frame = frame[col_names].copy()

# select for valid BA numbers (from BA metadata)
filename = 'C:/Users/mcgr323/projects/tell/EIA_BA_match.csv'
metadata = pd.read_csv(filename, index_col=None, header=0)



