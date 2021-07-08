"""
MATLAB to Python conversion from: Process_Composite_BA_Data.m.m
MATLAB: Casey D. Burleyson
Python: Casey R. McGrath


This script takes the time-series of average meteorology, total population, and load
in a given balancing authority (BA) and combines all of the data together into a single array.
The needed input files are stored on PIC at: /projects/im3/tell/.The output file format is
given below. This script corresponds to needed functionality 1.5 on this Confluence page:
https://immm-sfa.atlassian.net/wiki/spaces/IP/pages/1732050973/2021-02-22+TELL+Meeting+Notes.
The script relies on one function that provides BA metadata based on the
EIA BA number: EIA_930_BA_Information_From_BA_Number.m. All times are in UTC.
Missing values are reported as -9999 in the .csv output files and as NaNs in the .mat output files.


  .csv output file format:
  C1:  Year
  C2:  Month
  C3:  Day
  C4:  Hour
  C5:  U.S. Census Bureau total population in the counties the BA covers
  C6:  Population-weighted NLDAS temperature in K
  C7:  Population-weighted NLDAS specific humidity in kg/kg
  C8:  Population-weighted NLDAS downwelling shortwave radiative flux in W/m^2
  C9:  Population-weighted NLDAS downwelling longwave radiative flux in W/m^2
  C10: Population-weighted NLDAS wind speed in m/s
  C11: EIA 930 adjusted forecast demand in MWh
  C12: EIA 930 adjusted demand in MWh
  C13: EIA 930 adjusted generation in MWh
  C14: EIA 930 adjusted net interchange with adjacent balancing authorities in MWh
"""

# Set the data input and output directories:
load_data_input_dir = 'C:/Users/mcgr323/projects/tell/raw_data/inputs/BA_Hourly_Load/CSV_Files/';
population_data_input_dir = 'C:/Users/mcgr323/projects/tell/raw_data/inputs/BA_Hourly_Population/CSV_Files/';
meteorology_data_input_dir = 'C:/Users/mcgr323/projects/tell/raw_data/inputs/BA_Hourly_Meteorology/CSV_Files/';
csv_data_output_dir = '/Users/burl878/OneDrive - PNNL/Documents/IMMM/Data/TELL_Input_Data/inputs/Composite_BA_Hourly_Data/CSV_Files/';