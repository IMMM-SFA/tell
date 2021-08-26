![build](https://github.com/IMMM-SFA/tell/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/IMMM-SFA/tell/branch/package/graph/badge.svg?token=URP1KWRI6U)](https://codecov.io/gh/IMMM-SFA/tell)

# TELL
The Total ELectricity Load (TELL) model provides a framework that integrates aspects of both short- and long-term predictions of electricity demand in a coherent and scalable way. TELL takes as input gridded hourly time-series of meteorology and uses the temporal variations in weather to predict hourly profiles of total electricity demand for every county in the lower 48 United States using a multilayer perceptron (MLP) approach. Hourly predictions from TELL are then scaled to match the annual state-level total electricity loads predicted by the U.S. version of the Global Change Analysis Model (GCAM-USA). GCAM-USA is designed to capture the long-term co-evolution of the human-Earth system. Using this unique approach allows TELL to reflect both changes in the shape of the load profile due to variations in weather and climate and the long-term evolution of energy demand due to changes in population, technology, and economics. TELL is unique from other probabilistic load forecasting models in that it features an explicit spatial component that allows us to relate predicted loads to where they would occur spatially within a grid operations model.
## Example

###Data Download

TODO: 1.0 Automatically download a data package containing the EIA-861, EIA-930, and census population datasets

###Data Processing

1.1 Use the EIA-861 dataset to create a mapping between BAs and counties

```buildoutcfg
import os

import tell


if __name__ == '__main__':

    # set the target year
    target_year = 2015

    # get the input directory as it currently exists within this repository
    input_dir = os.path.join(os.path.dirname(__file__), 'inputs')

    # get the path to the EIA raw data for the target year
    eia_data_dir = os.path.join(input_dir, 'EIA_861', 'Raw_Data', str(target_year))

    # directory containing the outputs
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')

    # paths to files
    fips_file = os.path.join(input_dir, 'state_and_county_fips_codes.xlsx')
    service_area_file = os.path.join(eia_data_dir, f'Service_Territory_{target_year}.xlsx')
    sales_ult_file = os.path.join(eia_data_dir, f'Sales_Ult_Cust_{target_year}.xlsx')
    bal_auth_file = os.path.join(eia_data_dir, f'Balancing_Authority_{target_year}.xlsx')

    # prepare data
    tell.process_data(target_year, fips_file, service_area_file, sales_ult_file, bal_auth_file, output_dir)


    output = os.path.join(output_dir, f'fips_service_match_{target_year}.csv')
    output = pd.read_csv(output)
    output2 = output.drop_duplicates()
    output_file = os.path.join(output_dir, f'fips_service_match_rm{target_year}.csv')
    output2.to_csv(output_file, sep=',', index=False)
```

**1.2 Convert the EIA-930 BA demand dataset from Excel files into CSV files**

TODO 

**1.3 Average WRF meteorology within each BA’s territory**

TODO

**1.4 Compute time-series of total population within each BA’s territory**

TODO

**1.5 Merge the hourly-time series of meteorology from 1.3, population from 1.4, and demand for each BA from 1.2 into a single CSV file that can be used as input to 2.2**

TODO


###Run the MLP Simulation

**2.2 Build the MLP models that relate historical variations in meteorology and population to the historical time-series of hourly demand in each BA (@Aowabin Rahman). The code should have options for the user to specify the training and evaluation periods.**
```
#set the input and output directories 
data_dir = 'C:/Users/mcgr323/OneDrive - PNNL/Desktop/WRF_CSV_Files/CSV_Files'
output_dir = 'C:/Users/mcgr323/OneDrive - PNNL/Desktop/WRF_predict_output'

#set parameters 
batch_run = True
target_ba_list = None
generate_plots = True

#initally time
t0 = time.time()

#run the model 
df = tell.predict(data_dir=data_dir,
                  out_dir=output_dir,
                  batch_run=batch_run,
                  target_ba_list=target_ba_list,
                  generate_plots=generate_plots)
```

