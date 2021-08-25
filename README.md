![build](https://github.com/IMMM-SFA/tell/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/IMMM-SFA/tell/branch/package/graph/badge.svg?token=URP1KWRI6U)](https://codecov.io/gh/IMMM-SFA/tell)

# TELL
The Total ELectricity Load (TELL) model provides a framework that integrates aspects of both short- and long-term predictions of electricity demand in a coherent and scalable way. TELL takes as input gridded hourly time-series of meteorology and uses the temporal variations in weather to predict hourly profiles of total electricity demand for every county in the lower 48 United States using a multilayer perceptron (MLP) approach. Hourly predictions from TELL are then scaled to match the annual state-level total electricity loads predicted by the U.S. version of the Global Change Analysis Model (GCAM-USA). GCAM-USA is designed to capture the long-term co-evolution of the human-Earth system. Using this unique approach allows TELL to reflect both changes in the shape of the load profile due to variations in weather and climate and the long-term evolution of energy demand due to changes in population, technology, and economics. TELL is unique from other probabilistic load forecasting models in that it features an explicit spatial component that allows us to relate predicted loads to where they would occur spatially within a grid operations model.
## Examples
import tell
import time

data_dir = 'C:/Users/mcgr323/OneDrive - PNNL/Desktop/WRF_CSV_Files/CSV_Files'
output_dir = 'C:/Users/mcgr323/OneDrive - PNNL/Desktop/WRF_predict_output'
batch_run = True
target_ba_list = None
generate_plots = True

t0 = time.time()

df = predict(data_dir=data_dir,
                  out_dir=output_dir,
                  batch_run=batch_run,
                  target_ba_list=target_ba_list,
                  generate_plots=generate_plots)
