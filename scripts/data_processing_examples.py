import os
import glob
import pandas as pd
import datetime
import tell

EIA_930_input_dir = '//connie-1/im3fs/tell/inputs/raw_data/EIA_930/Balancing_Authority'
EIA_930_output_dir = 'C:/Users/mcgr323/projects/tell/BA_hourly_inputs/BA_Hourly_Load';

tell.process_EIA_930(EIA_930_input_dir, EIA_930_output_dir)