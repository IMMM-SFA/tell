import os
import numpy as np
import pandas as pd
import klib

data_dir = 'C:\\Users\\mcgr323\\projects\\tell\\inputs\\FERC_714\\'

# paths to files
ferc_hourly_file = os.path.join(data_dir, 'FERC_hourly_gen.csv')
ferc_resp_eia_code = os.path.join(data_dir, 'Respondent_IDs_fix_mismatch.csv')
eia_operators_nerc_region_mapping = os.path.join(data_dir, 'eia_operators_nerc_region_mapping.csv')


def prepare_data(ferc_hourly_file, ferc_resp_eia_code, eia_operators_nerc_region_mapping):
    """Load and prepare data.  Reduce complexity by making column names lower case and agree through data sets,
    deleting duplicates, splitting out commonly known trailing words that do not exist in all data sets.
    """
    # read in data
    df_ferc_hrly = pd.read_csv(ferc_hourly_file)
    df_ferc_resp = pd.read_csv(ferc_resp_eia_code)
    df_eia_mapping = pd.read_csv(eia_operators_nerc_region_mapping)

    # create a unified key to merge by changing Operator_ID to eia_code
    df_eia_mapping.rename(columns={"Operator.ID": "eia_code"})

    # cleaning the column names, dropping empty and virtually empty columns,
    # removes single valued columns,drops duplicate rows and memory reduction
    df_ferc_hrly = klib.data_cleaning(df_ferc_hrly)
    df_ferc_resp = klib.data_cleaning(df_ferc_resp)
    df_eia_mapping = klib.data_cleaning(df_eia_mapping)

    # change wide format to long format
    df_ferc_hrly = pd.melt(df_ferc_hrly, id_vars=match("^hour..$"), var_name='hour', value_name='generation')

    # filter for generation values over 0
    df_ferc_hrly = df_ferc_hrly[df_ferc_hrly['generation'] > 0]

    # exclude hour 25
    df_ferc_hrly = df_ferc_hrly[df_ferc_hrly['hour'] != 25]

    df_ferc_hrly.to_datetime

    return df_ferc_hrly, df_ferc_resp, df_eia_mapping


df_ferc_hrly, df_ferc_resp, df_eia_mapping = prepare_data(ferc_hourly_file, ferc_resp_eia_code,
                                                          eia_operators_nerc_region_mapping)
