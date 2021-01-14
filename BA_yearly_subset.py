import os
import pandas as pd
import klib
import re

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

    # cleaning the column names, dropping empty and virtually empty columns,
    # removes single valued columns,drops duplicate rows and memory reduction
    df_ferc_hrly = klib.data_cleaning(df_ferc_hrly)
    df_ferc_resp = klib.data_cleaning(df_ferc_resp)
    df_eia_mapping = klib.data_cleaning(df_eia_mapping)

    # Get columns to melt
    colsAll = df_ferc_hrly.columns
    colsVal = [i for i in colsAll if re.search(r'^hour\d\d$', i)]
    colsId = set(colsAll) - set(colsVal)

    # change wide format to long format
    df_ferc_hrly = pd.melt(df_ferc_hrly,
                 id_vars= colsId,
                 value_vars= colsVal,
                 var_name='hour',
                 value_name='generation')

    # filter for generation values over 0
    df_ferc_hrly = df_ferc_hrly[df_ferc_hrly['generation'] > 0]

    # exclude hour 25
    df_ferc_hrly['hour'] = pd.to_numeric(df_ferc_hrly['hour'].str.replace('hour',''))
    df_ferc_hrly = df_ferc_hrly[df_ferc_hrly['hour'] != 25]

    #change the datetime structure
    df_ferc_hrly.loc[:,'date'] = pd.to_datetime(df_ferc_hrly['plan_date'],
                                          format='%m/%d/%Y %H:%M:%S')
    df_ferc_hrly['date'] += pd.to_timedelta(df_ferc_hrly.hour-1, unit='h')

    #subset for only respondent_id, date and generation columns
    df_ferc_hrly = df_ferc_hrly[["respondent_id", "report_yr", "date", "generation"]]
    return df_ferc_hrly, df_ferc_resp, df_eia_mapping


def merge_and_subset(df_ferc_hrly, df_ferc_resp, df_eia_mapping, year):
    """Merge cleaned and prepared datasets subset by year and write to csv"""

    #merge hourly data with
    df_ferc_hrly = df_ferc_hrly.merge(df_ferc_resp, on='respondent_id', how='left', indicator=True)
    df_ferc_hrly = df_ferc_hrly.drop('_merge', axis=1)

    # create a unified key to merge by changing Operator_ID to eia_code
    df_eia_mapping.columns = ["eia_code", 'NERC_region']

    #merge the hourly data with the mapping file
    df_ferc_hrly = df_ferc_hrly.merge(df_eia_mapping, on='eia_code', how='left', indicator=True)
    df_ferc_hrly = df_ferc_hrly.drop('_merge', axis=1)

    # substep by year:need code
    df_ferc_hrly = df_ferc_hrly[df_ferc_hrly['report_yr'] == year]

    #group by the NERC_region and date
    df_ferc_hrly = df_ferc_hrly.groupby(['NERC_region', 'date']).agg({'generation':['sum']})

    return df_ferc_hrly


#run the prepare data function
df_ferc_hrly, df_ferc_resp, df_eia_mapping = prepare_data(ferc_hourly_file, ferc_resp_eia_code,
                                                          eia_operators_nerc_region_mapping)
#run the merge and subset by year function
ferc_hrly_year = merge_and_subset(df_ferc_hrly, df_ferc_resp, df_eia_mapping, year)

ferc_hrly_year.to_csv('FERC_hrly_2010.csv', sep=',')