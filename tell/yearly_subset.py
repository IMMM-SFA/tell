import pandas as pd
import klib

from tell.match import *


def prepare_data(ferc_hourly_file, ferc_resp_eia_code, eia_operators_nerc_region_mapping):
    """Load and prepare data.  Reduce complexity by making column names lower case and agree through data sets,
    deleting duplicates, splitting out commonly known trailing words that do not exist in all data sets.

    :param ferc_hourly_file:                      Dataframe of hourly FERC load data
    :type ferc_hourly_file:                       str

    :param ferc_resp_eia_code:                    Dataframe of FERC respondents with EIA code
    :type ferc_resp_eia_code:                     str

    :param eia_operators_nerc_region_mapping:     Mapping file of EIA codes and their asscioated NERC region mapping
    :type eia_operators_nerc_region_mapping:      str

    :return:                                    [0] df_ferc_hrly: Dataframe respondent_id, date and generation sum
                                                [1] df_ferc_resp: Dataframe of cleaned FERC respondent IDs and EIA code
                                                [2] df_eia_mapping: Cleaned mapping file

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

    # change wide format to long format
    df_ferc_hrly = pd.melt(df_ferc_hrly, id_vars=df_ferc_hrly.filter(like='hour').columns, var_name='hour', value_name='generation')

    # filter for generation values over 0
    df_ferc_hrly = df_ferc_hrly[df_ferc_hrly['generation'] > 0]

    # exclude hour 25
    df_ferc_hrly = df_ferc_hrly[df_ferc_hrly['hour'] != 25]

    # change the datetime structure
    df_ferc_hrly['date'] = pd.to_datetime(df_ferc_hrly['plan_date'], format='%d%b%Y:%H:%M:%S.%f')

    # subset for only respondent_id, date and generation columns
    df_ferc_hrly = df_ferc_hrly[["respondent_id", "date", "generation"]]

    return df_ferc_hrly, df_ferc_resp, df_eia_mapping


def merge_and_subset(df_ferc_hrly, df_ferc_resp, df_eia_mapping, year):
    """Merge cleaned and prepared datasets subset by year and write to csv

    :param df_ferc_hrly:              Dataframe respondent_id, date and generation sum from prepare_data
    :type df_ferc_hrly:               str

    :param df_ferc_resp:              Dataframe of cleaned FERC respondent IDs and EIA code from prepare_data
    :type df_ferc_resp:               str

    :param df_eia_mapping:            Cleaned mapping file from prepare_data
    :type df_eia_mapping:             str

    :param year:                      Year to subset data for
    :type year:                       str

    :return:                          Dataframe of FERC hourly loads for yearly subset

    """

    # merge hourly data with
    df_ferc_hrly = df_ferc_hrly.merge(df_ferc_resp, on='respondent_id', how='left', indicator=True)
    df_ferc_hrly = df_ferc_hrly.drop('_merge', axis=1)

    # create a unified key to merge by changing Operator_ID to eia_code
    df_eia_mapping.columns = ["eia_code", 'NERC_region']

    # merge the hourly data with the mapping file
    df_ferc_hrly = df_ferc_hrly.merge(df_eia_mapping, on='eia_code', how='left', indicator=True)
    df_ferc_hrly = df_ferc_hrly.drop('_merge', axis=1)

    # subset by year
    df_ferc_hrly = df_ferc_hrly[df_ferc_hrly['report_yr'] == year]

    # group by the NERC_region and date
    df_ferc_hrly = df_ferc_hrly.groupby(['NERC_region', 'date']).agg({'generation':['sum']})

    return df_ferc_hrly