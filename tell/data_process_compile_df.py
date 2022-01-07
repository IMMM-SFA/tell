import pandas as pd
import os
import glob as glob

def compile_data(eia_930_output_dir, pop_output_dir, wrf_output_dir, compile_output_dir):
    """Read in population data, format columns and return single df for all years
    :param population_input_dir:               Directory where county population is stored
    :type population_input_dir:                dir
    :param start_year:                         Year to start model ; four digit year (e.g., 1990)
    :type start_year:                          int
    :param end_year:                           Year to start model ; four digit year (e.g., 1990)
    :type end_year:                            int
    :return:                                   Dataframe of valid population data for select timeframe
    """
    ba_name = ['NBSO', 'AEC', 'YAD', 'AMPL', 'AZPS', 'AECI', 'BPAT', 'CISO', 'CPLE', 'CHPD', 'CEA', 'DOPD', 'DUK',
               'EPE',
               'ERCO', 'EEI', 'FPL', 'FPC', 'GVL', 'HST', 'IPCO', 'IID', 'JEA', 'LDWP', 'LGEE', 'NWMT', 'NEVP',
               'ISNE',
               'NSB', 'NYIS', 'OVEC', 'PACW', 'PACE', 'GRMA', 'FMPP', 'GCPD', 'PJM', 'AVRN', 'PSCO', 'PGE', 'PNM',
               'PSEI',
               'BANC', 'SRP', 'SCL', 'SCEG', 'SC', 'SPA', 'SOCO', 'TPWR', 'TAL', 'TEC', 'TVA', 'TIDC', 'HECO',
               'WAUW',
               'AVA', 'SEC', 'TEPC', 'WALC', 'WAUE', 'WACM', 'SEPA', 'HECO', 'GRIF', 'GWA', 'GRIS', 'MISO',
               'DEAA',
               'CPLW', 'GRID', 'WWA', 'SWPP']

    for i in ba_name:
        # get the paths for th EIA, population and WRF data
        eia_path = glob.glob(eia_930_output_dir + "\\" + f"{i}*.csv")
        pop_path = glob.glob(pop_output_dir + "\\" + f"{i}*.csv")
        wrf_path = glob.glob(wrf_output_dir + "\\" + f"{i}*.csv")

        # read in the csv
        eia_df = pd.read_csv(eia_path)
        pop_df = pd.read_csv(pop_path)
        wrf_df = pd.read_csv(wrf_path)

        # merge the EIA 930, population and WRF data by date
        merged = pd.merge(eia_df, pop_df, wrf_df, how='left', on=['Year', 'Month', 'Day', 'Hour'])

        # write the merged dataframe to a csv
        merged.to_csv(os.path.join(compile_output_dir, f'{i}_hourly_compiled_data.csv'), index=False, header=True)

    return merged
