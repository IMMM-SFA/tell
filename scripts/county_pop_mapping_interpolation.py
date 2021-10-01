import glob
import pandas as pd
import numpy as np
from datetime import date


def map_eia_ids():
    """Create dataframe of EIA BA Number, BA short name and BA long name for later mapping

     :return:            Dataframe of EIA BA Number, BA short name and BA long name

     """
    EIA_BA_Number = [1, 189, 317, 599, 803, 924, 1738, 2775, 3046, 3413, 3522, 5326, 5416, 5701, 5723, 5748, 6452, 6455,
                     6909, 8795,
                     9191, 9216, 9617, 11208, 11249, 12825, 13407, 13434, 13485, 13501, 14015, 14378, 14379, 14412,
                     14610, 14624,
                     14725, 15399, 15466, 15248, 15473, 15500, 16534, 16572, 16868, 17539, 17543, 17716, 18195, 18429,
                     18445, 18454,
                     18642, 19281, 19547, 19610, 20169, 21554, 24211, 25471, 28502, 28503, 29304, 32790, 56090, 56365,
                     56545, 56669, 56812, 58786, 58790, 58791, 59504]
    BA_Short_Name = ['NBSO', 'AEC', 'YAD', 'AMPL', 'AZPS', 'AECI', 'BPAT', 'CISO', 'CPLE', 'CHPD', 'CEA', 'DOPD', 'DUK',
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
    BA_Long_Name = ['New Brunswick System Operator', 'PowerSouth Energy Cooperative',
                    'Alcoa Power Generating Inc. - Yadkin Division', 'Anchorage Municipal Light and Power',
                    'Arizona Public Service Company', 'Associated Electric Cooperative Inc.',
                    'Bonneville Power Administration', 'California Independent System Operator',
                    'Duke Energy Progress East', 'PUD No. 1 of Chelan County', 'Chugach Electric Association Inc.',
                    'PUD No. 1 of Douglas County', 'Duke Energy Carolinas', 'El Paso Electric Company',
                    'Electric Reliability Council of Texas Inc.', 'Electric Energy Inc.',
                    'Florida Power & Light Company',
                    'Duke Energy Florida Inc.', 'Gainesville Regional Utilities', 'City of Homestead',
                    'Idaho Power Company',
                    'Imperial Irrigation District', 'JEA', 'Los Angeles Department of Water and Power',
                    'Louisville Gas & Electric Company and Kentucky Utilities', 'NorthWestern Energy',
                    'Nevada Power Company', 'ISO New England Inc.', 'New Smyrna Beach Utilities Commission',
                    'New York Independent System Operator', 'Ohio Valley Electric Corporation',
                    'PacifiCorp - West', 'PacifiCorp - East', 'Gila River Power LLC', 'Florida Municipal Power Pool',
                    'PUD No. 2 of Grant County', 'PJM Interconnection LLC', 'Avangrid Renewables LLC',
                    'Public Service Company of Colorado', 'Portland General Electric Company',
                    'Public Service Company of New Mexico', 'Puget Sound Energy',
                    'Balancing Authority of Northern California', 'Salt River Project', 'Seattle City Light',
                    'South Carolina Electric & Gas Company', 'South Carolina Public Service Authority',
                    'Southwestern Power Administration', 'Southern Company Services Inc. - Transmission',
                    'City of Tacoma Department of Public Utilities Light Division', 'City of Tallahassee',
                    'Tampa Electric Company', 'Tennessee Valley Authority', 'Turlock Irrigation District',
                    'Hawaiian Electric Company Inc.', 'Western Area Power Administration - UGP West',
                    'Avista Corporation', 'Seminole Electric Cooperative', 'Tucson Electric Power Company',
                    'Western Area Power Administration - Desert Southwest Region',
                    'Western Area Power Administration - UGP East',
                    'Western Area Power Administration - Rocky Mountain Region', 'Southeastern Power Administration',
                    'New Harquahala Generating Company LLC', 'Griffith Energy LLC', 'NaturEner Power Watch LLC',
                    'Gridforce South', 'Midcontinent Independent Transmission System Operator Inc.',
                    'Arlington Valley LLC',
                    'Duke Energy Progress West', 'Gridforce Energy Management LLC', 'NaturEner Wind Watch LLC',
                    'Southwest Power Pool']

    df = pd.DataFrame(list(zip(EIA_BA_Number, BA_Short_Name, BA_Long_Name)),
                      columns=['EIA_BA_Number', 'BA_Short_Name', 'BA_Long_Name'])
    return df



def fips_pop_yearly(population_input_dir, start_year, end_year):
    """Make a list of all of the files xlsx in the data_input_dir

    :return:            List of input files to process

    """
    # get population from merged mapping data
    df_pop = pd.read_csv(population_input_dir + '/county_populations_2000_to_2019.csv')

    # loop over years to sum population by year
    df = pd.DataFrame([])
    for y in range(start_year, end_year + 1):
        # only keep columns that are needed
        key = [f'pop_{y}', 'county_FIPS']

        # change pop yr name for later merging
        df_pop_yr = df_pop[key].copy()

        df_pop_yr['year'] = y
        df_pop_yr.rename(columns={f'pop_{y}': 'population'}, inplace=True)

        # combine all years for one dataset
        df = df.append(df_pop_yr)

    return df

def merge_mapping_data(mapping_input_dir, population_input_dir, start_year, end_year):
    """Make a list of all of the files xlsx in the data_input_dir

    :return:            List of input files to process

    """
    # load FIPS county data for BA number and FIPs code matching for later population sum by BA
    all_files = glob.glob(mapping_input_dir + "/*.csv")

    list = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        list.append(df)

    frame = pd.concat(list, axis=0, ignore_index=True)
    col_names = ['year', 'county_fips', 'ba_number']

    # only keep columns that are needed
    frame = frame[col_names].copy()
    frame['ba_number'] = frame['ba_number'].fillna(0).astype(np.int64)
    frame['county_fips'] = frame['county_fips'].fillna(0).astype(np.int64)

    # select for valid BA numbers (from BA metadata)
    metadata = map_eia_ids()
    metadata.rename(columns={"EIA_BA_Number": "ba_number"}, inplace=True)

    # merge mapping df to the the metadata
    df_map = frame.merge(metadata, on=['ba_number'])
    df_map.rename(columns={"county_fips": "county_FIPS"}, inplace=True)

    # get sum of population by FIPS and merge to mapping file
    df_pop = fips_pop_yearly(population_input_dir, start_year, end_year)

    df = pd.merge(df_pop, df_map, how='left', left_on=['county_FIPS', 'year'], right_on=['county_FIPS', 'year'])

    return df


def ba_pop_sum(mapping_input_dir, population_input_dir, start_year, end_year):
    """Make a list of all of the files xlsx in the data_input_dir

    :return:            List of input files to process

    """
    # get population from merged mapping data
    df_pop = merge_mapping_data(mapping_input_dir, population_input_dir, start_year, end_year)

    # loop over years to sum population by year
    df = pd.DataFrame([])
    for y in range(start_year, end_year + 1):

        # sum population by BA
        pop_sum_yr = df_pop.groupby(['BA_Short_Name','year'])['population'].sum().reset_index()

        # combine all years for one dataset
        df = df.append(pop_sum_yr)

    return df



def ba_pop_sum(start_year, end_year):

    ba_names = df_pop['BA_Short_Name'].unique()

    df = pd.DataFrame([])
    for i in ba_names):

        t = df_pop['year']
        y1 = df_pop['population']
        x = pd.DataFrame({'Hours': pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31',
                                                 freq='1H', closed='left')})

        BA_interp = interp1(t, y1, x, 'linear');

    df = BA_interp.append(pop_sum_yr)

    return df

    return df



