import glob
import pandas as pd
import numpy as np


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


def merge_mapping_data(mapping_input_dir):
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
    filename = 'C:/Users/mcgr323/projects/tell/EIA_BA_match.csv'
    metadata = pd.read_csv(filename, index_col=None, header=0)
    # rename columns
    metadata.rename(columns={"EIA_BA_Number": "ba_number"}, inplace=True)
    df = frame.merge(metadata, on=['ba_number'])
    df.rename(columns={"county_fips": "county_FIPS"}, inplace=True)

    return df
