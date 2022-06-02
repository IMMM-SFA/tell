import pandas as pd
from pandas import DataFrame


def metadata_eia(numbers: int) -> DataFrame:
    """Define the state FIPS code and state name from a given state abbreviation.

    :param numbers:                 EIA 930 BA number
    :type numbers:                  int

    :return:                        DataFrame with BA short and long name

     """

    results = []

    for eia_ba_number in numbers:
        if eia_ba_number == 1:      ba_short_name = 'NBSO'; ba_long_name = 'New Brunswick System Operator'
        elif eia_ba_number == 189:    ba_short_name = 'AEC';  ba_long_name = 'PowerSouth Energy Cooperative'
        elif eia_ba_number == 317:    ba_short_name = 'YAD';  ba_long_name = 'Alcoa Power Generating Inc. - Yadkin ' \
                                                                             'Division'
        elif eia_ba_number == 599:    ba_short_name = 'AMPL'; ba_long_name = 'Anchorage Municipal Light and Power'
        elif eia_ba_number == 803:    ba_short_name = 'AZPS'; ba_long_name = 'Arizona Public Service Company'
        elif eia_ba_number == 924:    ba_short_name = 'AECI'; ba_long_name = 'Associated Electric Cooperative Inc.'
        elif eia_ba_number == 1738:   ba_short_name = 'BPAT'; ba_long_name = 'Bonneville Power Administration'
        elif eia_ba_number == 2775:   ba_short_name = 'CISO'; ba_long_name = 'California Independent System Operator'
        elif eia_ba_number == 3046:   ba_short_name = 'CPLE'; ba_long_name = 'Duke Energy Progress East'
        elif eia_ba_number == 3413:   ba_short_name = 'CHPD'; ba_long_name = 'PUD No. 1 of Chelan County'
        elif eia_ba_number == 3522:   ba_short_name = 'CEA';  ba_long_name = 'Chugach Electric Association Inc.'
        elif eia_ba_number == 5326:   ba_short_name = 'DOPD'; ba_long_name = 'PUD No. 1 of Douglas County'
        elif eia_ba_number == 5416:   ba_short_name = 'DUK';  ba_long_name = 'Duke Energy Carolinas'
        elif eia_ba_number == 5701:   ba_short_name = 'EPE';  ba_long_name = 'El Paso Electric Company'
        elif eia_ba_number == 5723:   ba_short_name = 'ERCO'; ba_long_name = 'Electric Reliability Council of Texas Inc.'
        elif eia_ba_number == 5748:   ba_short_name = 'EEI';  ba_long_name = 'Electric Energy Inc.'
        elif eia_ba_number == 6452:   ba_short_name = 'FPL';  ba_long_name = 'Florida Power & Light Company'
        elif eia_ba_number == 6455:   ba_short_name = 'FPC';  ba_long_name = 'Duke Energy Florida Inc.'
        elif eia_ba_number == 6909:   ba_short_name = 'GVL';  ba_long_name = 'Gainesville Regional Utilities'
        elif eia_ba_number == 8795:   ba_short_name = 'HST';  ba_long_name = 'City of Homestead'
        elif eia_ba_number == 9191:   ba_short_name = 'IPCO'; ba_long_name = 'Idaho Power Company'
        elif eia_ba_number == 9216:   ba_short_name = 'IID';  ba_long_name = 'Imperial Irrigation District'
        elif eia_ba_number == 9617:   ba_short_name = 'JEA';  ba_long_name = 'JEA'
        elif eia_ba_number == 11208:  ba_short_name = 'LDWP'; ba_long_name = 'Los Angeles Department of Water and Power'
        elif eia_ba_number == 11249:  ba_short_name = 'LGEE'; ba_long_name = 'Louisville Gas & Electric Company and ' \
                                                                             'Kentucky Utilities'
        elif eia_ba_number == 12825:  ba_short_name = 'NWMT'; ba_long_name = 'NorthWestern Energy'
        elif eia_ba_number == 13407:  ba_short_name = 'NEVP'; ba_long_name = 'Nevada Power Company'
        elif eia_ba_number == 13434:  ba_short_name = 'ISNE'; ba_long_name = 'ISO New England Inc.'
        elif eia_ba_number == 13485:  ba_short_name = 'NSB';  ba_long_name = 'New Smyrna Beach Utilities Commission'
        elif eia_ba_number == 13501:  ba_short_name = 'NYIS'; ba_long_name = 'New York Independent System Operator'
        elif eia_ba_number == 14015:  ba_short_name = 'OVEC'; ba_long_name = 'Ohio Valley Electric Corporation'
        elif eia_ba_number == 14378:  ba_short_name = 'PACW'; ba_long_name = 'PacifiCorp - West'
        elif eia_ba_number == 14379:  ba_short_name = 'PACE'; ba_long_name = 'PacifiCorp - East'
        elif eia_ba_number == 14412:  ba_short_name = 'GRMA'; ba_long_name = 'Gila River Power LLC'
        elif eia_ba_number == 14610:  ba_short_name = 'FMPP'; ba_long_name = 'Florida Municipal Power Pool'
        elif eia_ba_number == 14624:  ba_short_name = 'GCPD'; ba_long_name = 'PUD No. 2 of Grant County'
        elif eia_ba_number == 14725:  ba_short_name = 'PJM';  ba_long_name = 'PJM Interconnection LLC'
        elif eia_ba_number == 15399:  ba_short_name = 'AVRN'; ba_long_name = 'Avangrid Renewables LLC'
        elif eia_ba_number == 15466:  ba_short_name = 'PSCO'; ba_long_name = 'Public Service Company of Colorado'
        elif eia_ba_number == 15248:  ba_short_name = 'PGE';  ba_long_name = 'Portland General Electric Company'
        elif eia_ba_number == 15473:  ba_short_name = 'PNM';  ba_long_name = 'Public Service Company of New Mexico'
        elif eia_ba_number == 15500:  ba_short_name = 'PSEI'; ba_long_name = 'Puget Sound Energy'
        elif eia_ba_number == 16534:  ba_short_name = 'BANC'; ba_long_name = 'Balancing Authority of Northern ' \
                                                                             'California'
        elif eia_ba_number == 16572:  ba_short_name = 'SRP';  ba_long_name = 'Salt River Project'
        elif eia_ba_number == 16868:  ba_short_name = 'SCL';  ba_long_name = 'Seattle City Light'
        elif eia_ba_number == 17539:  ba_short_name = 'SCEG'; ba_long_name = 'South Carolina Electric & Gas Company'
        elif eia_ba_number == 17543:  ba_short_name = 'SC';   ba_long_name = 'South Carolina Public Service Authority'
        elif eia_ba_number == 17716:  ba_short_name = 'SPA';  ba_long_name = 'Southwestern Power Administration'
        elif eia_ba_number == 18195:  ba_short_name = 'SOCO'; ba_long_name = 'Southern Company Services Inc. - ' \
                                                                             'Transmission'
        elif eia_ba_number == 18429:  ba_short_name = 'TPWR'; ba_long_name = 'City of Tacoma Department of Public' \
                                                                             ' Utilities Light Division'
        elif eia_ba_number == 18445:  ba_short_name = 'TAL';  ba_long_name = 'City of Tallahassee'
        elif eia_ba_number == 18454:  ba_short_name = 'TEC';  ba_long_name = 'Tampa Electric Company'
        elif eia_ba_number == 18642:  ba_short_name = 'TVA';  ba_long_name = 'Tennessee Valley Authority'
        elif eia_ba_number == 19281:  ba_short_name = 'TIDC'; ba_long_name = 'Turlock Irrigation District'
        elif eia_ba_number == 19547:  ba_short_name = 'HECO'; ba_long_name = 'Hawaiian Electric Company Inc.'
        elif eia_ba_number == 19610:  ba_short_name = 'WAUW'; ba_long_name = 'Western Area Power Administration - ' \
                                                                             'UGP West'
        elif eia_ba_number == 20169:  ba_short_name = 'AVA';  ba_long_name = 'Avista Corporation'
        elif eia_ba_number == 21554:  ba_short_name = 'SEC';  ba_long_name = 'Seminole Electric Cooperative'
        elif eia_ba_number == 24211:  ba_short_name = 'TEPC'; ba_long_name = 'Tucson Electric Power Company'
        elif eia_ba_number == 25471:  ba_short_name = 'WALC'; ba_long_name = 'Western Area Power Administration - ' \
                                                                             'Desert Southwest Region'
        elif eia_ba_number == 28502:  ba_short_name = 'WAUE'; ba_long_name = 'Western Area Power Administration - ' \
                                                                             'UGP East'
        elif eia_ba_number == 28503:  ba_short_name = 'WACM'; ba_long_name = 'Western Area Power Administration - ' \
                                                                             'Rocky Mountain Region'
        elif eia_ba_number == 29304:  ba_short_name = 'SEPA'; ba_long_name = 'Southeastern Power Administration'
        elif eia_ba_number == 32790:  ba_short_name = 'HECO'; ba_long_name = 'New Harquahala Generating Company LLC'
        elif eia_ba_number == 56090:  ba_short_name = 'GRIF'; ba_long_name = 'Griffith Energy LLC'
        elif eia_ba_number == 56365:  ba_short_name = 'GWA';  ba_long_name = 'NaturEner Power Watch LLC'
        elif eia_ba_number == 56545:  ba_short_name = 'GRIS'; ba_long_name = 'Gridforce South'
        elif eia_ba_number == 56669:  ba_short_name = 'MISO'; ba_long_name = 'Midcontinent Independent Transmission ' \
                                                                             'System Operator Inc.'
        elif eia_ba_number == 56812:  ba_short_name = 'DEAA'; ba_long_name = 'Arlington Valley LLC'
        elif eia_ba_number == 58786:  ba_short_name = 'CPLW'; ba_long_name = 'Duke Energy Progress West'
        elif eia_ba_number == 58790:  ba_short_name = 'GRID'; ba_long_name = 'Gridforce Energy Management LLC'
        elif eia_ba_number == 58791:  ba_short_name = 'WWA';  ba_long_name = 'NaturEner Wind Watch LLC'
        elif eia_ba_number == 59504:  ba_short_name = 'SWPP'; ba_long_name = 'Southwest Power Pool'
        else: ba_short_name = None; ba_long_name = None

        results.append([eia_ba_number, ba_short_name, ba_long_name])
        df = pd.DataFrame(results, columns=['BA_Number', 'BA_Name', 'BA_Long_Name'])

    return df
