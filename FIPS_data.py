import pandas as pd

#load the FIPS files
FIPSfileLocation = "C:\\Users\\mcgr323\\projects\\TELL\\inputs\\"
FIPSfileName = 'state_and_county_fips_codes.xlsx'
FIPS_code = pd.read_excel(FIPSfileLocation + FIPSfileName)

#load the service territories
fileLocation = "C:\\Users\\mcgr323\\projects\\TELL\\inputs\\EIA_861\\Raw_Data\\2019\\"
fileName = 'Service_Territory_2019.xlsx'
Counties_States = pd.read_excel(fileLocation + fileName, sheet_name='Counties_States')
Counties_Territories = pd.read_excel(fileLocation + fileName, sheet_name='Counties_Territories')

# dropping null value columns to avoid errors
FIPS_code.dropna(inplace=True)
FIPS_code['county_name'] = FIPS_code['county_name'].map(lambda x: x.lstrip('+-').rstrip('County'))
FIPS_code['County'] = FIPS_code['county_name']

result_2 = pd.merge(Counties_States, FIPS_code, on='County')

import numpy as np
Counties_States['County_FIPS'] = FIPS_code['county_name'] #add the Price2 column from df2 to df1
merge(Counties_States, FIPS_code, by.x = 2, by.y = 0, all.x = TRUE)
Counties_States['countiesMatch?'] = np.where(Counties_States['County'] == FIPS_code['County'], 'True', 'False')  #create new column in df1 to check if prices match
#print (Counties_States)