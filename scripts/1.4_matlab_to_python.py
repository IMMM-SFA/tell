import os
import glob
import pandas as pd


# load population data
df_population = pd.read_csv(population_data_input_dir +'/county_populations_2000_to_2019.csv')

# load FIPS county data for BA number and FIPs code matching for later population sum by BA
all_files = glob.glob(service_territory_data_input_dir + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
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

# add BA number to population df
df_pop = pd.merge(left=df_population, right=df, on='county_FIPS')

#subset population data by year
#for x in range(start_year, end_year+1):
df_year = df_pop['year'] == 2015
df_pop_2015 = df_pop[df_year]

# only keep columns that are needed
key = ['year', 'pop_2015', 'ba_number']
df_pop_2015 = df_pop_2015[key].copy()

# sum population by BA
pop_sum_2015 = df_pop_2015.groupby(['year','ba_number'])['pop_2015'].sum().reset_index()

# interpolate population to hourly series
new_x = [0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]

hourly_population = interp1d(x,y)(new_x)

# 'linear' has no effect since it's the default, but I'll plot it too:
from datetime import date
from datetime import dateime

#counteract start date by adding one year and a day
start_string = date.toordinal(date(start_year,1,1))+366
end_string = date.toordinal(date(end_year,12,31))+366

python_start_datetime = datetime.fromordinal(int(start_string-366))
python_end_datetime = datetime.fromordinal(int(end_string-366))

x = 1:inv(24):365;
T_interp = interp1(t,y1,x,'spline');