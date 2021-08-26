import os

import tell


if __name__ == '__main__':

    # set the target year
    target_year = 2015

    # get the input directory as it currently exists within this repository
    input_dir = os.path.join(os.path.dirname(__file__), 'inputs')

    # get the path to the EIA raw data for the target year
    eia_data_dir = os.path.join(input_dir, 'EIA_861', 'Raw_Data', str(target_year))

    # directory containing the outputs
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')

    # paths to files
    fips_file = os.path.join(input_dir, 'state_and_county_fips_codes.xlsx')
    service_area_file = os.path.join(eia_data_dir, f'Service_Territory_{target_year}.xlsx')
    sales_ult_file = os.path.join(eia_data_dir, f'Sales_Ult_Cust_{target_year}.xlsx')
    bal_auth_file = os.path.join(eia_data_dir, f'Balancing_Authority_{target_year}.xlsx')

    # prepare data
    tell.process_data(target_year, fips_file, service_area_file, sales_ult_file, bal_auth_file, output_dir)


    output = os.path.join(output_dir, f'fips_service_match_{target_year}.csv')
    output = pd.read_csv(output)
    output2 = output.drop_duplicates()
    output_file = os.path.join(output_dir, f'fips_service_match_rm{target_year}.csv')
    output2.to_csv(output_file, sep=',', index=False)
