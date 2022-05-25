def state_metadata_from_state_abbreviation(state_abbreviation: str) -> tuple[int, str]:
    """
    Define the state FIPS code and state name from a given state abbreviation.

    :param state_abbreviation:          state abbreviation
    :type state_abbreviation:           str

    :return:                            [0] state FIPS code
                                        [1] state name

    """

    state_fips = None
    state_name = None

    if state_abbreviation == 'AK': state_fips = 1000;  state_name = 'Alaska'
    if state_abbreviation == 'AL': state_fips = 1000;  state_name = 'Alabama'
    if state_abbreviation == 'AK': state_fips = 2000;  state_name = 'Alaska'
    if state_abbreviation == 'AZ': state_fips = 4000;  state_name = 'Arizona'
    if state_abbreviation == 'AR': state_fips = 5000;  state_name = 'Arkansas'
    if state_abbreviation == 'CA': state_fips = 6000;  state_name = 'California'
    if state_abbreviation == 'CO': state_fips = 8000;  state_name = 'Colorado'
    if state_abbreviation == 'CT': state_fips = 9000;  state_name = 'Connecticut'
    if state_abbreviation == 'DE': state_fips = 10000; state_name = 'Delaware'
    if state_abbreviation == 'DC': state_fips = 11000; state_name = 'District of Columbia'
    if state_abbreviation == 'FL': state_fips = 12000; state_name = 'Florida'
    if state_abbreviation == 'GA': state_fips = 13000; state_name = 'Georgia'
    if state_abbreviation == 'HI': state_fips = 15000; state_name = 'Hawaii'
    if state_abbreviation == 'ID': state_fips = 16000; state_name = 'Idaho'
    if state_abbreviation == 'IL': state_fips = 17000; state_name = 'Illinois'
    if state_abbreviation == 'IN': state_fips = 18000; state_name = 'Indiana'
    if state_abbreviation == 'IA': state_fips = 19000; state_name = 'Iowa'
    if state_abbreviation == 'KS': state_fips = 20000; state_name = 'Kansas'
    if state_abbreviation == 'KY': state_fips = 21000; state_name = 'Kentucky'
    if state_abbreviation == 'LA': state_fips = 22000; state_name = 'Louisiana'
    if state_abbreviation == 'ME': state_fips = 23000; state_name = 'Maine'
    if state_abbreviation == 'MD': state_fips = 24000; state_name = 'Maryland'
    if state_abbreviation == 'MA': state_fips = 25000; state_name = 'Massachusetts'
    if state_abbreviation == 'MI': state_fips = 26000; state_name = 'Michigan'
    if state_abbreviation == 'MN': state_fips = 27000; state_name = 'Minnesota'
    if state_abbreviation == 'MS': state_fips = 28000; state_name = 'Mississippi'
    if state_abbreviation == 'MO': state_fips = 29000; state_name = 'Missouri'
    if state_abbreviation == 'MT': state_fips = 30000; state_name = 'Montana'
    if state_abbreviation == 'NE': state_fips = 31000; state_name = 'Nebraska'
    if state_abbreviation == 'NV': state_fips = 32000; state_name = 'Nevada'
    if state_abbreviation == 'NH': state_fips = 33000; state_name = 'New Hampshire'
    if state_abbreviation == 'NJ': state_fips = 34000; state_name = 'New Jersey'
    if state_abbreviation == 'NM': state_fips = 35000; state_name = 'New Mexico'
    if state_abbreviation == 'NY': state_fips = 36000; state_name = 'New York'
    if state_abbreviation == 'NC': state_fips = 37000; state_name = 'North Carolina'
    if state_abbreviation == 'ND': state_fips = 38000; state_name = 'North Dakota'
    if state_abbreviation == 'OH': state_fips = 39000; state_name = 'Ohio'
    if state_abbreviation == 'OK': state_fips = 40000; state_name = 'Oklahoma'
    if state_abbreviation == 'OR': state_fips = 41000; state_name = 'Oregon'
    if state_abbreviation == 'PA': state_fips = 42000; state_name = 'Pennsylvania'
    if state_abbreviation == 'RI': state_fips = 44000; state_name = 'Rhode Island'
    if state_abbreviation == 'SC': state_fips = 45000; state_name = 'South Carolina'
    if state_abbreviation == 'SD': state_fips = 46000; state_name = 'South Dakota'
    if state_abbreviation == 'TN': state_fips = 47000; state_name = 'Tennessee'
    if state_abbreviation == 'TX': state_fips = 48000; state_name = 'Texas'
    if state_abbreviation == 'UT': state_fips = 49000; state_name = 'Utah'
    if state_abbreviation == 'VT': state_fips = 50000; state_name = 'Vermont'
    if state_abbreviation == 'VA': state_fips = 51000; state_name = 'Virginia'
    if state_abbreviation == 'WA': state_fips = 53000; state_name = 'Washington'
    if state_abbreviation == 'WV': state_fips = 54000; state_name = 'West Virginia'
    if state_abbreviation == 'WI': state_fips = 55000; state_name = 'Wisconsin'
    if state_abbreviation == 'WY': state_fips = 56000; state_name = 'Wyoming'

    if state_fips is None:
        raise KeyError(f"There are no FIPS codes available for state abbreviation:  '{state_abbreviation}'.")

    return state_fips, state_name
