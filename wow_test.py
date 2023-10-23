import os
import tell
import joblib
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 16

class WOWExp:

    def __init__(self):
        self.current_dir = os.path.join(os.path.dirname(os.getcwd()))
        self.tell_data_dir = os.path.join(self.current_dir, r'tell_data')
        self.tell_image_dir = os.path.join(self.tell_data_dir, r'visualizations')

        if not os.path.exists(self.tell_data_dir):
            os.makedirs(self.tell_data_dir)

        # If the "tell_image_dir" subdirectory doesn't exist then create it:
        if not os.path.exists(self.tell_image_dir):
            os.makedirs(self.tell_image_dir)

    def install(self):
        tell.install_quickstarter_data(data_dir=self.tell_data_dir)
        tell.install_sample_forcing_data(data_dir=tell_data_dir)
        return None

    def process_eia_930(self):
        tell.process_eia_930_data(data_input_dir=self.tell_data_dir,
                                  n_jobs=-1)
        return None

    def train(self):

        return None





class WOWData:

    def __init__(
            self,
            input_eia_dir,
            input_load_csv,
            target_dir=None
    ):
        """
        Class to take in the historic data from EIA and
        """

        self.input_eia_dir = input_eia_dir
        self.input_load_csv = input_load_csv
        self.target_dir = target_dir

        if self.target_dir is not None and not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

        self.df_loads = self._read_wow_data()
        self.eia_ba_list = self._get_list_of_eia_ba()

    def _read_wow_data(self):
        """
        Method to read a single WOW CSV, and parse by year
        :return df_list: (list) - list of df parsed by years
        """
        _df_all = pd.read_csv(self.input_load_csv)
        years = _df_all["climate_year"].unique()
        df_loads = []

        for yr in years:
            _df = _df_all.loc[_df_all["climate_year"] == yr]
            df_loads.append(_df)
        return df_loads


    def _get_list_of_eia_ba(self):
        """
        Method to get list of BAs included in EIA Weather data
        """
        _file_list = glob.glob(os.path.join(self.input_eia_dir, "*.csv"))
        ba_list = []
        for file in _file_list:
            _ba = file.split("_")[0]
            _ba =_ba.split("/")[-1]
            if _ba not in ba_list:
                ba_list.append(_ba)
        return ba_list


class MLTrainer:

    def __init__(
            self,
            target_dir,
            region="all",
            csv_dir="/people/rahm312/wow/WECC_2050_climate/results_dir",
            save_bool=True
    ):
        """
        Class to train MLP models
        """
        self.target_dir = target_dir
        self.region = region
        self.csv_dir = csv_dir
        self.save_bool = save_bool
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)

    def train(self):
        print(f"Training for ba: {self.region}")
        self.df_pred, self.df_perf = tell.train(region=self.region, data_dir=self.target_dir)
        print(self.df_perf)
        if self.save_bool:
            self.df_pred.to_csv(os.path.join(self.csv_dir, f"{self.region}_pred.csv"))
            self.df_perf.to_csv(os.path.join(self.csv_dir, f"{self.region}_perf.csv"))
        return None



def merge_data(ba_name, df_list, weather_dir, start_year, end_year, target_dir):
    """
    function to parallelize: locates the relevant file within eia_dir and merges the dataframe
    :param ba_name: (str) - name of ba.
    :param df_list: (list) - list of dfs
    :weather_dir: (str) - full path to directory contining the weather data
    :start_year: (int) - start year for which we're compiling the data
    :end_year: (int) - end year for which we're compiling the data.
    """
    _df_list = []
    #Get the filename in weather_dir corresponding to the ba_name + weather_dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    years = np.arange(start_year, end_year + 1)
    for n, year in enumerate(years):
        _file_list = glob.glob(os.path.join(weather_dir, f"{ba_name}_*_{year}.csv"))
        weather_file = _file_list[0]
        print(f"weather_file: {weather_file}")
        #Read the weather file
        _df_weather = pd.read_csv(weather_file)
        #Identify the corresponding loads file
        #_loads_list = glob.glob(os.path.join(load_dir, f"*_{year}_climate.csv"))
        #demand_file = _loads_list[0]
        df_load = df_list[n]
        #print(f"demand_file: {demand_file}")
        #Get the df_load
        load = df_load[ba_name].values
        # Add to the file
        #curtail weather file if its a leap year
        if _df_weather.shape[0] != load.shape[0]:
            _df_weather = _df_weather.iloc[:load.shape[0]]
        #month, day, year, hour = _df_weather.apply(lambda row: get_time_info(row["Time_UTC"]), axis=1)
        # print(f"month: {month}")
        # print(f"day: {day}")
        # print(f"year: {year}")
        # print(f"hour: {hour}")
        _df_weather['Datetime'] = pd.to_datetime(_df_weather.Time_UTC, format='%Y-%m-%d %H:%M:%S')
        _df_weather["Demand"] =  load
        _df_weather['Year'] = _df_weather['Datetime'].dt.strftime('%Y')
        _df_weather['Month'] = _df_weather['Datetime'].dt.strftime('%m')
        _df_weather['Day'] = _df_weather['Datetime'].dt.strftime('%d')
        _df_weather['Hour'] = _df_weather['Datetime'].dt.strftime('%H')
        #_df_weather.to_csv(os.path.join(target_dir, f"{ba_name}_compiled_data_{year}.csv"))
        _df_list.append(_df_weather)

    df = pd.concat(_df_list)
    df.to_csv(os.path.join(target_dir, f"{ba_name}_historical_data.csv"))
    print(f"df shape: {df.shape}")
    #Format the dataframes
    return None


def get_time_info(date_str):
    """
    Function to parse datestring information
    """
    _date, _time = date_str.split()[0], date_str.split()[1]
    _date_tmp = _date.split("/")
    month = int(_date_tmp[0])
    day = int(_date_tmp[1])
    year = int(_date_tmp[2])

    _time_tmp = _time.split(":")
    hour = int(_time_tmp[0])
    return month, day, year, hour

def plot_profile(filename, ba_code, start_idx, end_idx):
    """
    Plot profile within a fixed time interval
    """
    df = pd.read_csv(filename)
    load = df[ba_code].values
    segmented_load = load[start_idx:end_idx]
    time = np.arange(len(segmented_load))

    plt.plot(time, segmented_load, 'k-', linewidth=3)
    plt.xlabel("Time (hours)")
    plt.ylabel("Load (MWh)")
    plt.savefig("/people/rahm312/wow/tell_data/visualizations/test.svg")
    return None


def bar_plot():
    csv_dir = "/people/rahm312/wow/WECC_2050_climate/results_dir"
    _file_list = glob.glob(os.path.join(csv_dir, "*_perf.csv"))
    r2 = np.zeros(len(_file_list), )
    mape = np.zeros_like(r2)

    ba_list = []
    for i, csv in enumerate(_file_list):
        ba = (csv.split("/")[-1]).split("_")[0]
        ba_list.append(ba)
        #print(f"ba: {ba}")
        _df = pd.read_csv(csv)

        mape[i] = _df["MAPE"].values[0]
        r2[i] = _df["R2"].values[0]

    mape = 100*pd.Series(mape)
    #fig, ax = plt.subplots(figsize=(12, 8))
    plt.figure(figsize=(12, 8))
    plt.yticks(fontsize=16)
    ax = mape.plot(kind='bar')
    #ax.bar(np.arange(len(mape)), mape*100)
    ax.set_xlabel("BA", fontsize=16)
    ax.set_ylabel("MAPE (%)", fontsize=16)
    ax.set_xticklabels(ba_list, fontsize=16)
    plt.savefig(f"/people/rahm312/wow/tell_data/visualizations/mape.svg")
    plt.close()

    r2 = pd.Series(r2)
    plt.figure(figsize=(12, 8))
    plt.yticks(fontsize=16)
    ax = r2.plot(kind='bar')
    # ax.bar(np.arange(len(mape)), mape*100)
    ax.set_xlabel("BA", fontsize=16)
    ax.set_ylabel("R2", fontsize=16)
    ax.set_xticklabels(ba_list, fontsize=16)
    plt.savefig(f"/people/rahm312/wow/tell_data/visualizations/r2.svg")
    plt.close()
    return None


def plot_diurnal(ba_name, start_day=0, end_day=365):
    #first read from file the ground truth and predictions
    csv_dir = "/people/rahm312/wow/WECC_2050_climate/results_dir"
    csv_file = os.path.join(csv_dir, f"{ba_name}_pred.csv")
    _start_idx = start_day*24
    _end_idx = end_day*24

    _df = pd.read_csv(csv_file)
    _predictions = _df["predictions"].values #[_start_idx:_end_idx]
    _gt = _df["ground_truth"].values #[_start_idx:_end_idx]
    _time = np.arange(len(_predictions))
    print(_predictions.shape, _gt.shape, _time.shape)
    plt.figure(figsize=(20, 8))
    plt.plot(_time[_start_idx:_end_idx], _gt[_start_idx:_end_idx], 'k-', linewidth=3)
    plt.plot(_time[_start_idx:_end_idx], _predictions[_start_idx:_end_idx], 'r-', linewidth=3)
    # ax.bar(np.arange(len(mape)), mape*100)
    plt.xlabel("Time (hours)", fontsize=16)
    plt.ylabel("Demand (MWh)", fontsize=16)
    plt.savefig(f"/people/rahm312/wow/tell_data/visualizations/{ba_name}_{start_day}_{end_day}.svg")
    plt.close()
    return None

if __name__=="__main__":
    # Exp1 = WOWExp()
    # Exp1.process_eia_930()
    weather_dir = "/people/rahm312/wow/historic/"
    wecc_dir = "/people/rahm312/wow/WEEC_2050_climate/"
    wecc_csv = "/people/rahm312/wow/WEEC_2050_climate/WOW_all_data.csv"
    target_dir = "/people/rahm312/wow/WECC_2050_climate/compiled_data/"

    # Exp1 = WOWData(
    #     input_eia_dir=weather_dir,
    #     input_load_csv=wecc_csv
    # )

    #merge_data("AVA", Exp1.df_loads, weather_dir, 2007, 2013, target_dir)
    #ba_list = Exp1.df_loads[0].columns[4:].tolist()

    bar_plot()
    plot_diurnal(ba_name="BANC", start_day=0, end_day=365)
    plot_diurnal(ba_name="BANC", start_day=17, end_day=24)
    plot_diurnal(ba_name="BANC", start_day=235, end_day=242)

    plot_diurnal(ba_name="BPAT", start_day=0, end_day=365)
    plot_diurnal(ba_name="BPAT", start_day=17, end_day=24)
    plot_diurnal(ba_name="BPAT", start_day=235, end_day=242)
    # for ba in ba_list:
    #       print(f"ba: {ba}")
    #       if ba not in ["CIPB", "CIPV", "CISC", "CISD", "IPFE", "IPMV", "IPTV", "PAID", "PAUT", "PAWY", "SPPC", "VEA"]:
    #           #merge_data(ba, Exp1.df_loads, weather_dir, 2007, 2013, target_dir)
    #           Trainer = MLTrainer(target_dir=target_dir, region=ba)
    #           Trainer.train()

    #month, day, year, hour = get_time_info("1/2/1980  6:00:00 PM")
    #print(f"month: {month}, day: {day}, year: {year}, hour: {hour}")
    #check the profile
    # filename = "/people/rahm312/wow/WEEC_2050_climate/WECC_2050_Year_2008_climate.csv"
    # start_idx = (31 + 24)*24
    # end_idx = (31 + 24 + 7)*24
    # plot_profile(filename, "PSCO", start_idx, end_idx)