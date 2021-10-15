
class Dataset:
    """TODO:  fill in a description of this class


    :param region: str indicating region/BA we want to train and test on. must match with str in csv
    :param csv_dir: str dir where the csvs are located
    :param start_time: str start time in YYYY-MM-DD. all data before this time as not considered
    :param end_time: str end_time in YYYY-MM-DD. all data after this time is not considered
    :param split_time: partition time in YYYY-MM-DD to split into training and test_set
    :param x_var: list-> features to consider
    :param y_var: list-> targets to consider
    :param add_dayofweek: bool -> weather we want to consider weekday vs weekend informoation
    :param linear_mode_bool: bool-> whether it is a linear model or not. if so, the month and hour will need to be sinusoidal


    """

    # define month_list for plots
    MONTH_LIST = [
        "Jan",
        "Feb",
        "Mar",
        "April",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    def __init__(self,
                 region,
                 csv_dir="../data/csv/",
                 start_time="2016-01-01 00:00:00",
                 end_time="2019-12-31 23:00:00",
                 split_time="2018-12-31 23:00:00",
                 x_var=[
                    "Hour",
                    "Month",
                    "Temperature",
                    "Specific_Humidity",
                    "Wind_Speed",
                    "Longwave_Radiation",
                    "Shortwave_Radiation"
                 ],
                 # 'Longwave_Radiation', 'Wind_Speed'
                 y_var=["Demand"],
                 add_dayofweek=True,
                 linear_mode_bool=False,
                 random_seed=None
                 ):

        # set random seed, change this value as you prefer
        if random_seed is not None:
            np.random.seed(random_seed)

        self.region = region
        self.csv_dir = csv_dir
        self.start_time, self.end_time = start_time, end_time
        self.split_time = split_time
        self.x_var, self.y_var = x_var, y_var
        self.lin_model_bool = linear_mode_bool

        self.add_dayofweek = add_dayofweek

        # Get x_var and y_var
        self.df, self.df_t, self.df_e, day_list = self.read_data()

        # get training and test data
        (
            self.df_t,
            self.df_e,
            self.df_eval,
            self.X_t,
            self.X_e,
            self.X_eval,
            self.Y_t,
            self.Y_e,
            self.Y_eval
        ) = self.preprocess_data(day_list=day_list)

    def read_data(self):

        """
        Function to read csvs
        :return df (pd.DataFrame) - entire data contained within start_time and end_time
        :return df_t (pd.DataFrame) - training data (before split_time)
        :return df_e (pd.DataFrame) - evaluation data (after split_time)
        """

        # step 1-> check filename
        filename = self.get_filename()

        # step 2-> read csv
        df = pd.read_csv(filename)

        # step 3-> sort df by timeframe specified in start_time and end_time
        df = self.sort_timeframe(df=df)

        # step 4 -> add time variables (sin(hr) and weekday/weekend)
        df, day_list = self.preprocess_timedata(df=df)

        # step 4 -> split into training and test set using split_time
        df_t, df_e = self.partition_data(df=df)

        # dropping negative rows after the data has been partitioned
        df_t = self.drop_neg_rows(df=df_t)

        # dropping negative and NaN rows for
        df_eval = self.drop_neg_rows(df=df_e, drop_nan=False)

        return df, df_t, df_e, df_eval, day_list

    def preprocess_data(self, day_list=None):
        """Takes the features and targets from df

        :return: df_t -> training dataframe df_e -> eval dataframe

        """

        # sort the entire df by the column headers we require
        if self.add_dayofweek == True:
            self.x_var = self.x_var + ["Weekday", "Holidays"]

        # extract the training and test data. only including datetime, x_var (features) and y_var (targets)
        df_t, df_e, df_eval = (
            self.df_t[["Datetime"] + self.x_var + self.y_var],
            self.df_e[["Datetime"] + self.x_var + self.y_var],
            self.df_eval[["Datetime"] + self.x_var + self.y_var],
        )
        X_t, X_e, X_eval = df_t[self.x_var], df_e[self.x_var], df_eval[self.x_var]
        Y_t, Y_e, Y_eval = df_t[self.y_var], df_e[self.y_var], df_eval[self.y_var]

        return df_t, df_e, df_eval, X_t, X_e, X_eval, Y_t, Y_e, Y_eval

    def get_filename(self):

        """
        function to extract filename for that
        :return filename: str filename corresponding to that region
        """

        str_to_check = os.path.join(self.csv_dir, f"{self.region}_*.csv")  # pattern to search for
        filename = glob.glob(str_to_check)[
            0
        ]  # [0] list all filenames with the string str_to_check

        return filename

    def sort_timeframe(self, df):

        """
        function to filter dataframe by specified timeframe self.start_time and self.end Time
        :return df_out: sorted df
        """
        df["Datetime"] = pd.to_datetime(
            df[["Day", "Month", "Year", "Hour"]]
        )  # Hour added

        df = df[
            (df["Datetime"] >= self.start_time) & (df["Datetime"] <= self.end_time)
        ]  # sort df by start time endtime

        df = df.reset_index(drop=True)  # rese

        return df

    def drop_neg_rows(self, df, drop_nan=True):

        """
        This method drops -9999 values. it also drops "extreme" values, i.e. demand that lies outside +/- 5*sigma
        :param df:
        :return:
        """

        # step 1: identify and remove Nan values
        idx_nan = np.where(df == -9999)[0]  # identify which indices have NaN value

        if drop_nan:
            df = df.drop(df.index[idx_nan])  # dropping data points with an NaN value
        else:
            pass

            # step 2: identify where demand is zero, this is not feasible
        idx_zero = np.where(df["Demand"] <= 0)[0]

        if drop_nan:
            df = df.drop(df.index[idx_zero])
        else:
            # df.loc[df["Demand"] == 0]["Demand"] = -9999
            df.loc[df["Demand"] == 0, "Demand"] = -9999

        mu_y, sigma_y = df["Demand"].mean(), df["Demand"].std()
        idx_1 = np.where(
            (df["Demand"] <= mu_y - 5 * sigma_y) | (df["Demand"] >= mu_y + 5 * sigma_y)
        )

        if drop_nan:
            df_out = df.drop(df.index[idx_1])
        else:
            df.loc[(df["Demand"] <= mu_y - 5 * sigma_y) | (df["Demand"] >= mu_y + 5 * sigma_y), "Demand"] = -9999
            df_out = df.copy()

        return df_out

    def partition_data(self, df):
        """This method takes in df as an argument and splits them into a training and a test set

        :param df: contains entire dataset specified using start_time and end_time

        :return df_t, df_e: dfs for trining and test data respectively

        """

        df_t = df[(df["Datetime"] <= self.split_time)]
        df_e = df[(df["Datetime"] > self.split_time)]

        return df_t, df_e

    def remove_federal_holidays(self, df):
        """Function to identify federal holidays. if federal holidays, the df[Holidays] = 1

        :param df:                                  input dataframe
        :type df:                                   pd.DataFrame

        :return:                                    Dataframe without holidays included

        """

        year_min, year_max = df["Datetime"].dt.year.min(), df["Datetime"].dt.year.max()
        years = np.arange(year_min, year_max + 1)
        holiday_list = holidays.US(years=years)

        if not str(df["Datetime"].dtype).startswith("datetime64"):
            df["Datetime"] = pd.to_datetime(df["Datetime"])

        bool = df["Datetime"].isin(holiday_list)
        df["Holidays"] = bool * 1

        return df

    def troubleshoot_by_plot(self):
        """ plot to troubleshoot if the training data has missing values.

        :return:

        """

        plt.plot(self.df_t["Datetime"], self.df_t["Demand"])
        plt.xlabel("Date/Time", fontsize=20)
        plt.ylabel("Demand (MW)", fontsize=20)
        plt.show()

        return None

    def preprocess_timedata(self, df):
        """TODO"""

        day_list = None

        if "Hour" in self.x_var and self.lin_model_bool == True:
            # this block of code indicates if we want to featurize a sine function or the raw input
            hr = df["Hour"]
            df["Hour"] = np.sin((hr * np.pi) / 24)  # 24 hours is pi

        if "Month" in self.x_var and self.lin_model_bool == True:
            # this block of code indicates if we want to make the function sinosoidal
            mnth = df["Month"]
            df["Month"] = np.sin((mnth * np.pi / 12))  # 12 months is pi

        if self.add_dayofweek:
            dayofweek = df["Datetime"].dt.dayofweek.values

            # dayofweek 0: monday, 6: sunday
            weekday = np.zeros_like(dayofweek)
            weekday[dayofweek <= 4] = 1
            df["Weekday"] = weekday

            # Create a day of week variable
            day_of_week = np.zeros((dayofweek.shape[0], 7))
            for d in range(7):
                tmp_val = np.zeros_like(dayofweek)
                tmp_val[dayofweek == d] = 1
                day_of_week[:, d] = tmp_val

            # concat day of week with df
            df_dayofweek = pd.DataFrame(day_of_week)

            # weeklist
            day_list = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            df_dayofweek.columns = day_list
            df = pd.concat((df, df_dayofweek), axis=1)

            # federal holidays added as a new feature
            df = self.remove_federal_holidays(df=df)

        return df, day_list

    def compute_pearson(self):
        """Computes the pearson coeff for each x in self.x_var and each y in self.y_var

        :return:

        """

        for x in self.x_var:
            for y in self.y_var:
                print(
                    "The PC between {} and {} is {}".format(
                        x, y, pearsonr(self.df[x].values, self.df[y].values)
                    )
                )

        return None
