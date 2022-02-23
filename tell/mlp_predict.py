import os
import glob
import warnings

import holidays
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import DataFrame
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore", category=UserWarning)

# set random seed, change this value as you prefer
np.random.seed(391)


class Dataset:
    """Class for pre-processing the data set for all BAs

    :param region:                      Indicating region/BA we want to train and test on. must match with str in csv
    :type region:                       str

    :param csv_dir:                     The directory where the csvs are located
    :type csv_dir:                      str

    :param start_time:                  The start time in YYYY-MM-DD. all data before this time as not considered
    :type start_time:                   str

    :param end_time:                    The end_time in YYYY-MM-DD. all data after this time is not considered
    :type end_time:                     str

    :param split_time:                  The partition time in YYYY-MM-DD to split into training and test_set
    :type split_time:                   str

    :param x_var:                       The features to consider
    :type x_var:                        list

    :param y_var:                       The targets to consider
    :type y_var:                        list

    :param add_dayofweek:               Whether we want to consider weekday vs weekend information
    :type add_dayofweek:                bool

    :param linear_mode_bool:            Whether it is a linear model or not. if so, the month and hour will need to be
                                        sinusoidal
    :type linear_mode_bool:             bool

    """

    def __init__(
            self,
            region='PJM',
            csv_dir="..\\CSV_Files",
            start_time="2016-01-01 00:00:00",
            end_time="2019-12-31 23:00:00",
            split_time="2018-12-31 23:00:00",
            x_var=["Hour", "Month", "Temperature", "Specific_Humidity", "Wind_Speed", "Longwave_Radiation",
                   "Shortwave_Radiation"],
            y_var=["Demand"],
            add_dayofweek=True,
            linear_mode_bool=False,
    ):

        self.region = region
        self.csv_dir = csv_dir
        self.start_time, self.end_time = start_time, end_time
        self.split_time = split_time
        self.x_var, self.y_var = x_var, y_var
        self.lin_model_bool = linear_mode_bool

        self.add_dayofweek = add_dayofweek

        # Get x_var and y_var
        self.df, self.df_t, self.df_e, self.df_eval, day_list = self.read_data()

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

    def read_data(self) -> DataFrame:
        """Function to read the data sets created in the data_process_compile_df.py script

        :return                 [0] df (pd.DataFrame) - entire data contained within start_time and end_time
                                [1] df_t (pd.DataFrame) - training data (before split_time)
                                [2] df_e (pd.DataFrame) - evaluation data (after split_time)

        """

        # step 1-> check filename
        filename = self.get_filename()
        # step 2-> read csv
        df = pd.read_csv(filename)
        # Step 2B - Rename the columns
        df = self.rename_columns(df)
        print(df.columns)
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

    @staticmethod
    def rename_columns(df: DataFrame):
        """New method to map the column names in data_process_compile_df.py to the names in the original code

        :param df:                  DataFrame created from the data_process_compile_df.py script
        :type df:                   DataFrame

        :return:                    DataFrame with renamed columns
        """

        map_dict = {
            "Adjusted_Demand_MWh": "Demand",
            "Pop": "Population",
            "T2": "Temperature",
            "SWDOWN": "Shortwave_Radiation",
            "GLW": "Longwave_Radiation",
            "WSPD": "Wind_Speed",
            "Q2": "Specific_Humidity"
        }

        df_out = df.rename(map_dict, axis="columns") if "Adjusted_Demand_MWh" in df.columns else df

        return df_out

    def preprocess_data(self):
        """Takes the features and targets from df

        :return:                    df_t -> training dataframe df_e -> eval dataframe

        """

        # sort the entire df by the column headers we require
        if self.add_dayofweek:
            self.x_var = self.x_var + ['Weekday', 'Holidays']

        # extract the training and test data. only including datetime, x_var (features) and y_var (targets)
        df_t, df_e, df_eval = [
            self.df_t[['Datetime'] + self.x_var + self.y_var],
            self.df_e[['Datetime'] + self.x_var + self.y_var],
            self.df_eval[['Datetime'] + self.x_var + self.y_var], ]

        X_t, X_e, X_eval = df_t[self.x_var], df_e[self.x_var], df_eval[self.x_var]
        Y_t, Y_e, Y_eval = df_t[self.y_var], df_e[self.y_var], df_eval[self.y_var]

        return df_t, df_e, df_eval, X_t, X_e, X_eval, Y_t, Y_e, Y_eval

    def get_filename(self) -> str:
        """Function to extract filename for specific region

        :return:                 filename: str filename corresponding to that region

        """

        str_to_check = os.path.join(self.csv_dir, f"{self.region}_*.csv")  # pattern to search for
        filename = glob.glob(str_to_check)[0]  # [0] list all file names with the string str_to_check

        return filename

    def sort_timeframe(self, df: DataFrame) -> DataFrame:
        """Function to filter data frame by specified time frame self.start_time and self.end Time

        :param df:                      DataFrame created from the data_process_compile_df.py script
        :type df:                       DataFrame

        :return:                        df_out: sorted df

        """

        df["Datetime"] = pd.to_datetime(df[["Day", "Month", "Year", "Hour"]])  # Hour added

        df = df[
            (df["Datetime"] >= self.start_time) & (df["Datetime"] <= self.end_time)]  # sort df by start time endtime

        df = df.reset_index(drop=True)  # reset index

        return df

    def drop_neg_rows(self, df: DataFrame, drop_nan=True) -> DataFrame:
        """This method drops -9999 values. it also drops "extreme" values, i.e. demand that lies outside +/- 5*sigma

        :param df:                      DataFrame created from the data_process_compile_df.py script
        :type df:                       DataFrame

        :param drop_nan:                Whether or not to drop NAN values (default is TRUE)
        :type drop_nan:                 bool

        :return:                        DataFrame without NAN values

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
            df.loc[df["Demand"] == 0, "Demand"] = -9999

        mu_y, sigma_y = df["Demand"].mean(), df["Demand"].std()
        idx_1 = np.where((df["Demand"] <= mu_y - 5 * sigma_y) | (df["Demand"] >= mu_y + 5 * sigma_y))

        if drop_nan:
            df_out = df.drop(df.index[idx_1])
        else:
            df.loc[(df["Demand"] <= mu_y - 5 * sigma_y) | (df["Demand"] >= mu_y + 5 * sigma_y), "Demand"] = -9999
            df_out = df.copy()

        return df_out

    def partition_data(self, df: DataFrame) -> DataFrame:
        """This method takes in df as an argument and splits them into a training and a test set

        :param df:                      DataFrame created from the data_process_compile_df.py script
        :type df:                       DataFrame

        :return                         [0] df_t,
                                        [1] df_e: dfs for training and test data respectively

        """

        df_t = df[(df["Datetime"] <= self.split_time)]
        df_e = df[(df["Datetime"] > self.split_time)]

        return df_t, df_e

    def remove_federal_holidays(self, df: DataFrame) -> DataFrame:
        """Function to identify federal holidays. if federal holidays, the df[Holidays] = 1

        :param df:                      DataFrame created from the data_process_compile_df.py script
        :type df:                       DataFrame

        :return:                        DataFrame without holidays included

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
        """Plot to troubleshoot if the training data has missing values.

        :return:                    None
        """

        plt.plot(self.df_t["Datetime"], self.df_t["Demand"])
        plt.xlabel("Date/Time", fontsize=20)
        plt.ylabel("Demand (MW)", fontsize=20)
        plt.show()

    def preprocess_timedata(self, df: DataFrame):
        """Function to manipulate time varaibles

        :param df:                      DataFrame created from the data_process_compile_df.py script
        :type df:                       DataFrame

        :return:                        [0] DataFrame with manipulated time variables
                                        [1] Day of week list

        """

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

            # week list
            day_list = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            df_dayofweek.columns = day_list
            df = pd.concat((df, df_dayofweek), axis=1)

            # federal holidays added as a new feature
            df = self.remove_federal_holidays(df=df)

        return df, day_list

    def compute_pearson(self):
        """Computes the pearson coeff for each x in self.x_var and each y in self.y_var

        :return:                   None

        """

        for x in self.x_var:
            for y in self.y_var:
                print(
                    "The PC between {} and {} is {}".format(
                        x, y, pearsonr(self.df[x].values, self.df[y].values)
                    )
                )


class Hyperparameters:
    """Class for hyper-parameter search only used when we need to optimize hyper-parameters for a specific balancing
    authority

    """

    def __init__(self, model_str: str, X: str, Y: str):
        """
        :param model_str: Indicating model type. currently only mlp is supported
        :type model_str: str

        :param X: features in training set
        :type X: str

        :param Y: targets in training set
        :type Y: str

        """

        # assigning model variables
        self.model_name = model_str
        self.X, self.Y = X, Y

        if model_str == "mlp":
            self.mlp()

    def mlp(self):
        """Performs hyper-parameter optimization for an mlp

        :return:                None

        """

        model = MLP(max_iter=1000)  # instantiate MLP model with 100 training epochs
        params = self.set_mlp_params()  # params -> a dict containing search space of all hyperparameters
        clf = GridSearchCV(model, params, verbose=1)  # performing grid search cross-validation for one BA
        clf.fit(self.X, self.Y)  # fit gridsearch obj with data (X, Y)

    def set_mlp_params(self):
        """Function to set hyperparameters for MLP currently only MLP with 1 hidden layer is supported. the only
        hyper-parameter is size of hidden layer

        :return                         params: dict containing search space for hyperparameters

        """

        params = dict()  # instantiate params

        # define hyperparameters search space. currently only supporting MLP with 1 hidden l
        params["hidden_layer_sizes"] = [16, 32, 64, 128, 256, 712, 1028]

        return params


class MlLib:
    """Class to train and evaluate ML models

    :param X_t:                         Training data
    :type X_t:                          np.array

    :param Y_t:                         Training targets
    :type Y_t:                          np.array

    :param X_e:                         Test data
    :type X_e:                          np.array

    :param Y_e:                         Test targets
    :type Y_e:                          np.array

    :param model:                       Type of model to pick
    :type model:                        str

    :param datetime:                    Array of dates for the model
    :type datetime:                     np.array

    :param fig_names:                   Names of all the figures we want, including time-series, seasonal prob dist,
                                        and cdf
    :type fig_names:                    dict

    :param dict_res:                    Data needed for training and evaluation of residual model
    :type dict_res:                     dict

    :param generate_plots:              Choice to generate and save plots
    :type generate_plots:               bool

    """

    def __init__(
            self,
            X_t,
            Y_t,
            X_e,
            fig_names=None,
            datetime=None,
            Y_e=None,
            Y_eval=None,
            model="mlp",
            dict_res=None,
            generate_plots=True,
            plot_gt=False
    ):

        self.generate_plots = generate_plots
        self.plot_gt = plot_gt

        # set data
        self.X_t, self.X_e, self.Y_t, self.Y_e = (
            X_t.values,
            X_e.values,
            Y_t.values,
            Y_e.values
        )

        self.Y_eval = Y_eval.values if Y_eval is not None else Y_eval

        self.model = model
        self.dict_res = dict_res
        self.datetime = datetime
        # set variable of fig_names
        self.fig_names = fig_names

        # get the dict for residuals, if we want to fit residuals to the model

        # scale features. normalized features in lowercase
        out = self.scale_features()
        self.x_t, self.x_e, self.y_t, self.y_e = (
            out["x_t"],
            out["x_e"],
            out["y_t"],
            out["y_e"],
        )
        self.out = out
        # train and predict using the model. currently 'mlp' and 'linear' supported
        self.y_p = self.pick_model()

        # evaluation
        self.analyze_results()

    def linear_model(self, X: np.array, Y: np.array, X_e: np.array):
        """Training and test data of a linear model. can be used for either the main model or the residual model

        :param X:                       Training features
        :type X:                        np.array

        :param Y:                       Training targets
        :type Y:                        np.array

        :param X_e:                     Test features
        :type X_e:                      np.array

        :return                         [0] y_p: predictions over test set
                                        [1] reg.coef_: regression coefficients of a linear model

        """

        # instantiate and fit linear model

        reg = LR().fit(X=X, y=Y)

        # get predictions
        y_p = reg.predict(X_e)

        return y_p, reg.coef_

    def mlp_model(self, X: np.array, Y: np.array, X_e: np.array, X_eval=None):
        """Trains the MLP model. also calls the linear residual model to adjust for population correction

        :param X:                       Training features
        :type X:                        np.array

        :param Y:                       Training targets
        :type Y:                        np.array

        :param X_e:                     Test features
        :type X_e:                      np.array

        :return:                        y_p: np.array -> predictions over test set

        """

        # instantiate and train the mlp model. we found 256 to be a good hyperparameter pick over some of larger bas
        # performing hyperparameter search over all BAs may be computationally expensive
        # accuracies for most BAs not sensitive to hyperparameters
        mlp = MLP(hidden_layer_sizes=256, max_iter=500, validation_fraction=0.1)
        mlp.fit(X, Y)
        y_p = mlp.predict(X_e)

        if self.dict_res is not None:
            # fit residuals using linear_residual
            y_tmp = mlp.predict(X)  # predict on training data

            # compute residuals: epsilon
            epsilon = Y - y_tmp  # residuals in the training data

            # train linear model to find residuals
            epsilon_e, _ = self.linear_residual(epsilon=epsilon)
            y_p = y_p + epsilon_e

        return y_p

    def linear_residual(self, epsilon: int):
        """Function to fit the residuals of MLP predictions

        :param epsilon:                 epsilon
        :type epsilon:                  int

        :return:                        epsilon_p

        """

        # fit the residuals with a linear model if a dict_res is provided
        # if self.dict_res is not None:
        Xres_t, Xres_e = self.dict_res["Xres_t"], self.dict_res["Xres_e"]
        epsilon_p = self.linear_model(X=Xres_t, Y=epsilon, X_e=Xres_e)

        return epsilon_p

    def pick_model(self):
        """Function to pick the "best" model

        :return:                        y_p

        """

        if self.model == "linear":
            y_p, coeff = self.linear_model(X=self.x_t, Y=self.y_t, X_e=self.x_e)
        elif self.model == "svr":
            y_p = self.svr_model(X=self.x_t, Y=self.y_t.squeeze(), X_e=self.x_e)
        elif self.model == "gpr":
            y_p = self.gpr_model(X=self.x_t, Y=self.y_t.squeeze(), X_e=self.x_e)
        elif self.model == "mlp":
            y_p = self.mlp_model(X=self.x_t, Y=self.y_t.squeeze(), X_e=self.x_e)
        else:
            y_p = None

        return y_p

    def scale_features(self) -> dict:
        """Function to scale the features of the model

        :return:                        dict_out, Dictionary of scaled features

        """

        # get the mean and std of training set
        mu_x, sigma_x = np.mean(self.X_t), np.std(self.X_t)
        mu_y, sigma_y = np.mean(self.Y_t), np.std(self.Y_t)

        # normalize
        x_t, x_e = np.divide((self.X_t - mu_x), sigma_x), np.divide(
            (self.X_e - mu_x), sigma_x
        )
        y_t = np.divide((self.Y_t - mu_y), sigma_y)

        if self.Y_e is not None:
            y_e = np.divide((self.Y_e - mu_y), sigma_y)
        else:
            y_e = None

        dict_out = {
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "x_t": x_t,
            "y_t": y_t,
            "x_e": x_e,
            "y_e": y_e,
        }

        return dict_out

    def unscale_targets(self, out: dict, y_p: np.array) -> np.array:
        """Function to unscale the targets of the model

        :param out:                Dict output from scale_features function
        :type out:                 dict

        :param y_p:                Predictions over test set
        :type y_p:                 np.array

        :return:                    Y_p

        """

        mu_y, sigma_y = out["mu_y"], out["sigma_y"]
        Y_p = y_p * sigma_y + mu_y

        return Y_p

    def analyze_results(self):
        """Function to compute the evaluation metrics, and prepare plots

        :return:                    None

        """

        # define locator and formatter for plots
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)

        # first revert to the absolute values
        self.Y_p = self.unscale_targets(out=self.out, y_p=self.y_p)

        # complile results and predictions into a timeframe
        data = {
            "Datetime": self.datetime,
            "Y_p": self.Y_p.squeeze(),
            "Y_e": self.Y_e.squeeze(),
        }

        self.df_results = pd.DataFrame(data)

        if self.generate_plots:
            # evaluate metrics
            plt.rcParams.update({"font.size": 16})

            # set mdates formatter
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)

            # quick plot:
            time = np.arange(0, self.Y_e.shape[0])
            fig, ax1 = plt.subplots()

            if self.plot_gt:
                ax1.plot(self.datetime, np.ma.masked_where(self.Y_e <= 0, self.Y_e), label="Ground Truth")
            ax1.plot(self.datetime, self.Y_p, label="Predictions")
            ax1.xaxis.set_major_locator(locator)
            ax1.xaxis.set_major_formatter(formatter)

            plt.xlabel("Date/Time")
            plt.ylabel("Electricity Demand (MWh)")
            plt.legend()
            plt.tight_layout()

            if self.fig_names is not None:
                print("Created figure: ", self.fig_names["timeSeries"])
                plt.savefig(self.fig_names["timeSeries"])

            else:
                plt.show()

            # close figure
            plt.close()

        # evaluate the model
        self.evaluation_metrics()

    def evaluation_metrics(self):
        """Method to compute the evaluation metrics

        :return:                        None

        """

        # remove all the nan in self.Y_eval before evaluation
        idx_notnan = np.where(self.Y_eval != -9999)
        Y_p = self.Y_p[idx_notnan[0]]
        Y_eval = self.Y_eval[idx_notnan[0]].squeeze()

        # first the absolute root-mean-squared error
        self.rms_abs = np.sqrt(mean_squared_error(Y_p, Y_eval))
        self.rms_norm = self.rms_abs / np.mean(Y_eval)
        self.mape = mean_absolute_percentage_error(Y_p, Y_eval)
        self.r2_val = r2_score(Y_p, Y_eval)

        print("RMS-ABS: ", self.rms_abs)
        print("RMS NORM: ", self.rms_norm)
        print("MAPE: ", self.mape)
        print("R2 value: ", self.r2_val)

    def evaluate_peaks(self):
        """This function is to compute the difference in peak values and in peak timings

        :return:                    None

        """

        df_grp = self.df_results.groupby(pd.Grouper(key="Datetime", freq="D"))
        idx = np.where(np.isnan(self.df_results["Y_e"]))
        idx_Ye = df_grp["Y_e"].idxmax().values
        idx_Yp = df_grp["Y_p"].idxmax().values
        peak_Ye, peak_Yp = self.Y_e[idx_Ye], self.Y_p[idx_Yp]

        # define peak times
        time_Ye, time_Yp = self.datetime[idx_Ye], self.datetime[idx_Yp]

        delta_time = np.abs(idx_Yp - idx_Ye)
        # if it's more than 12 hours
        delta_time[delta_time > 12] = 24 - delta_time[delta_time > 12]
        self.peak_diff = np.mean(delta_time)
        self.peak_abs = np.sqrt(mean_squared_error(peak_Ye, peak_Yp))
        self.peak_err = mean_absolute_percentage_error(peak_Ye, peak_Yp)

    def monthly_plots(self):
        """This function creates the monthly plots of observed and predicted hourly load

         :return:                    None

         """

        # define month_list for plots
        MONTH_LIST = ["Jan", "Feb", "Mar", "April", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        # set rcparams
        plt.rcParams.update({"font.size": 18})

        df_grp = self.df_results.groupby(pd.Grouper(key="Datetime", freq="M"))
        df_month = [grp for _, grp in df_grp]

        # extract string for fig_name
        main_str = self.fig_names["dist"].rsplit(".svg")[0]

        for m, df in enumerate(df_month):
            fig, ax1 = plt.subplots()
            plt.hist(
                (df["Y_e"] - df["Y_p"]) / df["Y_e"],
                density=True,
                bins=75,
                histtype="bar",
                color="tab:blue",
                stacked=True,
            )
            fig_hist = main_str + "_" + MONTH_LIST[m] + ".svg"
            plt.tight_layout()
            plt.savefig(fig_hist)
            plt.clf()

            # print median value
            e = (df["Y_e"] - df["Y_p"]) / df["Y_e"]


class Analysis:
    """Train and evaluate each individual BAs. Generates output CSV files under directory outputs
    Trains the "residual model" as well for population correction.

    :param data_dir:                Full path to dir for data source
    :type data_dir:                 str

    :param out_dir:                 Full path to dir for output csvs and figures
    :type data_dir:                 str

    :param start_time:              Start-time for analysis, in YYYY-MM-DD HH:MM:SS. Default: 2016-01-01 00:00:00
    :type start_time:               str

    :param end_time:                End time for analysis. Default: 2019-12-31 23:00:00
    :type end_time:                 str

    :param split_time:              Timestamp splitting train and test data. Default: 2018-12-31 23:00:00
    :type split_time:               str

    """

    def __init__(
            self,
            data_dir,
            out_dir,
            start_time="2016-01-01 00:00:00",
            end_time="2019-12-31 23:00:00",
            split_time="2018-12-31 23:00:00",
            region="PJM",
            generate_plots=True,
            plot_gt=False
    ):

        self.start_time = start_time
        self.end_time = end_time
        self.split_time = split_time

        self.generate_plots = generate_plots
        self.plot_gt = plot_gt

        # note, region is same as BA!
        self.region = region

        # specify feature set for residuals
        self.x_res = ["Population", "Hour", "Month", "Year"]

        # specify dataset for both main MLP and residual linear model
        self.data = Dataset(
            region=region,
            start_time=start_time,
            end_time=end_time,
            split_time=split_time,
            csv_dir=data_dir
        )

        # data for residual model
        self.data_res = Dataset(
            region=region,
            x_var=self.x_res,
            linear_mode_bool=True,
            start_time=start_time,
            end_time=end_time,
            split_time=split_time, csv_dir=data_dir
        )

        # define training and test data for residual fits
        # training and test data for main MLP model
        self.X_t, self.Y_t, self.X_e, self.Y_e = (
            self.data.X_t,
            self.data.Y_t,
            self.data.X_e,
            self.data.Y_e,
        )

        self.Y_eval = self.data.Y_eval
        # training and test data for residual model
        self.Xres_t, self.Yres_t, self.Xres_e, self.Yres_e = (
            self.data_res.X_t,
            self.data_res.Y_t,
            self.data_res.X_e,
            self.data_res.Y_e,
        )

        # resetting indices for test data: both main model and residual model
        self.df_e, self.df_res_e = (
            self.data.df_e.reset_index(drop=True),
            self.data_res.df_e.reset_index(drop=True),
        )

        # list of models for which analysis is to be performed. currently only 'linear' and 'mlp' supported
        self.list_of_models = ["mlp"]

        self.out_dir = out_dir

        # create output dir if it not already created
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # test multiple models. currently only testing for 'mlp' model
        self.test_multiple_models()

    def test_multiple_models(self):
        """This function tests multiple model types

         :return:                    None

         """

        Yp_list = []
        # get labels for predictions based on type of model
        labels = [str_name + " predictions" for str_name in self.list_of_models]

        for m, model in enumerate(self.list_of_models):

            print("----CHECKING PREDICTIVE MODELS-----")
            fig_names = self.set_fignames(model_name=labels[m])
            # set up the dict for error correction
            err_correct = {"Xres_t": self.Xres_t, "Xres_e": self.Xres_e}

            # instantiate ml_lib object for to train the model and get predictions over the test set
            ml = MlLib(
                X_t=self.X_t,
                Y_t=self.Y_t,
                X_e=self.X_e,
                Y_e=self.Y_e,
                Y_eval=self.Y_eval,
                datetime=self.df_e["Datetime"],
                fig_names=fig_names,
                model=model,
                dict_res=err_correct,
                generate_plots=self.generate_plots,
                plot_gt=self.plot_gt
            )

            # get the predictions from the ML model
            Y_p = ml.Y_p
            self.df_e[labels[m]] = Y_p

            # currently regression is commented out, but you can have a regression plot if you chose
            # self.plot_reg(Y_a=self.Y_e, Y_p=Y_p, label=labels[m])

            # write to file
            self.write_output_to_file(Y_p=Y_p, model=model)

            # export for bar chart
            if model == "mlp":
                self.R2 = ml.r2_val
                self.MAPE = ml.mape

    def set_fignames(self, model_name: str) -> dict:
        """Set figure names for timeseries plots and probability distributions of error residuals by month

        :param model_name:                   Model name (e.g. 'linear', 'mlp')
        :type model_name:                    str

        :return:                            fig_names: dict -> containing list of figures for 'timeSeries' and 'dist'

        """

        fig_names = dict()
        fig_names["timeSeries"] = (
            os.path.join(self.out_dir, f"{self.region}_{model_name}_timeseries.svg")
        )
        fig_names["dist"] = (
            os.path.join(self.out_dir, f"{self.region}_{model_name}_probdist.svg")
        )

        return fig_names

    def plot_reg(self, Y_a, Y_p, label):
        """ Method for regression plots (automatically saves them)

        :param Y_a:                     Array containing ground truth
        :type Y_a:                      np.array

        :param Y_p:                     Array containing predicted hourly load
        :type Y_a:                      np.array

        :param label:                   Label for plot
        :type label:                    str

        :return:                        None
        """

        plt.rcParams.update({"font.size": 16})
        fig, ax = plt.subplots()
        plt.scatter(Y_a, Y_p)
        plt.plot(Y_a, Y_a, "r-", linewidth=3)
        plt.plot(Y_a, 1.1 * Y_a, "r--", linewidth=3)
        plt.plot(Y_a, 0.9 * Y_a, "r--", linewidth=3)
        plt.xlabel("Actual forecast of electricity Demand (MWh)")
        plt.ylabel("Predictions of electricity demand (MWh)")

        fig_name = os.path.join(self.out_dir, f"{self.region}_{label}.svg")
        plt.tight_layout()
        plt.savefig(fig_name)

    def write_output_to_file(self, Y_p: np.array, model: str):
        """This function is used to write output of each model

        :param Y_p:                     Model hourly load predictions
        :type Y_p:                      np.array

        :param model:                   Label for the model
        :type model:                    str

        :return:                        None

        """

        # need to flatten self.Y_e, as the dimensions are (..,1)
        out = {
            "Datetime": self.df_e["Datetime"].values,
            "Adjusted_Demand_MWh": self.Y_e.squeeze(),
            "NN-Predicted_Demand_MWh": Y_p,
        }

        # export as pandas dataframe and write to file
        out = pd.DataFrame(out).reset_index(drop=True)
        csv_filename = (
            os.path.join(self.out_dir, f"{self.region}_{model}_predictions.csv")
        )
        out.to_csv(csv_filename, index=False)

        return None


class Process:
    """Run multiple BA

    :param start_time:              Start-time for analysis, in YYYY-MM-DD HH:MM:SS. Default: 2016-01-01 00:00:00
    :type start_time:               str

    :param end_time:                End time for analysis. Default: 2019-12-31 23:00:00
    :param end_time:                str

    :param split_time:              Timestamp splitting train and test data. Default: 2018-12-31 23:00:00
    :type split_time:               str

    :param batch_run:               Indicating if we want to run the simulations for all BAs, or we handpick the BAs
                                    If batch_run = True, the code will search for all BAs in 'dir'
                                    If batch_run = False, we need to specify which BA to run
    :type batch_run:                bool

    :param data_dir:                Full path to the directory containing the target
                                    CSV files
    :type data_dir:                 str

    :param out_dir:                 Full path to the directory where the outputs are to be written
    :type out_dir:                  str

    :param target_ba_list:          A list of BA names to run if `batch_run` is False
    :type target_ba_list:           list

    :param generate_plots:          Choice to generate and save plots
    :type generate_plots:           bool

    :param plot_gt:                 Choice to plot ground truth data in plot along with MLP predictions
    :type plot_gt:                  bool

    """

    def __init__(
            self,
            start_time="2016-01-01 00:00:00",
            end_time="2019-12-31 23:00:00",
            split_time="2018-12-31 23:00:00",
            batch_run=False,
            data_dir=None,
            out_dir=None,
            target_ba_list=None,
            generate_plots=True,
            plot_gt=False
    ):

        self.start_time = start_time
        self.end_time = end_time
        self.split_time = split_time

        self.data_dir = data_dir
        self.out_dir = out_dir
        self.target_ba_list = target_ba_list
        self.generate_plots = generate_plots
        self.plot_gt = plot_gt

        # output summary file
        self.out_summary_file = os.path.join(self.out_dir, 'summary.csv')

        # checking dir to get list of CSV files
        self.pat_to_check = os.path.join(self.data_dir, '*.csv')

        if batch_run:
            # case to run for all BAs
            self.ba_list = self.search_for_pattern()
            if not len(self.ba_list) > 0:
                raise AssertionError("No csvs found in the directory!")

        else:
            # case for handpick BAs
            self.ba_list = self.target_ba_list

        # loop over all BAs. generates summary.csv to show accuracy of all BAs
        self.summary_df = self.gen_results()  # steo ii: gen_results

    def search_for_pattern(self) -> list:
        """Sets self.ba_list to get a list of BAs for training and evaluation.

        :return:            List of BAs to process

        """

        list_of_files = sorted(glob.glob(self.pat_to_check))

        BA_list = []
        for filename in list_of_files:
            main_str = filename.split(os.path.sep)[-1]
            main_str = main_str.split("_")[0]  # get the BA name
            BA_list.append(main_str)

        return BA_list

    def gen_results(self) -> DataFrame:
        """Writes all outputs to csvs + a summary file with the evaluation metrics.

        :return:                        Summary file of model statistics

        """

        ba_out, r2, mape = [], [], []

        for BA_name in self.ba_list:
            print("BA: {}".format(BA_name))
            try:
                # perform analysis for each BA, keep track of all BAs and corresponding accuracy metrics
                ba = Analysis(
                    region=BA_name,
                    start_time=self.start_time,
                    end_time=self.end_time,
                    split_time=self.split_time,
                    out_dir=self.out_dir,
                    data_dir=self.data_dir,
                    generate_plots=self.generate_plots,
                    plot_gt=self.plot_gt
                )

                ba_out.append(BA_name), r2.append(ba.R2), mape.append(ba.MAPE)

            except ValueError:
                continue

        # write to file
        out = {"BA": ba_out, "R2": r2, "MAPE": mape}
        df = pd.DataFrame(out)

        df.to_csv(self.out_summary_file, index=False)

        return df


def single_ba(data_dir, out_dir, ba='PJM', start_time="2016-01-01 00:00:00", end_time="2018-07-14 23:00:00",
             split_time="2018-06-30 23:00:00", plot_gt=False):
    """
    Convenience wrapper for the Analysis class which runs predictive models for a single BA input CSV in the input
    calls on the analysis function to train and infer on single BA
    useful for troubleshooting

    :param data_dir:                Full path to the directory containing the target CSV files
    :type data_dir:                 str

    :param out_dir:                 Full path to the directory where the outputs are to be written
    :type data_dir:                 str

    :param ba:                      Balancing authority to run analysis. default: PJM
    :type ba:                       str

    :param start_time:              Start-time for analysis, in YYY-MM-DD HH:MM:SS. Default: 2016-01-01 00:00:00
    :type start_time:               str

    :param end_time:                End time for analysis. Default: 2019-12-31 23:00:00
    :type end_time:                 str

    :param split_time:              Timestamp splitting train and test data. Default: 2018-12-31 23:00:00
    :type split_time:               str

    :param plot_gt:                 Choice to plot ground truth with MLP predictions
    :type plot_gt:                  bool

    :return:                        Data frame of BA, R2, MAPE statistics

    """

    Analysis(
        data_dir=data_dir,
        out_dir=out_dir,
        region=ba,
        start_time=start_time,
        split_time=split_time,
        end_time=end_time,
        plot_gt=plot_gt
    )


def predict(data_dir, out_dir, start_time="2016-01-01 00:00:00", end_time="2019-12-31 23:00:00",
        split_time="2018-12-31 23:00:00", batch_run=True, target_ba_list=None, generate_plots=True):

    """Convenience wrapper for the Process class which runs predictive models for each BA input CSV in the input
    directory and creates a summary and comparative figures of R2 and MAPE per BA.

    :param data_dir:                Full path to the directory containing the target CSV files
    :type data_dir:                 str

    :param out_dir:                 Full path to the directory where the outputs are to be written
    :type out_dir:                  str

    :param start_time:              Start-time for analysis, in YYY-MM-DD HH:MM:SS. Default: 2016-01-01 00:00:00
    :type start_time:               str

    :param end_time:                End time for analysis. Default: 2019-12-31 23:00:00
    :type end_time:                 str

    :param split_time:              Timestamp splitting train and test data. Default: 2018-12-31 23:00:00
    :type split_time:               str

    :param batch_run:               Indicating if we want to run the simulations for all BAs, or we handpick the BAs
                                    If batch_run = True, the code will search for all BAs in 'dir'
                                    If batch_run = False, we need to specify which BA to run
    :type batch_run:                bool

    :param target_ba_list:          A list of BA names to run if `batch_run` is False
    :type target_ba_list:           list


    :param generate_plots:          Choice to generate and save plots
    :type generate_plots:           bool

    :return:                        Data frame of BA, R2, MAPE statistics

    """

    proc = Process(
        start_time=start_time,
        end_time=end_time,
        split_time=split_time,
        batch_run=batch_run,
        data_dir=data_dir,
        out_dir=out_dir,
        target_ba_list=target_ba_list,
        generate_plots=generate_plots
    )

    return proc.summary_df
