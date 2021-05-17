"""
this file extracts data and predicts time series
"""

import os
import glob

import numpy as np
import pandas as pd
import holidays

from joblib.parallel import Parallel, delayed

from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.neural_network import MLPRegressor as MLP
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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


# set random seed, change this value as you prefer
np.random.seed(391)


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
                 #'Longwave_Radiation', 'Wind_Speed'
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
        self.df, self.df_t, self.df_e, day_list = self.read_data()

        # get training and test data
        (
            self.df_t,
            self.df_e,
            self.X_t,
            self.X_e,
            self.Y_t,
            self.Y_e,
        ) = self.preprocess_data(day_list=day_list)

    def read_data(self):

        """
        Function to read csvs
        :return: X, Y
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

        return df, df_t, df_e, day_list

    def preprocess_data(self, day_list=None):
        """Takes the features and targets from df

        :return: df_t -> training dataframe df_e -> eval dataframe

        """

        # sort the entire df by the column headers we require
        if self.add_dayofweek == True:
            self.x_var = self.x_var + ["Weekday", "Holidays"]

        # extract the training and test data. only including datetime, x_var (features) and y_var (targets)
        df_t, df_e = (
            self.df_t[["Datetime"] + self.x_var + self.y_var],
            self.df_e[["Datetime"] + self.x_var + self.y_var],
        )
        X_t, X_e = df_t[self.x_var], df_e[self.x_var]
        Y_t, Y_e = df_t[self.y_var], df_e[self.y_var]

        return df_t, df_e, X_t, X_e, Y_t, Y_e

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

        # dropping negative rows
        df = self.drop_neg_rows(
            df=df
        )  # might need to replace with interpolation, but currently simply drops the -ve rows
        df = df.reset_index(drop=True)  # rese

        return df

    def drop_neg_rows(self, df):

        """
        This method drops -9999 values. it also drops "extreme" values, i.e. demand that lies outside +/- 5*sigma
        :param df:
        :return:
        """

        # step 1: identify and remove Nan values
        idx_nan = np.where(df == -9999)[0]  # identify which indices have NaN value
        df = df.drop(df.index[idx_nan])  # dropping data points with an NaN value

        # step 2: identify where demand is zero, this is not feasible
        idx_zero = np.where(df["Demand"] == 0)[0]
        df = df.drop(df.index[idx_zero])

        ###dropping excessively high or low values (outside +/- 5*sigma)
        mu_y, sigma_y = df["Demand"].mean(), df["Demand"].std()
        idx_1 = np.where(
            (df["Demand"] <= mu_y - 5 * sigma_y) | (df["Demand"] >= mu_y + 5 * sigma_y)
        )
        df_out = df.drop(df.index[idx_1])

        return df_out

    def partition_data(self, df):

        """
        This method takes in df as an argument and splits them into a training and a test set
        :param df: contains entire dataset specified using start_time and end_time
        :return df_t, df_e: dfs for trining and test data respectively
        """

        df_t = df[(df["Datetime"] <= self.split_time)]
        df_e = df[(df["Datetime"] > self.split_time)]

        return df_t, df_e

    def remove_federal_holidays(self, df):

        """
        Function to identify federal holidays. if federal holidays, the df[Holidays] = 1
        :param df: input dataframe
        :param year: year for which the holiday list needs to be determined. default -> global variable YEAR
        :return: Dataframe without holidays included
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

        """
        plot to troubleshoot if the training data has missing values.
        :return:
        """

        plt.plot(self.df_t["Datetime"], self.df_t["Demand"])
        plt.xlabel("Date/Time", fontsize=20)
        plt.ylabel("Demand (MW)", fontsize=20)
        plt.show()

        return None

    def preprocess_timedata(self, df):

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

        """
        Computes the pearson coeff for each x in self.x_var and each y in self.y_var
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


class Hyperparameters:

    """
    class for hyperparameter search
    only used when we need to optimize hyperpaaremters for a specific balancing authority
    """

    def __init__(self, model_str, X, Y):
        """

        :param model_str: str -> indicating model type. currently only mlp is supported
        :param X: features in training set
        :param Y: targets in training set
        """

        #assigning model variables
        self.model_name = model_str
        self.X, self.Y = X, Y

        if model_str == "mlp":
            self.mlp()

    def mlp(self):

        """
        performs hyperparameter optimization for an mlp
        :return:
        """
        model = MLP(max_iter=1000) #instantiate MLP model with 100 training epochs
        params = self.set_mlp_params() #params -> a dict containing search space of all hyperparameters
        clf = GridSearchCV(model, params, verbose=1) #performing grid search cross-validation for one BA
        clf.fit(self.X, self.Y) #fit gridsearch obj with data (X, Y)

        #print(clf.cv_results_)

        return None

    def set_mlp_params(self):

        """
        function to set hyperparameters for MLP
        currently only MLP with 1 hidden layer is supported. the only hyperparameter is size of hidden layer
        :return params: dict containing search space for hyperparameters
        """
        params = dict() #instantiate params
        #define hyperparameters search space. currently only supporting MLP with 1 hidden l
        params["hidden_layer_sizes"] = [16, 32, 64, 128, 256, 712, 1028]

        return params


class MlLib:
    """TODO:  describe this class

    :param X_t: df -> training data,
    :param Y_t: df -> training targets
    :param X_e: df -> test data
    :param Y_e: df -> test targets
    :param model: str type of model to pick
    :param datetime: array of dates for the
    :param_fig_names: dict containing names of all the figures we want, including timeseries, seasonal prob dist, and cdf
    :param dict_res: dict containing data needed for training and avaluation of residual model

    :param generate_plots:              Choice to generate and save plots
    :type generate_plots:               bool

    """

    def __init__(self,
                 X_t,
                 Y_t,
                 X_e,
                 fig_names=None,
                 datetime=None,
                 Y_e=None,
                 model="mlp",
                 dict_res=None,
                 generate_plots=True):

        self.generate_plots = generate_plots

        # set data
        self.X_t, self.X_e, self.Y_t, self.Y_e = (
            X_t.values,
            X_e.values,
            Y_t.values,
            Y_e.values,
        )
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
        # self.evaluation_metrics()

    def linear_model(self, X, Y, X_e):

        """
        training and test data of a linear model. can be used for either the main model or the residual model
        :param X: training features
        :param Y: training targets
        :param X_e: test features
        :return y_p: predictions over test set
        :return reg.coef_: regression coefficients of a linear model
        """

        # instantiate and fit linear model
        reg = LR().fit(X=X, y=Y)

        # get predictions
        y_p = reg.predict(X_e)

        if self.y_e is not None:
            print(r2_score(y_p, self.y_e))

        return y_p, reg.coef_

    def mlp_model(self, X, Y, X_e):

        """
        trains the MLP model. also calls the linear residual model to adjust for population correction
        :param X: arr -> train features
        :param Y: arr -> train targets
        :param X_e: arr -> test features
        :return: y_p: arr -> predictions over test set
        """

        # currently hyperparameter selection is deactivate
        # hyp_mlp = hyperparameters(model_str='mlp', X=X, Y=Y)

        # instantiate and train the mlp model. we found 256 to be a good hyperparameter pick over some of larger bas
        # performing hyperparameter search over all BAs may be computationally expensive
        # accuracies for most BAs not sensitive to hyperparameters
        mlp = MLP(hidden_layer_sizes=256, max_iter=500, validation_fraction=0.1)
        mlp.fit(X, Y)
        y_p = mlp.predict(X_e)

        if self.y_e is not None:
            print(r2_score(y_p, self.y_e.squeeze()))

        if self.dict_res is not None:
            # fit residuals using linear_residual
            y_tmp = mlp.predict(X)  # predict on training data

            # compute residuals: epsilon
            epsilon = Y - y_tmp  # residuals in the training data

            # train linear model to find residuals
            epsilon_e, _ = self.linear_residual(epsilon=epsilon)
            y_p = y_p + epsilon_e

        return y_p

    def linear_residual(self, epsilon):

        """
        function to fit the residuals of MLP predictions
        """

        # fit the residuals with a linear model if a dict_res is provided
        # if self.dict_res is not None:
        Xres_t, Xres_e = self.dict_res["Xres_t"], self.dict_res["Xres_e"]
        epsilon_p = self.linear_model(X=Xres_t, Y=epsilon, X_e=Xres_e)

        return epsilon_p

    def pick_model(self):

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

    def scale_features(self):

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

    def unscale_targets(self, out, y_p):

        mu_y, sigma_y = out["mu_y"], out["sigma_y"]
        Y_p = y_p * sigma_y + mu_y

        return Y_p

    def analyze_results(self):

        """
        Function to compute the evaluation metrics, and prepare plots
        :return:
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

            ax1.plot(self.datetime, self.Y_e, label="Ground Truth")
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

        return None

    def evaluation_metrics(self):

        # first the absolute root-mean-squared error
        self.rms_abs = np.sqrt(mean_squared_error(self.Y_p, self.Y_e))

        # next, the root mean squared error as a function of
        self.rms_norm = self.rms_abs / np.mean(self.Y_e)
        self.mape = mean_absolute_percentage_error(self.Y_e, self.Y_p)

        # R2 score
        self.r2_val = r2_score(self.y_p, self.y_e)

        print("RMS-ABS: ", self.rms_abs)
        print("RMS NORM: ", self.rms_norm)
        print("MAPE: ", self.mape)
        print("R2 value: ", self.r2_val)


        return None

    def evaluate_peaks(self):

        """
        This function is to compute the difference in peak values and in peak timimngs
        :return:
        """

        df_grp = self.df_results.groupby(pd.Grouper(key="Datetime", freq="D"))
        idx = np.where(np.isnan(self.df_results["Y_e"]))
        # print(df_grp['Y_e'])
        idx_Ye = df_grp["Y_e"].idxmax().values
        idx_Yp = df_grp["Y_p"].idxmax().values
        peak_Ye, peak_Yp = self.Y_e[idx_Ye], self.Y_p[idx_Yp]

        # define peaktimes
        time_Ye, time_Yp = self.datetime[idx_Ye], self.datetime[idx_Yp]

        delta_time = np.abs(idx_Yp - idx_Ye)
        # if it's more than 12 hours
        delta_time[delta_time > 12] = 24 - delta_time[delta_time > 12]
        self.peak_diff = np.mean(delta_time)

        self.peak_abs = np.sqrt(mean_squared_error(peak_Ye, peak_Yp))
        self.peak_err = mean_absolute_percentage_error(peak_Ye, peak_Yp)

        return None

    def monthly_plots(self):

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
            # plt.xlabel('$(Y_e - Y_p)/Y_e$')
            # plt.ylabel('p')
            plt.tight_layout()
            plt.savefig(fig_hist)
            plt.clf()

            # print median value
            e = (df["Y_e"] - df["Y_p"]) / df["Y_e"]
            print("Median for month {} is {}".format(MONTH_LIST[m], e.median()))

        return None


class Analysis:

    def __init__(self, data_dir, out_dir, region="PJM", generate_plots=True):
        """Train and evaluate each individual BAs. Generates output CSV files under directory outputs
        Trains the "residual model" as well for population correction.

        :param df: df corresponding to the evaluation period
        :param Y_e: ground truth during the evaluation period

        """

        self.generate_plots = generate_plots

        # note, region is same as BA!
        self.region = region

        # specify feature set for residuals
        self.x_res = ["Population", "Hour", "Month", "Year"]

        # specify dataset for both main MLP and residual linear model
        self.data = Dataset(region=region, csv_dir=data_dir)

        # data for residual model
        self.data_res = Dataset(region=region, x_var=self.x_res, linear_mode_bool=True, csv_dir=data_dir)

        # define training and test data for residual fits
        # training and test data for main MLP model
        self.X_t, self.Y_t, self.X_e, self.Y_e = (
            self.data.X_t,
            self.data.Y_t,
            self.data.X_e,
            self.data.Y_e,
        )
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
                datetime=self.df_e["Datetime"],
                fig_names=fig_names,
                model=model,
                dict_res=err_correct,
                generate_plots=self.generate_plots
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

        return None

    def set_fignames(self, model_name):

        """Set figure names for timeseries plots and probability distributions of error residuals by month

        :param model_name: str -> model name (e.g. 'linear', 'mlp')
        :return: fig_names: dict -> containing list of figures for 'timeSeries' and 'dist'

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

        """
        Method for regression Plots
        :param Y_a: array containing ground truth
        :param Y_p:
        :param label:
        :return:
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

        return None

    def write_output_to_file(self, Y_p, model):

        """
        This function is used to write output
        Y_p: predictions (np array)
        model: str to label which model it is
        :return:
        """

        # need to flatten self.Y_e, as the dimensions are (..,1)
        out = {
            "Datetime": self.df_e["Datetime"].values,
            "Ground Truth": self.Y_e.squeeze(),
            "Predictions": Y_p,
        }

        # export as pandas dataframe and write to file
        out = pd.DataFrame(out).reset_index(drop=True)
        csv_filename = (
            os.path.join(self.out_dir, f"{self.region}_{model}_predictions.csv")
        )
        out.to_csv(csv_filename, index=False)

        return None


class Process:

    def __init__(self, batch_run=False, data_dir=None, out_dir=None, target_ba_list=None, generate_plots=True,
                 write_summary=False):
        """Run multiple BA

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

        :param write_summary:           Choice to write summary output file


        """

        self.data_dir = data_dir
        self.out_dir = out_dir
        self.target_ba_list = target_ba_list
        self.generate_plots = generate_plots

        # output summary file
        self.out_summary_file = os.path.join(self.out_dir, 'summary.csv')

        # checking dir to get list of CSV files
        self.pat_to_check = os.path.join(self.data_dir, '*.csv')

        if batch_run:
            # case to run for all BAs
            self.ba_list = self.search_for_pattern()

        else:
            # case for handpick BAs
            self.ba_list = self.target_ba_list

        # loop over all BAs. generates summary.csv to show accuracy of all BAs
        self.summary_df = self.gen_results()  # steo ii: gen_results

    def search_for_pattern(self):
        """Sets self.ba_list to get a list of BAs for training and evaluation.

        :return:            List of BAs to process

        """

        list_of_files = sorted(glob.glob(self.pat_to_check))

        BA_list = []
        for filename in list_of_files:

            main_str = filename.split("\\")[-1]
            main_str = main_str.split("_")[0]  # get the BA name
            BA_list.append(main_str)

        return BA_list

    def gen_results(self):
        """Writes all outputs to csvs + a summary file with the evaluation metrics.

        :return:

        """

        ba_out, r2, mape = [], [], []

        for BA_name in self.ba_list:
            print("BA: {}".format(BA_name))
            try:
                # perform analysis for each BA, keep track of all BAs and corresponding accuracy metrics
                ba = Analysis(region=BA_name,
                              out_dir=self.out_dir,
                              data_dir=self.data_dir,
                              generate_plots=self.generate_plots)

                ba_out.append(BA_name), r2.append(ba.R2), mape.append(ba.MAPE)

            except ValueError:
                continue

        # write to file
        out = {"BA": ba_out, "R2": r2, "MAPE": mape}
        df = pd.DataFrame(out)

        df.to_csv(self.out_summary_file, index=False, mode='a')

        return df


def aggregate_summary(output_list):
    """Aggregates summary output for all BAs

    :return:Summary of all R2 and MAPE for all BAs
    """

    output_dfs = [i.summary_df for i in output_list]

    return pd.concat(output_dfs)

def list_ba(input_list):
        """Sets list of BAs for training and evaluation fro target BA in predict.

        :return:            List of BAs to process

        """

        path_to_check = os.path.join(input_list, '*.csv')
        list_of_files = sorted(glob.glob(path_to_check))

        ba_names = []
        for filename in list_of_files:

            main_str = filename.split("\\")[-1]
            main_str = main_str.split("_")[0]  # get the BA name
            ba_names.append(main_str)

        return ba_names


def predict(data_dir, out_dir, batch_run=True, target_ba_list=None, generate_plots=True, run_parallel=True, n_jobs=-1):
    """Convenience wrapper for the Process class which runs predictive models for each BA input CSV in the input
    directory and creates a summary and comparative figures of R2 and MAPE per BA.

    :param data_dir:                Full path to the directory containing the target
                                    CSV files
    :type data_dir:                 str

    :param out_dir:                 Full path to the directory where the outputs are to be written
    :type out_dir:                  str

    :param batch_run:               Indicating if we want to run the simulations for all BAs, or we handpick the BAs
                                    If batch_run = True, the code will search for all BAs in 'dir'
                                    If batch_run = False, we need to specify which BA to run
    :type batch_run:                bool

    :param target_ba_list:          A list of BA names to run if `batch_run` is False
    :type target_ba_list:           list


    :param generate_plots:          Choice to generate and save plots
    :type generate_plots:           bool

    :param run_parallel:            Choose to run BA's in parallel
    :type run_parallel:             bool

    :param n_jobs:                  Set number of CPUs
    :type n_jobs:                   int

    :return:                        Data frame of BA, R2, MAPE statistics

    """

    if run_parallel:
        ba_list = list_ba(input_list=data_dir)
        outputs = Parallel(n_jobs=n_jobs)(delayed(Process)(batch_run=batch_run,
                                                       data_dir=data_dir,
                                                       out_dir=out_dir,
                                                       target_ba_list=[i],
                                                       generate_plots=generate_plots) for i in ba_list)

        # aggregate outputs
        df = aggregate_summary(outputs)

        # output summary file
        out_summary_file = os.path.join(out_dir, 'summary.csv')

        df.to_csv(out_summary_file, index=False)

        return df

    else:
        proc = Process(batch_run=batch_run,
                       data_dir=data_dir,
                       out_dir=out_dir,
                       target_ba_list=target_ba_list,
                       generate_plots=generate_plots)

        return proc.summary_df