import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor as MLP

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tell.construct_data import Dataset


class MlLib:
    """
    Class to train and evaluate ML models
    :param X_t: df -> training data,
    :param Y_t: df -> training targets
    :param X_e: df -> test data
    :param Y_e: df -> test targets
    :param model: str type of model to pick
    :param datetime: array of dates for the
    :param_fig_names: dict containing names of all the figures we want, including timeseries,
                        seasonal prob dist, and cdf
    :param dict_res: dict containing data needed for training and avaluation of residual model

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

        return y_p, reg.coef_


    def mlp_model(self, X, Y, X_e):
        """Trains the MLP model. also calls the linear residual model to adjust for population correction

        :param X: arr -> train features
        :param Y: arr -> train targets
        :param X_e: arr -> test features

        :return: y_p: arr -> predictions over test set

        """

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
        """TODO: docstring"""

        if self.model == "linear":
            y_p, coeff = self.linear_model(X=self.x_t, Y=self.y_t, X_e=self.x_e)

        elif self.model == "mlp":
            y_p = self.mlp_model(X=self.x_t, Y=self.y_t.squeeze(), X_e=self.x_e)

        else:
            y_p = None

        return y_p

    def scale_features(self):
        """TODO: docstring"""

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

        return None

    def evaluation_metrics(self):

        """
          method to compute the evaluation metrics
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
            fig_hist = main_str + "_" + Dataset.MONTH_LIST[m] + ".svg"
            plt.tight_layout()
            plt.savefig(fig_hist)
            plt.clf()

            # print median value
            e = (df["Y_e"] - df["Y_p"]) / df["Y_e"]

        return None
