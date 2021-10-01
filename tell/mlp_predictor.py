"""
this file extracts data and predicts time series
"""

import os
import glob

import numpy as np
import pandas as pd
import holidays

from joblib.parallel import Parallel, delayed

from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor as MLP
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tell.train import MlLib
from tell.construct_data import Dataset


class Hyperparameters:
    """class for hyperparameter search
    only used when we need to optimize hyperpaaremters for a specific balancing authority


    :param model_str: str -> indicating model type. currently only mlp is supported
    :param X: features in training set
    :param Y: targets in training set

    """

    def __init__(self, model_str, X, Y):

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
        """Function to set hyperparameters for MLP
        currently only MLP with 1 hidden layer is supported. the only hyperparameter is size of hidden layer

        :return params: dict containing search space for hyperparameters

        """
        params = dict() #instantiate params

        # define hyperparameters search space. currently only supporting MLP with 1 hidden l
        params["hidden_layer_sizes"] = [16, 32, 64, 128, 256, 712, 1028]

        return params

class Analysis:
    """Train and evaluate each individual BAs. Generates output CSV files under directory outputs
    Trains the "residual model" as well for population correction.

    :param df:                                  Data frame corresponding to the evaluation period
    :param Y_e:                                 Ground truth during the evaluation period

    """

    def __init__(self, data_dir, out_dir, region="PJM", generate_plots=True, model_list=["mlp"]):

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
        self.list_of_models = model_list

        self.out_dir = out_dir

        # create output dir if it not already created
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # test multiple models. currently only testing for 'mlp' model
        self.test_multiple_models()

    def test_multiple_models(self):
        """TODO: add docstring"""

        # get labels for predictions based on type of model
        labels = [f"{str_name}_predictions" for str_name in self.list_of_models]

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
        """ Method for regression Plots

        :param Y_a: array containing ground truth
        :param Y_p:
        :param label:

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
        """This function is used to write output

        Y_p:                                predictions (np array)
        model:                              str to label which model it is

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
    """Run multiple BA

    :param data_dir:                Full path to the directory containing the target
                                    CSV files
    :type data_dir:                 str

    :param out_dir:                 Full path to the directory where the outputs are to be written
    :type out_dir:                  str

    :param target_ba_list:          A list of BA names to run. If None, a list of all BA's found in the file names
                                    of CSV file files in the data directory will be used.
    :type target_ba_list:           list

    :param generate_plots:          Choice to generate and save plots
    :type generate_plots:           bool

    :param write_summary:           Choice to write summary output file
    :type write_summary:            bool

    """

    def __init__(self, data_dir=None, out_dir=None, target_ba_list=None, generate_plots=True,
                 write_summary=True):

        self.data_dir = data_dir
        self.out_dir = out_dir
        self.target_ba_list = target_ba_list
        self.generate_plots = generate_plots
        self.write_summary = write_summary

        # output summary file
        self.out_summary_file = os.path.join(self.out_dir, 'summary.csv')

        # checking dir to get list of CSV files
        self.pat_to_check = os.path.join(self.data_dir, '*.csv')

        # loop over all BAs. generates summary.csv to show accuracy of all BAs
        self.summary_df = self.gen_results()  # steo ii: gen_results

    def gen_results(self):
        """Writes all outputs to csvs + a summary file with the evaluation metrics.

        :return:

        """

        ba_out, r2, mape = [], [], []

        for ba_name in self.target_ba_list:

            print(f"Processing BA: {ba_name}")

            try:
                # perform analysis for each BA, keep track of all BAs and corresponding accuracy metrics
                ba = Analysis(region=ba_name,
                              out_dir=self.out_dir,
                              data_dir=self.data_dir,
                              generate_plots=self.generate_plots)

                ba_out.append(ba_name), r2.append(ba.R2), mape.append(ba.MAPE)

            except ValueError:
                print(ba_name)
                continue

        # write to file
        out = {"BA": ba_out, "R2": r2, "MAPE": mape}
        df = pd.DataFrame(out)

        if self.write_summary:
            df.to_csv(self.out_summary_file, index=False, mode='a')

        return df


def aggregate_summary(output_list):
    """Aggregates summary output for all BAs

    :return:Summary of all R2 and MAPE for all BAs
    """

    output_dfs = [i.summary_df for i in output_list]

    return pd.concat(output_dfs)


def list_ba(data_directory):
    """Sets list of BAs for training and evaluation fro target BA in predict.

    :param data_directory:                  Directory containing the input CSV files
    :type data_directory:                   str

    :return:                                List of BAs to process

    """

    # generate a list of csv files in the input directory
    list_of_files = [os.path.basename(i) for i in os.listdir(data_directory) if os.path.splitext(i)[-1] == '.csv']

    return [i.split('_')[0] for i in list_of_files]


def predict(data_dir, out_dir, target_ba_list=None, generate_plots=True, run_parallel=True, n_jobs=-1,
            write_summary=True):
    """Convenience wrapper for the Process class which runs predictive models for each BA input CSV in the input
    directory and creates a summary and comparative figures of R2 and MAPE per BA.

    :param data_dir:                Full path to the directory containing the target
                                    CSV files
    :type data_dir:                 str

    :param out_dir:                 Full path to the directory where the outputs are to be written
    :type out_dir:                  str

    :param target_ba_list:          A list of BA names to run.  If None, a list of all BA's found in the file names
                                        of CSV file files in the data directory will be used.
    :type target_ba_list:           list

    :param generate_plots:          Choice to generate and save plots
    :type generate_plots:           bool

    :param run_parallel:            Choose to run BA's in parallel
    :type run_parallel:             bool

    :param n_jobs:                  Set number of CPUs
    :type n_jobs:                   int

    :param write_summary:           Choice to write summary output file
    :type write_summary:            bool

    :return:                        Data frame of BA, R2, MAPE statistics

    """

    # check existence of data directory
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"`data_dir` value '{data_dir}' not a valid directory.")

    # check if user wants to find all BA files in the or pass their own
    if target_ba_list is None:

        # generate a list of files to process
        ba_list = list_ba(data_directory=data_dir)

    # if input is a list and it has content
    elif type(target_ba_list) == list and len(target_ba_list) > 0:

        ba_list = target_ba_list

    elif type(target_ba_list) == list and len(target_ba_list) == 0:
        raise ValueError(f"`target_ba_list` is an empty list.  Pass `None` if you wish to use all BAs in the data dir.")

    else:
        raise ValueError(f"`target_ba_list` is not in a recognized format of type `list` or `None`")

    if run_parallel:

        outputs = Parallel(n_jobs=n_jobs)(delayed(Process)(data_dir=data_dir,
                                                           out_dir=out_dir,
                                                           target_ba_list=[i],
                                                           generate_plots=generate_plots,
                                                           write_summary=write_summary) for i in ba_list)

        # aggregate outputs
        df = aggregate_summary(outputs)

        # # output summary file
        # out_summary_file = os.path.join(out_dir, 'summary.csv')
        #
        # df.to_csv(out_summary_file, index=False)

        return df

    else:
        proc = Process(data_dir=data_dir,
                       out_dir=out_dir,
                       target_ba_list=target_ba_list,
                       generate_plots=generate_plots,
                       write_summary=write_summary)

        return proc.summary_df
