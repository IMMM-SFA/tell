import os
import glob
import warnings
from typing import Union

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

from .mlp_prepare_data import Dataset, DefaultSettings


def scale_features(x_train: np.ndarray,
                   x_test: np.ndarray,
                   y_train: np.ndarray,
                   y_test: np.ndarray) -> dict:
    """Normalize the features and targets of the model.

    :param x_train:                         Training features
    :type x_train:                          np.ndarray

    :param x_test:                          Test features
    :type x_test:                           np.ndarray

    :param y_train:                         Training targets
    :type y_train:                          np.ndarray

    :param y_test:                          Training targets
    :type y_test:                           np.ndarray

    :return:                                Dictionary of scaled features

    """

    # get the mean and std of training set
    mu_x_train = np.mean(x_train)
    sigma_x_train = np.std(x_train)
    mu_y_train = np.mean(y_train)
    sigma_y_train = np.std(y_train)

    # normalize
    x_train_norm = np.divide((x_train - mu_x_train), sigma_x_train)
    x_test_norm = np.divide((x_test - mu_x_train), sigma_x_train)
    y_train_norm = np.divide((y_train - mu_y_train), sigma_y_train)

    if y_test is not None:
        y_test_norm = np.divide((y_test - mu_y_train), sigma_y_train)
    else:
        y_test_norm = None

    dict_out = {
        "mu_x_train": mu_x_train,
        "mu_y_train": mu_y_train,
        "sigma_x_train": sigma_x_train,
        "sigma_y_train": sigma_y_train,
        "x_train_norm": x_train_norm,
        "y_train_norm": y_train_norm,
        "x_test_norm": x_test_norm,
        "y_test_norm": y_test_norm,
    }

    return dict_out


def run_linear_model(x_train: np.ndarray,
                     y_train: np.ndarray,
                     x_test: np.ndarray):
    """Training and test data of a linear model. Can be used for either the main model or the residual model.

    :param x_train:                         Training features
    :type x_train:                          np.ndarray

    :param y_train:                         Training targets
    :type y_train:                          np.ndarray

    :param x_test:                          Test features
    :type x_test:                           np.ndarray

    :return:                                [0] y_p: predictions over test set
                                            [1] reg.coef_: regression coefficients of a linear model

    """

    # instantiate the linear model
    linear_mod = LR()

    # fit the model
    reg = linear_mod.fit(x_train, y_train)

    # get predictions using test features
    y_p = reg.predict(x_test)

    return y_p, reg.coef_


def run_mlp_model(x_train: np.ndarray,
                  y_train: np.ndarray,
                  x_test: np.ndarray,
                  mlp_hidden_layer_sizes: int,
                  mlp_max_iter: int,
                  mlp_validation_fraction: float,
                  mlp_linear_adjustment: bool,
                  x_linear_train: Union[np.ndarray, None] = None,
                  x_linear_test: Union[np.ndarray, None] = None) -> np.ndarray:
    """Trains the MLP model. also calls the linear residual model to adjust for population correction.

    :param x_train:                         Training features
    :type x_train:                          np.ndarray

    :param y_train:                         Training targets
    :type y_train:                          np.ndarray

    :param x_test:                          Test features
    :type x_test:                           np.ndarray

    :param mlp_hidden_layer_sizes:          The ith element represents the number of neurons in the ith hidden layer.
    :type mlp_hidden_layer_sizes:           int

    :param mlp_max_iter:                    Maximum number of iterations. The solver iterates until convergence
                                            (determined by ‘tol’) or this number of iterations. For stochastic solvers
                                            (‘sgd’, ‘adam’), note that this determines the number of epochs (how many
                                            times each data point will be used), not the number of gradient steps.
    :type mlp_max_iter:                     int

    :param mlp_validation_fraction:         The proportion of training data to set aside as validation set for early
                                            stopping. Must be between 0 and 1.
    :type mlp_validation_fraction:          float

    :param mlp_linear_adjustment:           True if setting up data for a linear model that will be run and will cause
                                            the application of the sine function for hour and month fields if they
                                            are present in the data.
    :type mlp_linear_adjustment:            bool

    :param x_linear_train:                  Training data for features from the linear model if using correction.
    :type x_linear_train:                   Union[np.ndarray, None]

    :param x_linear_test:                   Testing data for features from the linear model if using correction.
    :type x_linear_test:                    Union[np.ndarray, None]

    :return:                                y_p: np.ndarray -> predictions over test set

    """

    # instantiate the MLP model
    mlp = MLP(hidden_layer_sizes=mlp_hidden_layer_sizes,
              max_iter=mlp_max_iter,
              validation_fraction=mlp_validation_fraction)

    # fit the model to data matrix X (training features) and target Y (training targets)
    mlp.fit(x_train, y_train)

    # predict using the multi-layer perceptron model using the test features
    y_p = mlp.predict(x_test)

    # if the user desires, adjust the prediction using a linear residual model
    if mlp_linear_adjustment:

        # predict on training features
        y_tmp = mlp.predict(x_train)

        # compute the residuals in the training data
        epsilon = y_train - y_tmp

        # train the linear model to find residuals
        epsilon_e, regression_coeff = run_linear_model(x_train=x_linear_train,
                                                       y_train=epsilon,
                                                       x_test=x_linear_test)

        # apply the adjustment
        y_p += epsilon_e

    return y_p


def predict(region: str,
            data_dir: str,
            **kwargs):
    """Generate predictions for MLP model for a target region from an input CSV file.

    :param region:                      Indicating region / balancing authority we want to train and test on.
                                        Must match with string in CSV files.
    :type region:                       str

    :param data_dir:                    Full path to the directory that houses the input CSV files.
    :type data_dir:                     str

    :param mlp_hidden_layer_sizes:      The ith element represents the number of neurons in the ith hidden layer.
    :type mlp_hidden_layer_sizes:       Optional[int]

    :param mlp_max_iter:                Maximum number of iterations. The solver iterates until convergence
                                        (determined by ‘tol’) or this number of iterations. For stochastic solvers
                                        (‘sgd’, ‘adam’), note that this determines the number of epochs (how many
                                        times each data point will be used), not the number of gradient steps.
    :type mlp_max_iter:                 Optional[int]

    :param mlp_validation_fraction:     The proportion of training data to set aside as validation set for early
                                        stopping. Must be between 0 and 1.
    :type mlp_validation_fraction:      Optional[float]

    :param mlp_linear_adjustment:       True if you want to correct the MLP model using a linear model.
    :type mlp_linear_adjustment:        Optional[bool]

    :param apply_sine_function:         True if setting up data for a linear model that will be run and will cause
                                        the application of the sine function for hour and month fields if they
                                        are present in the data.
    :type apply_sine_function:          Optional[bool]

    :param data_column_rename_dict:     Dictionary for the field names present in the input CSV file (keys) to what the
                                        code expects them to be (values).
    :type data_column_rename_dict:      Optional[dict[str]]

    :param expected_datetime_columns:   Expected names of the date time columns in the input CSV file.
    :type expected_datetime_columns:    Optional[list[str]]

    :param hour_field_name:             Field name of the hour field in the input CSV file.
    :type hour_field_name:              Optional[str]

    :param month_field_name:            Field name of the month field in the input CSV file.
    :type month_field_name:             Optional[str]

    :param x_variables:                 Target variable list.
    :type x_variables:                  Optional[list[str]]

    :param add_dayofweek_xvars:         True if the user wishes to add weekday and holiday targets to the x variables.
    :type add_dayofweek_xvars:          Optional[bool]

    :param y_variables:                 Feature variable list.
    :type y_variables:                  Optional[list[str]]

    :param day_list:                    List of day abbreviations and their order.
    :type day_list:                     Optional[list[str]]

    :param start_time:                  Timestamp showing the datetime of for the run to start
                                        (e.g., 2016-01-01 00:00:00).
    :type start_time:                   Optional[str]

    :param end_time:                    Timestamp showing the datetime of for the run to end
                                        (e.g., 2019-12-31 23:00:00).
    :type end_time:                     Optional[str]

    :param split_datetime:              Timestamp showing the datetime to split the train and test data by
                                        (e.g., 2018-12-31 23:00:00).
    :type split_datetime:               Optional[str]

    :param nodata_value:                No data value in the input CSV file.
    :type nodata_value:                 Optional[int]

    :param seed_value:                  Seed value to reproduce randomization.
    :type seed_value:                   Optional[int]

    :param x_variables_linear:          Target variable list for the linear model.
    :type x_variables_linear:           Optional[list[str]]

    :param y_variables_linear:          Feature variable list for the linear model.
    :type y_variables_linear:           Optional[list[str]]

    """

    # get project level settings data
    settings = DefaultSettings(region=region,
                               data_dir=data_dir,
                               **kwargs)

    # set random seed
    np.random.seed(settings.seed_value)

    # prepare data for MLP model
    data_mlp = Dataset(region=region,
                       data_dir=data_dir,
                       **kwargs)

    # prepare data for linear model if adjustment is desired
    if settings.mlp_linear_adjustment:
        data_linear = Dataset(region,
                              data_dir,
                              x_variables=settings.x_variables_linear,
                              apply_sine_function=True,
                              **kwargs)

        x_linear_train = data_linear.x_train
        x_linear_test = data_linear.x_test

    else:
        x_linear_train = None
        x_linear_test = None

    # scale model features and targets for the MLP model
    normalized_dict = scale_features(x_train=data_mlp.x_train,
                                     x_test=data_mlp.x_test,
                                     y_train=data_mlp.y_train,
                                     y_test=data_mlp.y_test)

    # unpack normalized data needed to run the MLP model
    x_train_norm = normalized_dict.get("x_train_norm")
    y_train_norm = normalized_dict.get("y_train_norm")
    x_test_norm = normalized_dict.get("x_test_norm")

    # run the MLP model with the linear correction if desired
    y_p = run_mlp_model(x_train=x_train_norm,
                        y_train=y_train_norm.squeeze(),
                        x_test=x_test_norm,
                        mlp_hidden_layer_sizes=settings.mlp_hidden_layer_sizes,
                        mlp_max_iter=settings.mlp_max_iter,
                        mlp_validation_fraction=settings.mlp_validation_fraction,
                        mlp_linear_adjustment=settings.mlp_linear_adjustment,
                        x_linear_train=x_linear_train,
                        x_linear_test=x_linear_test)
    return y_p

