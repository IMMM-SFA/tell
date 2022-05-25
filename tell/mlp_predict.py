import os

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from typing import Union
from .mlp_prepare_data import DatasetPredict, DefaultSettings
from .mlp_utils import normalize_prediction_data, load_predictive_models


def predict(region: str,
            year: int,
            data_dir: str,
            datetime_field_name: str = "Time_UTC",
            save_prediction: bool = False,
            prediction_output_directory: Union[str, None] = None,
            **kwargs):
    """Generate predictions for MLP model for a target region from an input CSV file.

    :param region:                      Indicating region / balancing authority we want to train and test on.
                                        Must match with string in CSV files.
    :type region:                       str

    :param year:                        Target year to use in YYYY format.
    :type year:                         int

    :param data_dir:                    Full path to the directory that houses the input CSV files.
    :type data_dir:                     str

    :param save_prediction:             Choice to write predictions to a .csv file
    :type save_prediction:              bool

    :param prediction_output_directory: Full path to output directory where prediction files will be written.
    :type prediction_output_directory:  Union[str, None]

    :param datetime_field_name:         Name of the datetime field.
    :type datetime_field_name:          str

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

    :param seed_value:                  Seed value to reproduce randomization.
    :type seed_value:                   Optional[int]

    :param verbose:                     Choice to see logged outputs.
    :type verbose:                      bool

    :return:                            Prediction data frame

    """

    # get project level settings data
    settings = DefaultSettings(region=region,
                               data_dir=data_dir,
                               **kwargs)

    # set random seed
    np.random.seed(settings.seed_value)

    # prepare data for MLP model
    data_mlp = DatasetPredict(region=region,
                              year=year,
                              data_dir=data_dir,
                              datetime_field_name=datetime_field_name,
                              **kwargs)

    # load models and the normalization dictionary from file
    mlp_model, normalized_dict = load_predictive_models(region=region,
                                                        model_output_directory=settings.model_output_directory)

    # normalize model features and targets for the MLP model
    x_mlp_norm = normalize_prediction_data(data_arr=data_mlp.x_data,
                                           min_train_arr=normalized_dict["min_x_train"],
                                           max_train_arr=normalized_dict["max_x_train"])

    # run the MLP model with normalized data
    y_predicted_norm = mlp_model.predict(x_mlp_norm)

    # denormalize predicted data
    y_predicted = (y_predicted_norm * (normalized_dict["max_y_train"] - normalized_dict["min_y_train"]) + normalized_dict["min_y_train"]).round(2)

    # generate output data frame
    prediction_df = pd.DataFrame({"Time_UTC": data_mlp.df_data[settings.DATETIME_FIELD].values,
                                  "Load": y_predicted,
                                  "BA": region})

    # save the prediction to a .csv file:
    if save_prediction:

        # if the subdirectory for the year being processed doesn't exist then create it:
        if not os.path.exists(os.path.join(prediction_output_directory, str(year))):
            os.makedirs(os.path.join(prediction_output_directory, str(year)))

        prediction_df.to_csv(os.path.join(prediction_output_directory, str(year), f'{region}_'f'{year}_mlp_output.csv'), index=False)

    return prediction_df


def predict_batch(target_region_list: list,
                  year: int,
                  data_dir: str,
                  n_jobs: int = -1,
                  datetime_field_name: str = "Time_UTC",
                  save_prediction: bool = False,
                  prediction_output_directory: Union[str, None] = None,
                  **kwargs):
    """Generate predictions for MLP model for a target region from an input CSV file for all regions
    in input list in parallel.

    :param target_region_list:          List of names indicating region / balancing authority we want to train and test
                                        on. Must match with string in CSV files.
    :type target_region_list:           list

    :param year:                        Target year to use in YYYY format.
    :type year:                         int

    :param data_dir:                    Full path to the directory that houses the input CSV files.
    :type data_dir:                     str

    :param n_jobs:                      The maximum number of concurrently running jobs, such as the number of Python
                                        worker processes when backend=”multiprocessing” or the size of the thread-pool
                                        when backend=”threading”. If -1 all CPUs are used. If 1 is given, no parallel
                                        computing code is used at all, which is useful for debugging. For n_jobs
                                        below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs
                                        but one are used. None is a marker for ‘unset’ that will be interpreted as
                                        n_jobs=1 (sequential execution) unless the call is performed under a
                                        parallel_backend context manager that sets another value for n_jobs.
    :type n_jobs:                       int

    :param datetime_field_name:         Name of the datetime field.
    :type datetime_field_name:          str

    :param save_prediction:             Choice to write predictions to a .csv file
    :type save_prediction:              bool

    :param prediction_output_directory: Full path to output directory where prediction files will be written.
    :type prediction_output_directory:  Union[str, None]

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

    :param seed_value:                  Seed value to reproduce randomization.
    :type seed_value:                   Optional[int]

    :param verbose:                     Choice to see logged outputs.
    :type verbose:                      bool

    :return:                            Prediction data frame

    """

    # run all regions in target list in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(predict)(region=region,
                                                                       year=year,
                                                                       data_dir=data_dir,
                                                                       datetime_field_name=datetime_field_name,
                                                                       save_prediction=save_prediction,
                                                                       prediction_output_directory=prediction_output_directory,
                                                                       **kwargs) for region in target_region_list)

    # aggregate outputs
    for index, i in enumerate(results):

        if index == 0:
            prediction_df = i
        else:
            prediction_df = pd.concat([prediction_df, i])

    return prediction_df
