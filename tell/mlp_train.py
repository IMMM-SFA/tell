import numpy as np
import pandas as pd

from typing import Union
from joblib import Parallel, delayed
from sklearn.neural_network import MLPRegressor as MLP

from tell.mlp_prepare_data import DatasetTrain, DefaultSettings
from tell.mlp_utils import normalize_features, denormalize_features, pickle_model, evaluate, pickle_normalization_dict


def train_mlp_model(region: str,
                    x_train: np.ndarray,
                    y_train: np.ndarray,
                    x_test: np.ndarray,
                    mlp_hidden_layer_sizes: int,
                    mlp_max_iter: int,
                    mlp_validation_fraction: float,
                    save_model: bool = False,
                    model_output_directory: Union[str, None] = None) -> np.ndarray:
    """Trains the MLP model.

    :param region:                          Indicating region / balancing authority we want to train and test on.
                                            Must match with string in CSV files.
    :type region:                           str

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

    :param save_model:                      Choice to write ML models to a pickled file via joblib.
    :type save_model:                       bool

    :param model_output_directory:          Full path to output directory where model file will be written.
    :type model_output_directory:           Union[str, None]

    :return:                                y_p: np.ndarray -> predictions over test set

    """

    # instantiate the MLP model
    mlp = MLP(hidden_layer_sizes=int(mlp_hidden_layer_sizes),
              max_iter=int(mlp_max_iter),
              validation_fraction=mlp_validation_fraction)

    # fit the model to data matrix X (training features) and target Y (training targets)
    mlp.fit(x_train, y_train)

    # predict using the multi-layer perceptron model using the test features
    y_p = mlp.predict(x_test)

    # write the model to file if desired
    if save_model:
        pickle_model(region=region,
                     model_object=mlp,
                     model_name="multi-layer-perceptron-regressor",
                     model_output_directory=model_output_directory)

    return y_p


def train(region: str,
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

    :param save_model:                  Choice to write ML models to a pickled file via joblib.
    :type save_model:                   bool

    :param model_output_directory:      Full path to output directory where model file will be written.
    :type model_output_directory:       Union[str, None]

    :param verbose:                     Choice to see logged outputs.
    :type verbose:                      bool

    :return:                            [0] Predictions as a dataframe
                                        [1] Summary statistics as a dataframe

    """

    # get project level settings data
    settings = DefaultSettings(region=region,
                               data_dir=data_dir,
                               **kwargs)

    # set random seed
    np.random.seed(settings.seed_value)

    # prepare data for MLP model
    data_mlp = DatasetTrain(region=region,
                            data_dir=data_dir,
                            **kwargs)

    # scale model features and targets for the MLP model
    normalized_dict = normalize_features(x_train=data_mlp.x_train,
                                         x_test=data_mlp.x_test,
                                         y_train=data_mlp.y_train,
                                         y_test=data_mlp.y_test)

    if settings.save_model:
        pickle_normalization_dict(region=region,
                                  normalization_dict=normalized_dict,
                                  model_output_directory=settings.model_output_directory)

    # unpack normalized data needed to run the MLP model
    x_train_norm = normalized_dict.get("x_train_norm")
    y_train_norm = normalized_dict.get("y_train_norm")
    x_test_norm = normalized_dict.get("x_test_norm")

    # run the MLP model
    y_predicted_normalized = train_mlp_model(region=region,
                                             x_train=x_train_norm,
                                             y_train=y_train_norm.squeeze(),
                                             x_test=x_test_norm,
                                             mlp_hidden_layer_sizes=settings.mlp_hidden_layer_sizes,
                                             mlp_max_iter=settings.mlp_max_iter,
                                             mlp_validation_fraction=settings.mlp_validation_fraction,
                                             save_model=settings.save_model,
                                             model_output_directory=settings.model_output_directory)

    # denormalize predicted data
    prediction_df = denormalize_features(region=region,
                                         normalized_dict=normalized_dict,
                                         y_predicted_normalized=y_predicted_normalized,
                                         y_comparison=data_mlp.y_test,
                                         datetime_arr=data_mlp.df_test[settings.DATETIME_FIELD].values)

    # generate evaluation stats
    performance_df = evaluate(region=region,
                              y_predicted=prediction_df["predictions"].values,
                              y_comparison=data_mlp.y_comp)

    return prediction_df, performance_df


def train_batch(target_region_list: list,
                data_dir: str,
                n_jobs: int = -1,
                **kwargs):
    """Generate predictions for MLP model for a target region from an input CSV file.

    :param target_region_list:          List of names indicating region / balancing authority we want to train and test
                                        on. Must match with string in CSV files.
    :type target_region_list:           list

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

    :param save_model:                  Choice to write ML models to a pickled file via joblib.
    :type save_model:                   bool

    :param model_output_directory:      Full path to output directory where model file will be written.
    :type model_output_directory:       Union[str, None]

    :param verbose:                     Choice to see logged outputs.
    :type verbose:                      bool

    """

    # run all regions in target list in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(train)(region=region,
                                                                     data_dir=data_dir,
                                                                     **kwargs) for region in target_region_list)

    # aggregate outputs
    for index, i in enumerate(results):

        if index == 0:
            prediction_df = i[0]
            performance_df = i[1]
        else:
            prediction_df = pd.concat([prediction_df, i[0]])
            performance_df = pd.concat([performance_df, i[1]])

    return prediction_df, performance_df
