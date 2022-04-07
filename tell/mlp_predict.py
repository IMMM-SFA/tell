import numpy as np
import pandas as pd

from .mlp_prepare_data import DatasetPredict, DefaultSettings
from .mlp_utils import normalize_prediction_data, load_predictive_models


def predict(region: str,
            year: int,
            data_dir: str,
            datetime_field_name: str = "Time_UTC",
            **kwargs):
    """Generate predictions for MLP model for a target region from an input CSV file.

    :param region:                      Indicating region / balancing authority we want to train and test on.
                                        Must match with string in CSV files.
    :type region:                       str

    :param year:                        Target year to use in YYYY format.
    :type year:                         int

    :param data_dir:                    Full path to the directory that houses the input CSV files.
    :type data_dir:                     str

    :param datetime_field_name:         Name of the datetime field.
    :type datetime_field_name:          str

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

    :param seed_value:                  Seed value to reproduce randomization.
    :type seed_value:                   Optional[int]

    :param x_variables_linear:          Target variable list for the linear model.
    :type x_variables_linear:           Optional[list[str]]

    :param y_variables_linear:          Feature variable list for the linear model.
    :type y_variables_linear:           Optional[list[str]]

    :param verbose:                     Choice to see logged outputs.
    :type verbose:                      bool

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

    # prepare data for linear model if adjustment is desired
    if settings.mlp_linear_adjustment:
        data_linear = DatasetPredict(region,
                                     year=year,
                                     data_dir=data_dir,
                                     datetime_field_name=datetime_field_name,
                                     x_variables=settings.x_variables_linear,
                                     apply_sine_function=True,
                                     **kwargs)

        x_linear_data = data_linear.x_data

    else:
        x_linear_data = None

    # load models and the normalization dictionary from file
    mlp_model, linear_model, normalized_dict = load_predictive_models(region=region,
                                                                      model_output_directory=settings.model_output_directory,
                                                                      mlp_linear_adjustment=settings.mlp_linear_adjustment)

    # normalize model features and targets for the MLP model
    x_mlp_norm = normalize_prediction_data(data_arr=data_mlp.x_data,
                                           min_train_arr=normalized_dict["min_x_train"],
                                           max_train_arr=normalized_dict["max_x_train"])

    # run the MLP model with normalized data
    y_predicted_norm = mlp_model.predict(x_mlp_norm)

    if settings.mlp_linear_adjustment:
        y_predicted_linear = linear_model.predict(x_linear_data)

        # apply the linear adjustment to the MLP predictions
        y_predicted_norm += y_predicted_linear

    # denormalize predicted data
    y_predicted = y_predicted_norm * (normalized_dict["max_y_train"] - normalized_dict["min_y_train"]) + normalized_dict["min_y_train"]

    # generate output data frame
    prediction_df = pd.DataFrame({"datetime": data_mlp.df_data[settings.DATETIME_FIELD].values,
                                  "predictions": y_predicted,
                                  "region": region})

    return prediction_df
