import os
import pkg_resources
from typing import Union

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error


def get_balancing_authority_to_model_dict():
    """Return a list of balancing authority abbreviations."""

    ba_file = pkg_resources.resource_filename("tell", "data/balancing_authority_modeled.yml")

    # read into a dictionary
    with open(ba_file, 'r') as yml:
        return yaml.load(yml, Loader=yaml.FullLoader)


def normalize_features(x_train: np.ndarray,
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

    # get the min and max of each variable in each array
    min_x_train = np.min(x_train, axis=0)
    max_x_train = np.max(x_train, axis=0)
    min_y_train = np.min(y_train, axis=0)
    max_y_train = np.max(y_train, axis=0)

    # normalize
    x_train_norm = np.divide((x_train - min_x_train), (max_x_train - min_x_train))
    x_test_norm = np.divide((x_test - min_x_train), (max_x_train - min_x_train))
    y_train_norm = np.divide((y_train - min_y_train), (max_y_train - min_y_train))

    if y_test is not None:
        y_test_norm = np.divide((y_test - min_y_train), (max_y_train - min_y_train))
    else:
        y_test_norm = None

    dict_out = {
        "min_x_train": min_x_train,
        "max_x_train": max_x_train,
        "min_y_train": min_y_train,
        "max_y_train": max_y_train,
        "x_train_norm": x_train_norm,
        "y_train_norm": y_train_norm,
        "x_test_norm": x_test_norm,
        "y_test_norm": y_test_norm
    }

    return dict_out


def denormalize_features(region: str,
                         normalized_dict: dict,
                         y_predicted_normalized: np.ndarray,
                         y_comparison: np.ndarray,
                         datetime_arr: np.ndarray) -> pd.DataFrame:
    """Function to denormlaize the predictions of the model.

    :param region:                              Indicating region / balancing authority we want to train and test on.
                                                Must match with string in CSV files.
    :type region:                               str

    :param normalized_dict:                     Dictionary output from normalization function.
    :type normalized_dict:                      dict

    :param y_predicted_normalized:              Normalized predictions over the test set.
    :type y_predicted_normalized:               np.ndarray

    :param y_comparison:                        Testing data to compare predictions to.
    :type y_comparison:                         np.ndarray

    :param datetime_arr:                        Array of datetimes corresponding to the predictions.
    :type datetime_arr:                         np.ndarray

    :return:                                    Denormalized predictions

    """

    # denormalize predicted Y
    y_p = y_predicted_normalized * (normalized_dict["max_y_train"] - normalized_dict["min_y_train"]) + normalized_dict["min_y_train"]

    # create data frame with datetime attached
    df = pd.DataFrame({"datetime": datetime_arr, "predictions": y_p, "ground_truth": np.squeeze(y_comparison)})

    # add in region field
    df["region"] = region

    return df


def pickle_model(region: str,
                 model_object: object,
                 model_name: str,
                 model_output_directory: Union[str, None]):
    """Pickle model to file using joblib.  Version of scikit-learn is included in the file name as a compatible
    version is required to reload the data safely.

    :param region:                          Indicating region / balancing authority we want to train and test on.
                                            Must match with string in CSV files.
    :type region:                           str

    :param model_object:                    scikit-learn model object.
    :type model_object:                     object

    :param model_name:                      Name of sklearn model.
    :type model_name:                       str

    :param model_output_directory:          Full path to output directory where model file will be written.
    :type model_output_directory:           str

    """

    # build output file name
    basename = f"{region}_{model_name}_scikit-learn-version-{sklearn.__version__}.joblib"
    output_file = os.path.join(model_output_directory, basename)

    # dump model to file
    joblib.dump(model_object, output_file)


def load_model(model_file: str) -> object:
    """Load pickled model from file using joblib.  Version of scikit-learn is included in the file name as a compatible
    version is required to reload the data safely.

    :param model_file:                  Full path with filename an extension to the joblib pickled model file.
    :type model_file:                   str

    :return:                            Model as an object.

    """

    # get version of scikit-learn and compare with the model from file to ensure compatibility
    sk_model_version = os.path.splitext(model_file)[0].split('-')[-1]

    # get version of scikit-learn being used during runtime
    sk_run_version = sklearn.__version__

    if sk_model_version != sk_run_version:
        msg = f"Incompatible scikit-learn version for saved model ({sk_model_version}) and current version ({sk_run_version})."
        raise AssertionError(msg)

    # load model from
    return joblib.load(model_file)


def load_predictive_models(region: str,
                           model_output_directory: Union[str, None],
                           mlp_linear_adjustment: bool):
    """Load predictive models based off of what is stored in the package or from a user provided directory.
    The scikit-learn version being used must match the one the model was generated with.

    :param region:                          Indicating region / balancing authority we want to train and test on.
                                            Must match with string in CSV files.
    :type region:                           str

    :param model_output_directory:          Full path to output directory where model file will be written.
    :type model_output_directory:           Union[str, None]

    :param mlp_linear_adjustment:           True if you want to correct the MLP model using a linear model.
    :type mlp_linear_adjustment:            Optional[bool]

    :return:                                [0] MLP model
                                            [1] linear model or None

    """

    # current scikit-learn version
    sk_version = sklearn.__version__

    # load the models from the package data if no alternate directory is passed
    if len(model_output_directory) == 0:

        # get default model file
        mlp_model_id = "multi-layer-perceptron-regressor"
        mlp_model_file = os.path.join("data", "models", f"{region}_{mlp_model_id}_scikit-learn-version-{sk_version}.joblib")
        mlp_model_path = pkg_resources.resource_filename("tell", mlp_model_file)

        if mlp_linear_adjustment:

            # get default model file
            linear_model_id = "ordinary-least-squares-linear-regression"
            linear_model_file = os.path.join("data", "models", f"{region}_{linear_model_id}_scikit-learn-version-{sk_version}.joblib")
            linear_model_path = pkg_resources.resource_filename("tell", linear_model_file)

    else:

        # get provided model file
        mlp_model_id = "multi-layer-perceptron-regressor"
        mlp_model_file = f"{region}_{mlp_model_id}_scikit-learn-version-{sk_version}.joblib"
        mlp_model_path = os.path.join(model_output_directory, mlp_model_file)

        if mlp_linear_adjustment:

            # get provided model file
            linear_model_id = "ordinary-least-squares-linear-regression"
            linear_model_file = f"{region}_{linear_model_id}*.joblib"
            linear_model_path = os.path.join(model_output_directory, linear_model_file)

    # load the mlp model
    mlp_model = load_model(model_file=mlp_model_path)

    if mlp_linear_adjustment:

        # load the linear model
        linear_model = load_model(model_file=linear_model_path)

    else:

        linear_model = None

    return mlp_model, linear_model


def pickle_normalization_dict(region: str,
                              normalization_dict: dict,
                              model_output_directory: Union[str, None]):
    """Pickle model to file using joblib.  Version of scikit-learn is included in the file name as a compatible
    version is required to reload the data safely.

    :param region:                          Indicating region / balancing authority we want to train and test on.
                                            Must match with string in CSV files.
    :type region:                           str

    :param normalization_dict:              Dictionary of normalization data
    :type normalization_dict:               dict

    :param model_output_directory:          Full path to output directory where model file will be written.
    :type model_output_directory:           str

    """

    # build output file name
    basename = f"{region}_normalization_dict.joblib"
    output_file = os.path.join(model_output_directory, basename)

    # dump dictionary to file
    joblib.dump(normalization_dict, output_file)


def load_normalization_dict(region: str,
                            model_output_directory: str) -> dict:
    """Load pickled model from file using joblib.

    :param region:                          Indicating region / balancing authority we want to train and test on.
                                            Must match with string in CSV files.
    :type region:                           str

    :param model_output_directory:          Full path to output directory where model file will be written.
    :type model_output_directory:           str

    :return:                                Normalization dictionary

    """

    basename = os.path.join(model_output_directory, f"{region}_normalization_dict.joblib")
    file = os.path.join(model_output_directory, basename)

    return joblib.load(file)


def evaluate(region: str,
             y_predicted: np.ndarray,
             y_comparison: np.ndarray) -> pd.DataFrame:
    """Evaluation of model performance using the predicted compared to the test data.

    :param region:                      Indicating region / balancing authority we want to train and test on.
                                        Must match with string in CSV files.
    :type region:                       str

    :param y_predicted:                 Predicted Y result array.
    :type y_predicted:                  np.ndarray

    :param y_comparison:                Comparison test data for Y array.
    :type y_comparison:                 np.ndarray

    :return:                            Data frame of stats.

    """

    # remove all the no data values in the comparison test data
    y_comparison = y_comparison.squeeze()
    y_comp_clean_idx = np.where(~np.isnan(y_comparison))

    y_comp = y_comparison[y_comp_clean_idx].squeeze()

    # get matching predicted data
    y_pred = y_predicted[y_comp_clean_idx]

    # absolute RMSE
    rms_abs = np.sqrt(mean_squared_error(y_pred, y_comp))

    # RMSE normalized
    rms_norm = rms_abs / np.mean(y_comp)

    # mean absolute percentage error
    mape = mean_absolute_percentage_error(y_pred, y_comp)

    # R2
    r2_val = r2_score(y_pred, y_comp)

    stats_dict = {"region": [region],
                  "RMS_ABS": [rms_abs],
                  "RMS_NORM": [rms_norm],
                  "MAPE": [mape],
                  "R2": [r2_val]}

    return pd.DataFrame(stats_dict)