import os
import glob
import pkg_resources
from typing import Optional

import holidays
import numpy as np
import pandas as pd
import yaml


class Dataset:
    """Clean and format input data for use in predictive models.

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

    :param mlp_linear_adjustment:       True if a linear model will be run and will cause the application of the
                                        sine function for hour and month fields if they are present in the data.
    :type mlp_linear_adjustment:        Optional[bool]

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

    """

    def __init__(self,
                 region: str,
                 data_dir: str,
                 **kwargs):

        self.region = region
        self.data_dir = data_dir

        # update the default settings with what the user provides
        self.settings_dict = self.update_default_settings(kwargs)

        # get argument defaults or custom settings
        self.expected_datetime_columns = self.settings_dict.get("expected_datetime_columns")
        self.data_column_rename_dict = self.settings_dict.get("data_column_rename_dict")
        self.x_variables = self.settings_dict.get("x_variables")
        self.y_variables = self.settings_dict.get("y_variables")
        self.add_dayofweek_xvars = self.settings_dict.get("add_dayofweek_xvars")
        self.mlp_linear_adjustment = self.settings_dict.get("mlp_linear_adjustment")
        self.hour_field_name = self.settings_dict.get("hour_field_name")
        self.month_field_name = self.settings_dict.get("month_field_name")
        self.day_list = self.settings_dict.get("day_list")
        self.start_time = str(self.settings_dict.get("start_time"))
        self.end_time = str(self.settings_dict.get("end_time"))
        self.split_datetime = str(self.settings_dict.get("split_datetime"))
        self.nodata_value = self.settings_dict.get("nodata_value")
        self.verbose = self.settings_dict.get("verbose")

        # populate class attributes for data
        self.df_train, self.df_test = self.generate_data()

        # break out training and testing targets and features into individual data frames
        self.X_train = self.df_train[self.x_variables].copy()
        self.X_test = self.df_test[self.x_variables].copy()
        self.Y_train = self.df_train[self.y_variables].copy()
        self.Y_test = self.df_test[self.y_variables].copy()

    @staticmethod
    def update_default_settings(kwargs) -> dict:
        """Read the default settings YAML file into a dictionary.  Update any settings passed in from a
        settings dictionary or via kwargs.

        :param kwargs:                      Keyword argument dictonary from user.
        :type kwargs:                       dict

        :return:                            A dictionary of updated default settings.

        """

        # get file path to settings YAML file stored in the package data
        settings_file = pkg_resources.resource_filename("tell", "data/mlp_settings.yml")

        # read into a dictionary
        with open(settings_file, 'r') as yml:
            default_settings_dict = yaml.load(yml, Loader=yaml.FullLoader)

        # update base on any data passed in through keyword arguments
        default_settings_dict.update(kwargs)

        return default_settings_dict

    def generate_data(self):
        """Workhorse function to clean and format input data for use in the predictive model."""

        # get the input file from the data directory matching the region name and read it into a data frame
        df = self.fetch_read_file()

        # format the input data file
        df_filtered = self.format_filter_data(df)

        # apply sine to hour and month fields if present and if using a linear model
        df_smooth = self.apply_sine_for_linear_model(df_filtered)

        # add fields for weekday, each day of the week, and holidays to the data frame; also adds "Weekday" and
        # "Holidays" as fields to the x_variables list
        if self.add_dayofweek_xvars:
            df_smooth = self.breakout_day_designation(df_smooth)

        # split the data frame into test and training data based on a datetime
        df_train_raw, df_test_raw = self.split_train_test(df_smooth)

        # clean data to drop no data records, non-feasible, and extreme values
        df_train_clean = self.clean_data(df_train_raw, drop_records=True)

        # clean data to alter no data records, non-feasible, and extreme values
        df_test_clean = self.clean_data(df_test_raw, drop_records=False)

        # extract the targets and features from the cleaned training data
        df_train_extract = self.extract_targets_features(df_train_clean)

        # extract the targets and features from the test data
        df_test_extract = self.extract_targets_features(df_test_raw)

        return df_train_extract, df_test_extract

    def fetch_read_file(self) -> pd.DataFrame:
        """Get the input file from the data directory matching the region name and read it into a pandas data frame."""

        file_pattern = os.path.join(self.data_dir, f"{self.region}_*.csv")

        # get file list from the data directory using the pattern
        file_list = glob.glob(file_pattern)

        # raise error if no files are found
        if len(file_list) == 0:
            msg = f"No data files were found for region '{self.region}' in directory '{self.data_dir}'."
            raise FileNotFoundError(msg)

        # raise error if more than one file was found
        if len(file_list) > 1:
            msg = f"More than one data files were found for region '{self.region}' in directory '{self.data_dir}'."
            raise ValueError(msg)

        # log feedback to user if desired
        if self.verbose:
            print(f"Processing file:  {file_list[0]}")

        return pd.read_csv(file_list[0])

    def format_filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format the input data file.  Filter data by user provided date range and sort in
        ascending order by the timestamp.

        :param df:               Data frame for the target region
        :type df:                pd.DataFrame

        :return:                 Formatted data frame

        """

        # rename columns to default or user desired
        df.rename(columns=self.data_column_rename_dict, inplace=True)

        # generate datetime timestamp field
        df["Datetime"] = pd.to_datetime(df[self.expected_datetime_columns])

        # filter by date range
        df = df.loc[(df["Datetime"] >= self.start_time) & (df["Datetime"] <= self.end_time)].copy()

        # sort values by timestamp
        df.sort_values(by=["Datetime"], inplace=True)

        # reset and drop index
        df.reset_index(drop=True, inplace=True)

        return df

    def apply_sine_for_linear_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the sine function to the hour and month fields for use in a linear model as predictive variables.

        :param df:               Data frame for the target region
        :type df:                pd.DataFrame


        """

        # if a linear model will be ran and an hour field is present in the data frame apply the sine function
        if self.mlp_linear_adjustment and self.hour_field_name in df.columns:
            df[self.hour_field_name] = np.sin(df[self.hour_field_name] * np.pi / 24)

            # if a linear model will be ran and an month field is present in the data frame apply the sine function
        if self.mlp_linear_adjustment and self.month_field_name in df.columns:
            df[self.month_field_name] = np.sin(df[self.month_field_name] * np.pi / 12)

        return df

    def breakout_day_designation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a field for weekday, each day of the week, and holidays to the data frame.

        Weekdays are designated as 1 for weekdays (Mon through Fri) and weekends are designated as 0 (Sat and Sun).
        Each day of the week is given its own field which has a 1 if the record is in that day and a 0 if not.
        Holidays are set to 1 to indicate a US Federal holiday and 0 if not.

        :param df:                          Data frame for the target region.
        :type df:                           pd.DataFrame

        :return:                            [0] Formatted data frame
                                            [1] List of extended x_variables

        """

        # create an array of day of the week values from the timestamp; 0 = Monday ... 6 = Sunday
        day_of_week_arr = df["Datetime"].dt.dayofweek.values

        # adjust to specify weekdays (Mon through Fri) as 1 and weekends (Sat and Sun) as 0
        df["Weekday"] = np.where(day_of_week_arr <= 4, 1, 0)

        # add a field for each day of the week and populate with 1 if the record is the day and 0 if not
        for index, i in enumerate(self.day_list):
            df[i] = np.where(day_of_week_arr == index, 1, 0)

        # build a sorted range of years in the data frame
        years_arr = np.sort(df["Datetime"].dt.year.unique())

        # identify the US holidays for the years in the data frame
        holiday_list = holidays.US(years=years_arr)

        # add a field designating whether the day is a US holiday where 1 == yes and 0 == no
        df["Holidays"] = df["Datetime"].dt.date.isin(holiday_list) * 1

        # extend the x_variables list to include the new predictive fields
        self.x_variables.extend(["Weekday", "Holidays"])

        return df

    def split_train_test(self, df: pd.DataFrame):
        """Split the data frame into test and training data based on a datetime.

        :param df:                         Input data frame for the target region.
        :type df:                          pd.DataFrame

        :return:                           [0] training data frame
                                           [1] testing data frame

        """

        # extract datetime less than or equal to the user provided split datetime as training data
        df_train = df.loc[df["Datetime"] <= self.split_datetime].copy()

        # extract datetime greater than the user provided split datetime as test data
        df_test = df.loc[df["Datetime"] > self.split_datetime].copy()

        return df_train, df_test

    def clean_data(self, df: pd.DataFrame, drop_records: bool = True) -> pd.DataFrame:
        """Clean data based on criteria for handling NoData and extreme values.

        :param df:                         Input data frame for the target region.
        :type df:                          pd.DataFrame

        :param drop_records:               If True, drop records; else, alter records
        :type drop_records:                bool

        :return:                           Processed data frame

        """

        # calculate error bounds
        mu_y = df["Demand"].mean()
        sigma_y = df["Demand"].std()
        lower_bound = (df["Demand"] <= mu_y - 5 * sigma_y)
        upper_bound = (df["Demand"] >= mu_y + 5 * sigma_y)

        if drop_records:

            # drop nodata value if so desired
            df.drop(df.index[np.where(df == self.nodata_value)[0]], inplace=True)

            # drop and records where demand is zero which is not feasible if desired
            df.drop(df.index[np.where(df["Demand"] == 0)[0]], inplace=True)

            # drop extreme value that lie outside + / - 5*sigma
            df.drop(df.index[np.where(lower_bound | upper_bound)], inplace=True)

        else:

            # alter records where demand is zero which is not feasible if desired
            df.loc[df["Demand"] == 0, "Demand"] = self.nodata_value

            # alter exterme value that lie outside + / - 5*sigma
            df.loc[(lower_bound | upper_bound), "Demand"] = self.nodata_value

        return df

    def extract_targets_features(self, df):
        """asdf"""

        # generate a list of field names to keep
        keep_fields = ["Datetime"] + self.x_variables + self.y_variables

        # extract desired fields
        return df[keep_fields]
