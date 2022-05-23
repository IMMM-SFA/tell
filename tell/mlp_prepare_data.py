import os
import glob
import pkg_resources
from typing import Optional

import holidays
import numpy as np
import pandas as pd
import yaml


class DefaultSettings:
    """Default settings for the MLP model. Updates any settings passed in from via kwargs from the user.

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

    :param year_field_name:             Field name of the year field in the input CSV file.
    :type year_field_name:              Optional[str]

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

    :param seed_value:                  Seed value to reproduce randomization.
    :type seed_value:                   Optional[int]

    :param save_model:                  Choice to write ML models to a pickled file via joblib.
    :type save_model:                   bool

    :param model_output_directory:      Full path to output directory where model file will be written.
    :type model_output_directory:       Union[str, None]

    :param save_prediction:             Choice to write predictions to a .csv file
    :type save_prediction:              bool

    :param prediction_output_directory: Full path to output directory where prediction files will be written.
    :type prediction_output_directory:  Union[str, None]

    :param verbose:                     Choice to see logged outputs.
    :type verbose:                      bool

    """

    # internally generated field names
    DATETIME_FIELD = "Datetime"
    WEEKDAY_FIELD = "Weekday"
    HOLIDAY_FIELD = "Holidays"

    # default no data value
    NODATA_VALUE = np.nan

    def __init__(self,
                 region: str,
                 data_dir: str,
                 **kwargs):

        self.region = region
        self.data_dir = data_dir

        # update the default settings with what the user provides
        self.settings_dict = self.update_default_settings(kwargs)

        # get argument defaults or custom settings
        self.mlp_hidden_layer_sizes = int(self.settings_dict.get("mlp_hidden_layer_sizes"))
        self.mlp_max_iter = int(self.settings_dict.get("mlp_max_iter"))
        self.mlp_validation_fraction = self.settings_dict.get("mlp_validation_fraction")
        self.expected_datetime_columns = self.settings_dict.get("expected_datetime_columns")
        self.data_column_rename_dict = self.settings_dict.get("data_column_rename_dict")
        self.x_variables = self.settings_dict.get("x_variables")
        self.y_variables = self.settings_dict.get("y_variables")
        self.add_dayofweek_xvars = self.settings_dict.get("add_dayofweek_xvars")
        self.hour_field_name = self.settings_dict.get("hour_field_name")
        self.month_field_name = self.settings_dict.get("month_field_name")
        self.year_field_name = self.settings_dict.get("year_field_name")
        self.day_list = self.settings_dict.get("day_list")
        self.start_time = str(self.settings_dict.get("start_time"))
        self.end_time = str(self.settings_dict.get("end_time"))
        self.split_datetime = str(self.settings_dict.get("split_datetime"))
        self.nodata_value = self.NODATA_VALUE
        self.seed_value = self.settings_dict.get("seed_value")
        self.save_model = self.settings_dict.get("save_model")
        self.model_output_directory = self.settings_dict.get("model_output_directory")
        self.save_prediction = self.settings_dict.get("save_prediction")
        self.prediction_output_directory = self.settings_dict.get("prediction_output_directory")
        self.verbose = self.settings_dict.get("verbose")

        # set to default package data if not provided
        if self.model_output_directory == "Default":
            self.model_output_directory = pkg_resources.resource_filename("tell", "data/models")

        # update hyperparameter values from defaults if the user does not provide them
        self.update_hyperparameters()

    def update_hyperparameters(self):
        """Update hyperparameter values from defaults if the user does not provide them."""

        # read in default hyperparameters for the target region
        hyperparams_file = pkg_resources.resource_filename("tell", "data/hyperparameters.csv")

        # read into data frame
        hdf = pd.read_csv(hyperparams_file)

        # if region is in preexisting hyperparameters
        if self.region in hdf["region"].unique():

            # query out target region
            hidden_layer_sizes = hdf.loc[hdf["region"] == self.region]["hidden_layer_sizes"].values[0]
            max_iter = hdf.loc[hdf["region"] == self.region]["max_iter"].values[0]
            validation_fraction = hdf.loc[hdf["region"] == self.region]["validation_fraction"].values[0]

            # update values for hyperparameters if user does not provide
            if self.mlp_hidden_layer_sizes == 447:
                self.mlp_hidden_layer_sizes = hidden_layer_sizes

            if self.mlp_max_iter == 269:
                self.mlp_max_iter = max_iter

            if self.mlp_validation_fraction == 0.2:
                self.mlp_validation_fraction = validation_fraction

        # otherwise use default
        else:
            if self.verbose:
                print(f"No exiting hyperparameters found for region:  '{self.region}'.  Assigning defaults.")

        if self.verbose:
            print(f"Using the following hyperparameter values for '{self.region}':")
            print(f"hidden_layer_sizes: {self.mlp_hidden_layer_sizes}")
            print(f"max_iter: {self.mlp_max_iter}")
            print(f"validation_fraction: {self.mlp_validation_fraction}")

    @staticmethod
    def update_default_settings(kwargs) -> dict:
        """Read the default settings YAML file into a dictionary.  Updates any settings passed in from via kwargs
        from the user.

        :param kwargs:                      Keyword argument dictionary from user.
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


class DatasetTrain(DefaultSettings):
    """Clean and format input data for use in training predictive models.

    :param region:                      Indicating region / balancing authority we want to train and test on.
                                        Must match with string in CSV files.
    :type region:                       str

    :param data_dir:                    Full path to the directory that houses the input CSV files.
    :type data_dir:                     str

    """

    def __init__(self,
                 region: str,
                 data_dir: str,
                 **kwargs):

        self.region = region
        self.data_dir = data_dir

        # get the parent class attributes and methods
        super().__init__(region=region,
                         data_dir=data_dir,
                         **kwargs)

        # populate class attributes for data
        self.df_train, self.df_test, self.df_test_comp = self.generate_data()

        # break out training and testing targets and features into individual data frames
        self.x_train = self.df_train[self.x_variables].values
        self.x_test = self.df_test[self.x_variables].values
        self.y_train = self.df_train[self.y_variables].values
        self.y_test = self.df_test[self.y_variables].values
        self.y_comp = self.df_test_comp[self.y_variables].values

        # reset index for test data
        self.df_test.reset_index(drop=True, inplace=True)

    def generate_data(self):
        """Workhorse function to clean and format input data for use in the predictive model."""

        # get the input file from the data directory matching the region name and read it into a data frame
        df = self.fetch_read_file()

        # format the input data file
        df_filtered = self.format_filter_data(df)

        # add fields for weekday, each day of the week, and holidays to the data frame; also adds "Weekday" and
        # "Holidays" as fields to the x_variables list
        if self.add_dayofweek_xvars:
            df_filtered = self.breakout_day_designation(df_filtered)

        # split the data frame into test and training data based on a datetime
        df_train_raw, df_test_raw = self.split_train_test(df_filtered)

        # clean data to drop no data records, non-feasible, and extreme values
        df_train_clean = self.clean_data(df_train_raw, drop_records=True)

        # clean data to alter no data records, non-feasible, and extreme values
        df_test_clean = self.clean_data(df_test_raw, drop_records=False)

        # extract the targets and features from the cleaned training data
        df_train_extract_clean = self.extract_targets_features(df_train_clean)

        # extract the targets and features from the test data
        df_test_extract_raw = self.extract_targets_features(df_test_raw)

        # extract the targets and features from the cleaned test data
        df_test_extract_clean = self.extract_targets_features(df_test_clean)

        return df_train_extract_clean, df_test_extract_raw, df_test_extract_clean

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
        df[self.DATETIME_FIELD] = pd.to_datetime(df[self.expected_datetime_columns])

        # filter by date range
        df = df.loc[(df[self.DATETIME_FIELD] >= self.start_time) & (df[self.DATETIME_FIELD] <= self.end_time)].copy()

        # sort values by timestamp
        df.sort_values(by=[self.DATETIME_FIELD], inplace=True)

        # reset and drop index
        df.reset_index(drop=True, inplace=True)

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
        day_of_week_arr = df[self.DATETIME_FIELD].dt.dayofweek.values

        # adjust to specify weekdays (Mon through Fri) as 1 and weekends (Sat and Sun) as 0
        df[self.WEEKDAY_FIELD] = np.where(day_of_week_arr <= 4, 1, 0)

        # add a field for each day of the week and populate with 1 if the record is the day and 0 if not
        for index, i in enumerate(self.day_list):
            df[i] = np.where(day_of_week_arr == index, 1, 0)

        # build a sorted range of years in the data frame
        years_arr = np.sort(df[self.DATETIME_FIELD].dt.year.unique())

        # identify the US holidays for the years in the data frame
        holiday_list = holidays.US(years=years_arr)

        # add a field designating whether the day is a US holiday where 1 == yes and 0 == no
        df[self.HOLIDAY_FIELD] = df[self.DATETIME_FIELD].dt.date.isin(holiday_list) * 1

        # extend the x_variables list to include the new predictive fields
        self.x_variables.extend([self.WEEKDAY_FIELD, self.HOLIDAY_FIELD])

        return df

    def split_train_test(self, df: pd.DataFrame):
        """Split the data frame into test and training data based on a datetime.

        :param df:                         Input data frame for the target region.
        :type df:                          pd.DataFrame

        :return:                           [0] training data frame
                                           [1] testing data frame

        """

        # extract datetime less than or equal to the user provided split datetime as training data
        df_train = df.loc[df[self.DATETIME_FIELD] <= self.split_datetime].copy()

        # extract datetime greater than the user provided split datetime as test data
        df_test = df.loc[df[self.DATETIME_FIELD] > self.split_datetime].copy()

        return df_train, df_test

    def iqr_outlier_detection(self,
                              df: pd.DataFrame,
                              drop_records: bool = True,
                              scale_constant: float = 3.5) -> pd.DataFrame:
        """Outlier detection using interquartile range (IQR).  Drops or adjusts outliers that are outside
        the acceptable range, NaN, or at or below 0.

        :param df:                          Input data frame for the target region.
        :type df:                           pd.DataFrame

        :param drop_records:                If True, drop records; else, alter records
        :type drop_records:                 bool

        :param scale_constant:              Scale factor controlling the sensitivity of the IQR to outliers
        :type scale_constant:               float

        :return:                            Processed data frame

        """

        # prediction variable name
        feature_field = self.y_variables[0]

        # drop nan rows and above 0 rows to calculate IQR
        dfx = df.loc[(~df[feature_field].isnull()) & (df[feature_field] > 0)].copy()

        # extract an array of values for the target field
        arr = dfx[feature_field].values

        # sort values
        arr_sort = np.sort(arr)

        # get first and third quartile
        q1, q3 = np.percentile(arr_sort, [25, 75])

        # calc IQR
        iqr = q3 - q1

        # calculate upper and lower bounds
        lower_bound = q1 - (scale_constant * iqr)
        upper_bound = q3 + (scale_constant * iqr)

        if self.verbose:
            print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")
            print(f"Lower: {lower_bound}, Upper: {upper_bound}")

        if drop_records:
            return df.loc[(df[feature_field] >= lower_bound) &
                          (df[feature_field] <= upper_bound) &
                          (~df[feature_field].isnull()) &
                          (df[feature_field] > 0)].copy()

        else:
            df[feature_field] = np.where((df[feature_field] <= lower_bound) |
                                         (df[feature_field] >= upper_bound) |
                                         (df[feature_field].isnull()) |
                                         (df[feature_field] <= 0),
                                         self.nodata_value,
                                         df[feature_field])

            return df

    def clean_data(self,
                   df: pd.DataFrame,
                   drop_records: bool = True,
                   iqr_scale_constant: float = 3.5) -> pd.DataFrame:
        """Clean data based on criteria for handling NoData and extreme values.

        :param df:                         Input data frame for the target region.
        :type df:                          pd.DataFrame

        :param drop_records:               If True, drop records; else, alter records
        :type drop_records:                bool

        :param iqr_scale_constant:         Scale factor controlling the sensitivity of the IQR to outliers
        :type iqr_scale_constant:          float

        :return:                           Processed data frame

        """

        # generate a copy of the input data frame
        dfx = df.copy()

        # number of rows in the data frame
        pre_drop_n = df.shape[0]

        if drop_records:

            # drop any outliers
            df = self.iqr_outlier_detection(df=dfx, drop_records=drop_records, scale_constant=iqr_scale_constant)

            if self.verbose:
                print(f"Dropped {pre_drop_n - dfx.shape[0]} row(s)")

        else:

            df = self.iqr_outlier_detection(df=dfx, drop_records=drop_records, scale_constant=iqr_scale_constant)

        return df

    def extract_targets_features(self, df) -> pd.DataFrame:
        """Keep datetime, target, and feature fields.

        :param df:                         Input data frame for the target region.
        :type df:                          pd.DataFrame

        """

        # generate a list of field names to keep
        keep_fields = [self.DATETIME_FIELD] + self.x_variables + self.y_variables

        # extract desired fields
        return df[keep_fields]


class DatasetPredict(DefaultSettings):
    """Clean and format input weather data for use in predictive models.

    :param region:                      Indicating region / balancing authority we want to train and test on.
                                        Must match with string in CSV files.
    :type region:                       str

    :param year:                        Target year to use in YYYY format.
    :type year:                         int

    :param data_dir:                    Full path to the directory that houses the input CSV files.
    :type data_dir:                     str

    :param datetime_field_name:         Name of the datetime field.
    :type datetime_field_name:          str

    """

    def __init__(self,
                 region: str,
                 year: int,
                 data_dir: str,
                 datetime_field_name: str = "Time_UTC",
                 **kwargs):

        self.region = region
        self.year = year
        self.data_dir = data_dir
        self.datetime_field_name = datetime_field_name

        # get the parent class attributes and methods
        super().__init__(region=region,
                         data_dir=data_dir,
                         **kwargs)

        # populate class attributes for data
        self.df_data = self.generate_data()

        # break out training and testing targets and features into individual data frames
        self.x_data = self.df_data[self.x_variables].values

        # reset index for test data
        self.df_data.reset_index(drop=True, inplace=True)

    def generate_data(self):
        """Workhorse function to clean and format input data for use in the predictive model."""

        # get the input file from the data directory matching the region name and read it into a data frame
        df = self.fetch_read_file()

        # format the input data file
        df_filtered = self.format_filter_data(df)

        # add fields for weekday, each day of the week, and holidays to the data frame; also adds "Weekday" and
        # "Holidays" as fields to the x_variables list
        if self.add_dayofweek_xvars:
            df_filtered = self.breakout_day_designation(df_filtered)

        # clean data to alter no data records, non-feasible, and extreme values
        df_test_clean = self.clean_data(df_filtered, drop_records=False)

        # extract the targets and features from the cleaned test data
        df_test_extract_clean = self.extract_targets_features(df_test_clean)

        return df_test_extract_clean

    def fetch_read_file(self) -> pd.DataFrame:
        """Get the input file from the data directory matching the region name and year
        and read it into a pandas data frame.

        """

        file_pattern = os.path.join(self.data_dir, f"{self.region}_*_{self.year}.csv")

        # get file list from the data directory using the pattern
        file_list = glob.glob(file_pattern)

        # raise error if no files are found
        if len(file_list) == 0:
            msg = f"No data files were found for region '{self.region}' and year '{self.year}' in directory '{self.data_dir}'."
            raise FileNotFoundError(msg)

        # raise error if more than one file was found
        if len(file_list) > 1:
            msg = f"More than one data files were found for region '{self.region}' and year '{self.year}' in directory '{self.data_dir}'."
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
        df[self.DATETIME_FIELD] = pd.to_datetime(df[self.datetime_field_name])

        # break out date time fields
        df[self.year_field_name] = df[self.DATETIME_FIELD].dt.year
        df[self.month_field_name] = df[self.DATETIME_FIELD].dt.month
        df[self.hour_field_name] = df[self.DATETIME_FIELD].dt.hour

        # sort values by timestamp
        df.sort_values(by=[self.DATETIME_FIELD], inplace=True)

        # reset and drop index
        df.reset_index(drop=True, inplace=True)

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
        day_of_week_arr = df[self.DATETIME_FIELD].dt.dayofweek.values

        # adjust to specify weekdays (Mon through Fri) as 1 and weekends (Sat and Sun) as 0
        df[self.WEEKDAY_FIELD] = np.where(day_of_week_arr <= 4, 1, 0)

        # add a field for each day of the week and populate with 1 if the record is the day and 0 if not
        for index, i in enumerate(self.day_list):
            df[i] = np.where(day_of_week_arr == index, 1, 0)

        # build a sorted range of years in the data frame
        years_arr = np.sort(df[self.DATETIME_FIELD].dt.year.unique())

        # identify the US holidays for the years in the data frame
        holiday_list = holidays.US(years=years_arr)

        # add a field designating whether the day is a US holiday where 1 == yes and 0 == no
        df[self.HOLIDAY_FIELD] = df[self.DATETIME_FIELD].dt.date.isin(holiday_list) * 1

        # extend the x_variables list to include the new predictive fields
        self.x_variables.extend([self.WEEKDAY_FIELD, self.HOLIDAY_FIELD])

        return df

    def clean_data(self, df: pd.DataFrame, drop_records: bool = True) -> pd.DataFrame:
        """Clean data based on criteria for handling NoData and extreme values.

        :param df:                         Input data frame for the target region.
        :type df:                          pd.DataFrame

        :param drop_records:               If True, drop records; else, alter records
        :type drop_records:                bool

        :return:                           Processed data frame

        """

        if drop_records:

            # drop records containing any native np.nan
            df.drop(df.index[np.where(np.isnan(df))[0]], inplace=True)

        return df

    def extract_targets_features(self, df) -> pd.DataFrame:
        """Keep datetime, target, and feature fields.

        :param df:                         Input data frame for the target region.
        :type df:                          pd.DataFrame

        """

        # generate a list of field names to keep
        keep_fields = [self.DATETIME_FIELD] + self.x_variables

        # extract desired fields
        return df[keep_fields]
