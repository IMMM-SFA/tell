"""
this file extracts data and predicts time series
"""

import numpy as np
import pandas as pd

import glob

import os

from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.neural_network import MLPRegressor as MLP

import matplotlib.pyplot as plt

from scipy.stats import pearsonr

class dataset:

    def __init__(self,
                 region,
                 csv_dir='C:/Users/mcgr323/projects/tell/CSV_Files',
                 start_time = '2016-01-01',
                 end_time = '2019-12-31',
                 split_time = '2018-12-31',
                 x_var=['Hour', 'Month', 'Temperature', 'Specific_Humidity', 'Wind_Speed',
                        'Longwave_Radiation', 'Shortwave_Radiation'],
                                                            #'Longwave_Radiation', 'Wind_Speed'
                 y_var = ['Demand'],
                 add_dayofweek=True):

        """
        :param region: str indicating region we want to train and test on. must match with str in csv
        :param csv_dir: str dir where the csvs are located
        :param start_time: str start time in YYYY-MM-DD. all data before this time as not considered
        :param end_time: str end_time in YYYY-MM-DD. all data after this time is not considered
        :param split_time: partition time in YYYY-MM-DD to split into training and test_set
        :param x_var: list-> features to consider
        :param y_var: list-> targets to consider
        """
        self.region = region
        self.csv_dir = csv_dir
        self.start_time, self.end_time = start_time, end_time
        self.split_time = split_time
        self.x_var, self.y_var = x_var, y_var

        self.add_dayofweek = add_dayofweek

        #Get x_var and y_var
        self.df, self.df_t, self.df_e, day_list = self.read_data()

        #get training and test data
        self.df_t, self.df_e, self.X_t, self.X_e, self.Y_t, self.Y_e = self.preprocess_data(day_list=day_list)

        #check pearson coeff
        self.compute_pearson()


        #get

    def read_data(self):

        """
        Function to read csvs
        :return: X, Y
        """

        #step 1-> check filename
        filename = self.get_filename()
        #step 2-> read csv
        df = pd.read_csv(filename)
        #step 3-> sort df by timeframe specified in start_time and end_time
        df = self.sort_timeframe(df=df)
        #step 4 -> add time variables (sin(hr) and weekday/weekend)

        df, day_list = self.preprocess_timedata(df=df)
        #step 4 -> split into training and test set using split_time
        df_t, df_e = self.partition_data(df=df)

        return df, df_t, df_e, day_list


    def preprocess_data(self, day_list=None):

        """
        takes the features and targets from df
        :return: df_t -> training dataframe df_e -> eval dataframe
        """

        #sort the entire df by the column headers we require
        if self.add_dayofweek == True:
            self.x_var = self.x_var + ['Weekday']
            ##this block of code indicates should we consider all days of week or just weekday vs weekend
            # if day_list is not None:
            #     self.x_var += day_list

        df_t, df_e = self.df_t[['Datetime'] + self.x_var+self.y_var], self.df_e[['Datetime'] + self.x_var+self.y_var]
        X_t, X_e = df_t[self.x_var], df_e[self.x_var]
        Y_t, Y_e = df_t[self.y_var], df_e[self.y_var]

        return df_t, df_e, X_t, X_e, Y_t, Y_e


    def get_filename(self):

        """
        function to extract filename for that
        :return filename: str filename corresponding to that region
        """
        str_to_check = self.csv_dir + self.region + '_*.csv'
        filename = glob.glob(str_to_check)[0] #[0] because there will be one element in this list


        return filename


    def sort_timeframe(self, df):

        """
        function to filter dataframe by specified timeframe self.start_time and self.end Time
        :return df_out: sorted df
        """
        df['Datetime'] = pd.to_datetime(df[['Day', 'Month', 'Year']])

        df = df[(df['Datetime'] >= self.start_time) & (df['Datetime'] <=self.end_time)]
        #dropping negative rows
        df = self.drop_neg_rows(df=df) #might need to replace with interpolation
        df = df.reset_index(drop=True)


        return df


    def drop_neg_rows(self, df):

        """
        drops df rows that have -9999 values (which is unrealistic)
        :param df:
        :return:
        """

        idx_nan = np.where(df == -9999)[0]
        df_out = df.drop(df.index[idx_nan])



        return df_out

    def partition_data(self, df):

        """
        :param df: contains entire dataset specified using start_time and end_time
        :return df_t, df_e: dfs for trining and test data respectively
        """

        df_t = df[(df['Datetime'] <= self.split_time)]
        df_e = df[(df['Datetime'] > self.split_time)]

        return df_t, df_e



    def preprocess_timedata(self, df):


        if 'Hour' in self.x_var:
            #this block of code indicates if we want to featurize a sine function or the raw input
            hr = df['Hour']
            #df['Hour'] = np.sin((hr*np.pi)/24) #24 hours is pi
            #df['Hour'] =

        if 'Month' in self.x_var:
            #this block of code indicates if we want to make the function sinosoidal
            mnth = df['Month']
            #df['Month'] = np.sin((mnth*np.pi/12)) #12 months is pi

        if self.add_dayofweek == True:
            dayofweek = df['Datetime'].dt.dayofweek.values
            #dayofweek 0: monday, 6: sunday
            weekday = np.zeros_like(dayofweek)
            weekday[dayofweek <= 4] = 1
            df['Weekday'] = weekday

            ###Create a day of week variable
            day_of_week = np.zeros((dayofweek.shape[0], 7))
            for d in range(7):
                tmp_val = np.zeros_like(dayofweek)
                tmp_val[dayofweek==d] = 1
                day_of_week[:, d] = tmp_val


            #concat day of week with df
            df_dayofweek = pd.DataFrame(day_of_week)
            #weeklist
            day_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            df_dayofweek.columns = day_list
            df = pd.concat((df, df_dayofweek), axis=1)


        return df, day_list


    def compute_pearson(self):

        """
        Computes the pearson coeff for each x in self.x_var and each y in self.y_var
        :return:
        """

        for x in self.x_var:
            for y in self.y_var:
                print('The PC between {} and {} is {}'.format(x, y, pearsonr(self.df[x].values, self.df[y].values)))




        return None



class ml_lib():

    def __init__(self, X_t, Y_t, X_e, Y_e=None, model='mlp'):

        """

        :param X_t: df -> training data,
        :param Y_t: df -> training targets
        :param X_e: df -> test data
        :param Y_e: df -> test targets
        :param model: type of model to pick
        """

        #set data
        self.X_t, self.X_e, self.Y_t, self.Y_e = X_t.values, X_e.values, Y_t.values, Y_e.values
        self.model = model

        #print(X_t.shape, X_e.shape, Y_t.shape, Y_e.shape)

        #scale features. normalized features in lowercase
        out = self.scale_features()
        self.x_t, self.x_e, self.y_t, self.y_e = out['x_t'], out['x_e'], out['y_t'], out['y_e']
        self.out = out
        #pick a model and predict
        self.y_p = self.pick_model()

        #evaluation
        self.analyze_results()
        #self.evaluation_metrics()



    def linear_model(self, X, Y, X_e):

        reg = LR().fit(X=X, y=Y)
        y_p = reg.predict(X_e)

        if self.y_e is not None:
            print(r2_score(y_p, self.y_e))


        return y_p, reg.coef_


    def svr_model(self, X, Y, X_e):

        svr = SVR()
        svr.fit(X, Y)
        y_p = svr.predict(X_e)



        if self.y_e is not None:
            print(r2_score(y_p, self.y_e))
            print(y_p)
            print(self.y_e)

        return y_p


    def mlp_model(self, X, Y, X_e):

        mlp = MLP(hidden_layer_sizes=256, max_iter=1000, validation_fraction=0.1)
        mlp.fit(X, Y)
        y_p = mlp.predict(X_e)

        if self.y_e is not None:
            print(r2_score(y_p, self.y_e.squeeze()))




        return y_p

    def gpr_model(self, X, Y, X_e):

        gpr = GPR().fit(X, Y)
        y_p = gpr.predict(X_e)

        if self.y_e is not None:
            print(r2_score(y_p, self.y_e))

        return None


    def pick_model(self):

        if self.model == 'linear':
            y_p, coeff = self.linear_model(X=self.x_t, Y=self.y_t, X_e=self.x_e)
        elif self.model == 'svr':
            y_p = self.svr_model(X=self.x_t, Y=self.y_t.squeeze(), X_e=self.x_e)
        elif self.model == 'gpr':
            y_p = self.gpr_model(X=self.x_t, Y=self.y_t.squeeze(), X_e=self.x_e)
        elif self.model == 'mlp':
            y_p = self.mlp_model(X=self.x_t, Y=self.y_t.squeeze(), X_e=self.x_e)
        else:
            y_p = None



        return y_p



    def scale_features(self):

        #get the mean and std of training set
        mu_x, sigma_x = np.mean(self.X_t), np.std(self.X_t)
        mu_y, sigma_y = np.mean(self.Y_t), np.std(self.Y_t)

        #normalize
        x_t, x_e= np.divide((self.X_t - mu_x), sigma_x), np.divide((self.X_e - mu_x), sigma_x)
        y_t = np.divide((self.Y_t - mu_y), sigma_y)

        if self.Y_e is not None:
            y_e = np.divide((self.Y_e - mu_y), sigma_y)
        else:
            y_e = None

        dict_out = {'mu_x': mu_x,
                    'mu_y': mu_y,
                    'sigma_x': sigma_x,
                    'sigma_y': sigma_y,
                    'x_t': x_t,
                    'y_t': y_t,
                    'x_e': x_e,
                    'y_e': y_e}

        return dict_out


    def unscale_targets(self, out, y_p):

        mu_y, sigma_y = out['mu_y'], out['sigma_y']
        Y_p = y_p*sigma_y + mu_y

        return Y_p


    def analyze_results(self):

        """
        Function to compute the evaluation metrics, and prepare plots
        :return:
        """
        #first revert to the absolute values
        self.Y_p = self.unscale_targets(out=self.out, y_p=self.y_p)

        #evaluate metrics
        self.evaluation_metrics()

        ##quick plot:
        time = np.arange(0, self.Y_e.shape[0])
        plt.plot(time, self.Y_e, label='Ground Truth')
        plt.plot(time, self.Y_p, label='Predictions')
        plt.legend()
        plt.show()


        return None


    def evaluation_metrics(self):

        #first the absolute root-mean-squared error
        self.rms_abs = np.sqrt(mean_squared_error(self.Y_p, self.Y_e))
        #next, the root mean squared error as a function of
        self.rms_norm = self.rms_abs/np.mean(self.Y_e)
        #R2 score
        self.r2_val = r2_score(self.y_p, self.y_e)

        print('RMS-ABS: ', self.rms_abs)
        print('RMS NORM: ', self.rms_norm)
        print('R2 value: ', self.r2_val)

        return None






class analysis:

    def __init__(self, region='PJM', fig_dir='prediction_figs'):

        """
        class to predict
        :param df: df corresponding to the evaluation period
        :param Y_e: ground truth during the evaluation period
        """
        self.region = region
        self.data = dataset(region=region)
        self.X_t, self.Y_t, self.X_e, self.Y_e = self.data.X_t, self.data.Y_t, self.data.X_e, self.data.Y_e
        self.df_e = self.data.df_e
        self.list_of_models = ['linear', 'mlp']

        #create fig_dir
        self.fig_dir = fig_dir

        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)

        #test multiple models
        self.test_multiple_models()



    def test_multiple_models(self):

        Yp_list = []
        labels = [str_name + ' predictions' for str_name in self.list_of_models]

        for m, model in enumerate(self.list_of_models):
            print('----CHECKING PREDICTIVE MODELS-----')
            print('Model: {}'.format(model))
            ml = ml_lib(X_t=self.X_t, Y_t=self.Y_t, X_e=self.X_e, Y_e=self.Y_e, model=model)
            Y_p = ml.Y_p
            self.df_e[labels[m]] = Y_p
            self.plot_reg(Y_a=self.Y_e, Y_p=Y_p, label=labels[m])


        return None


    def plot_reg(self, Y_a, Y_p, label):
        plt.rcParams.update({'font.size': 16})
        #fig = plt.Figure()
        plt.scatter(Y_a, Y_p)
        plt.plot(Y_a, Y_a)
        plt.xlabel('Actual forecast of electricity Demand (MWh)')
        plt.ylabel('Predictions of electricity demand (MWh)')
        fig_name = self.fig_dir + '/' + label + '.svg'
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.show()




        return None





if __name__ == "__main__":
    # pjm = dataset(region='PJM')
    # ml = ml_lib(X_t=pjm.X_t, X_e=pjm.X_e, Y_t=pjm.Y_t, Y_e=pjm.Y_e)
    analysis(region='TAL')