import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from typing import Union
from joblib import Parallel, delayed
from sklearn.neural_network import MLPRegressor as MLP

from tell.mlp_prepare_data import DatasetTrain, DefaultSettings
from tell.mlp_utils import normalize_features, denormalize_features, pickle_model, evaluate, pickle_normalization_dict

from tell.model_rnn import Seq2Seq

from fastprogress import master_bar, progress_bar

class Trainer:

    def __init__(
            self,
            model,
            data_dir: str,
            region: str,
            seq_len: int = 24,
            val_split: int = 0.67,
            epochs = 1000

    ):

        self.model = model
        self.data_dir = data_dir
        self.region = region
        self.seq_len = seq_len
        self.val_split = val_split
        self.epochs = epochs

        #get the device
        self.device = self._get_device()

        #Get the normalized data
        self.norm_data, self.n_features, self.n_output = self.prepare_data()

        #get model and training params
        self.model = self._get_model()
        self.optimizer, self.loss = self._get_training_params()


    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _init_optimizer(self):
        """
        Method to set up optimizer and loss_function
        :return:
        """
        optimizer = optim.Adam(self.model.parameters())
        loss = nn.MSELoss()
        return optimizer, loss

    def _get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

    def prepare_data(self, **kwargs):

        # get project level settings data
        settings = DefaultSettings(region=self.region,
                                   data_dir=self.data_dir,
                                   **kwargs)


        # prepare data for MLP model
        data = DatasetTrain(region=self.region,
                                data_dir=self.data_dir,
                                **kwargs)

        normalized_dict = normalize_features(x_train=data.x_train,
                                             x_test=data.x_test,
                                             y_train=data.y_train,
                                             y_test=data.y_test)

        x_train, y_train = self.seq_dataloader(normalized_dict['x_train_norm'], normalized_dict['y_train_norm'])
        x_test, y_test = self.seq_dataloader(normalized_dict['x_test_norm'], normalized_dict['y_test_norm'])

        #Get the features and the output
        n_features = normalized_dict['x_train_norm'].shape[1]
        n_output = normalized_dict['y_train_norm'].shape[1]

        #get the validation and updated training set
        x_train, y_train, x_val, y_val = self._split_validation(x_train, y_train)

        #TO DO: Package everything to a torch model
        x_train = Variable(torch.Tensor(x_train))
        y_train = Variable(torch.Tensor(y_train))
        x_val = Variable(torch.Tensor(x_val))
        y_val = Variable(torch.Tensor(y_val))


        #package all the terms into normalized dict
        norm_data = {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test
        }

        return norm_data, n_features, n_output

    def seq_dataloader(self, x, y):
        """
        Method to make each one
        :param x: (np.array) -
        :return:
        """

        assert x.shape[0] == y.shape[0]
        assert x.shape[0] % self.seq_len == 0

        #get number of batches
        N = int(x.shape[0]/self.seq_len)

        x_out = []
        y_out = []

        for n in range(N):
            _x = x[n*self.seq_len:(n+1)*self.seq_len, :]
            _y = y[n*self.seq_len:(n+1)*self.seq_len, :]

            x_out.append(_x)
            y_out.append(_y)


        return x_out, y_out


    def _get_model(self):

        model = Seq2Seq(
            n_features=self.n_features,
            seq_len=self.seq_len,
            n_output=self.n_output
        )

        return model


    def _get_training_params(self):

        optimizer = torch.optim.Adam(self.model.parameters())
        loss = torch.nn.MSELoss().to(self.device)

        return optimizer, loss

    def _split_validation(self, x_train, y_train):
        """
        Method to split training data by traimn
        :return:
        """

        x_t, y_t = x_train.copy(), y_train.copy()

        n_val = int(self.val_split*len(x_t))

        x_t = x_train[:n_val]
        y_t = y_train[:n_val]

        x_v = x_train[n_val:]
        y_v = y_train[n_val:]

        return x_t, y_t, x_v, y_v


    def train(self):
        """
        Method
        :return:
        """
        x_train = self.norm_data["x_train"]
        y_train = self.norm_data["y_train"]
        x_val = self.norm_data["x_val"]
        y_val = self.norm_data["y_val"]
        x_test = self.norm_data["x_test"]
        y_test = self.norm_data["y_test"]

        #initialize history
        history = dict(train=[], val=[])

        best_loss = 10000.0

        mb = master_bar(range(1, self.epochs + 1))


        for epoch in mb:

            model = self.model.train()
            train_losses = []

            print(model)

            for i in progress_bar(range(x_train.size()[0]), parent=mb):
                _x = x_train[i, :, :].to(self.device)
                _y_gt = y_train[i, :, :].to(self.device)

                self.optimizer.zero_grad()

                _y_p = model(_x)

                # print(f"y_p: {_y_p.shape}")
                # print(_y_p.squeeze())
                # print(f"y_gt: {_y_gt.shape}")
                # print(_y_gt.squeeze())

                loss = self.loss(_y_p, _y_gt)
                print(f"loss: {loss}")
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                train_losses.append(loss.item())

            #go to validation mode
            val_losses = []
            model = self.model.eval()

            with torch.no_grad():
                for i in progress_bar(range(x_val.size()[0]), parent=mb):
                    _x_v = x_val[i, :, :].to(self.device)
                    _y_v_gt = y_val[i, :, :].to(self.device)

                    _y_v_pred = self.model(_x_v)

                    vloss = self.loss(_y_v_pred, _y_v_gt)
                    val_losses.append(vloss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')
                print("saved best model epoch:", epoch, "val loss is:", val_loss)

            print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

        return model.eval(), history



if __name__=="__main__":
    args = {
        "model": None,
        "region": "AZPS",
        "data_dir": "/qfs/projects/optimas/tell_data/data/data/composite_projections"
    }

    RNNTrainer = Trainer(**args)
    RNNTrainer.train()




