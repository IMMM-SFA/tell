import os
import re
import random

import torch
import torch.nn as nn
import torch.optim as optim

class RNN:

    def __init__(
            self,
            params: dict = None
    ):

        self.params = params

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, param_value):
        if param_value == None:
            self._params  = {
                "hidden_nodes": 32
            }
        else:
            self._params = param_value



class Seq2Seq(nn.Module):
    """
    Method to build the sequence to sequence model
    Replicating the implementation in https://www.kaggle.com/code/omershect/learning-pytorch-seq2seq-with-m5-data-set
    """

    def __init__(
            self,
            n_features: int,
            seq_len: int = 24,
            device: str = 'cuda',
            embedding_dim: int = 16,
            n_layers: int = 1,
            dropout_rate: float = 0.2,
            n_output: int = 1,
            activation: str = "GRU"
    ):

        super().__init__()

        self.device = device


        #Initialize encoder
        self.encoder = Encoder(
            seq_len=seq_len,
            n_features=n_features,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            rnn_activation=activation,
            dropout_rate=dropout_rate
        ).to(self.device)


        #initialize decoder
        self.decoder = Decoder(
            n_features=n_features,
            seq_len=seq_len,
            embedding_dim=embedding_dim,
            n_output=n_output,
            n_layers=n_layers,
            rnn_activation=activation,
            dropout_rate=dropout_rate,
            device=self.device

        ).to(self.device)

        #define optimizer and loss
        self.optimizer = torch.optim.Adam(self.parameters())
        # The best loss function to use depends on the problem.
        # We will see a different loss function later for probabilistic
        # forecasting
        self.loss_function = nn.MSELoss()

    def forward(self, X):

        #get encoder outputs
        Y_enc, hidden = self.encoder(X)

        #get the decoder outputs
        print("Troubleshoot")
        X_t = X[-1, :]
        X_t = X_t[None, :]
        Y_p = self.decoder(X_t, Y_enc, hidden)

        return Y_p

    def compute_loss(self, Y_p, Y):
        return self.loss_function(Y, Y_p)

    def optimize(self, Y_p, Y):
        self.optimizer.zero_grad()
        loss = self.compute_loss(Y_p, Y)
        loss.backward()
        self.optimizer.step()


class Encoder(nn.Module):
    """
    Class to build the encoder.
    Taken from https://www.kaggle.com/code/omershect/learning-pytorch-seq2seq-with-m5-data-set
    """

    def __init__(
            self,
            seq_len: int,
            n_features: int,
            embedding_dim: int = 16,
            n_layers: int = 1,
            rnn_activation: str = "GRU",
            dropout_rate: float = 0.2
    ):

        super().__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        #make sure assertion layers are legit
        assert rnn_activation in ["LSTM", "GRU"]

        self.activation = rnn_activation

        #create rnnfunction
        self.rnn = self.activation(
            input_size=self.n_features,
            hidden_size=self.embedding_dim,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=self.dropout_rate
        )

        #Note that input to RNN is [batch_size, seq_length, n_features)

        #Create the dropout function
        #self.dropout = nn.Dropout(self.dropout_rate)


    def forward(self, X):

        outputs, hidden = self.rnn(X)

        #outputs: encoder outputs of length: (batch_size, seq_length, embedding_dim)
        #hidden: [n_layers, batch_size, embedding_size]

        return outputs,hidden

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, rnn_activation):
        #Map rnn_activation (key) -> actual function
        self._activation = {
            "LSTM": nn.LSTM,
            "GRU": nn.GRU
        }[rnn_activation]



class Decoder(nn.Module):
    """
    Class for decoder

    :param seq_len: (int) - sequence length of output (i.e. 24 hour sequences)
    :param embedding_dim: (int) - input dimension of latent vector c
    :param hidden_dim: (int) - size of hidden layer in RNN
    :param n_output: (int) - number of output variables (=1 for only energy consumption)
    :param n_layers: (int) - number of hidden layers in RNN
    :param rnn_activation
    """

    def __init__(
            self,
            n_features: int,
            seq_len: int = 24,
            embedding_dim: int = 16,
            n_output: int = 1,
            n_layers: int = 1,
            rnn_activation: str = "GRU",
            dropout_rate: int = 0.2,
            device : str = 'cuda'
    ):
        super().__init__()

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim #embedding dim is the input dim for the decoder
        self.n_features = n_features
        self.n_output = n_output
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        self.activation = rnn_activation
        self.device = device

        #create the RNN
        self.rnn = self.activation(
            input_size=self.n_features,
            hidden_size=self.embedding_dim,
            num_layers=self.n_layers,
            dropout=self.dropout_rate
        )

        #create the dropout function
        self.dropout = nn.Dropout(self.dropout_rate)

        #create the linear map to the output
        self.linear_map = nn.Linear(self.embedding_dim, self.n_output)

    def forward(self, X_t, encoder_outputs, hidden):

        #identify decoder_seq_length. This should be the same as seq_len
        decoder_seq_len = self.seq_len

        #initialize decoder_seq_len

        #outputs = [None for _ in range(decoder_seq_len)]
        outputs = torch.empty(size=(decoder_seq_len, 1), dtype=torch.float32).to(self.device)

        for t in range(decoder_seq_len):
            output, hidden = self.rnn(X_t, hidden)
            output = self.dropout(output)
            outputs[t, :] = self.linear_map(output)

        return outputs


    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, rnn_activation):
        #Map rnn_activation (key) -> actual function
        self._activation = {
            "LSTM": nn.LSTM,
            "GRU": nn.GRU
        }[rnn_activation]



