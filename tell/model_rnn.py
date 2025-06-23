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

    def __init__(
            self,
            input_dim: int,
            device: str = 'cuda',
            encoder_embedding_dim: int = 16,
            decoder_embedding_dim: int = 16,
            hidden_dim: int= 32,
            n_layers: int = 1,
            encoder_dropout: float = 0.2,
            decoder_dropout: float = 0.2,
            out_dim: int = 1,
            activation: str = "LSTM"
    ):

        super().__init__()

        self.device = device

        #Initialize encoder
        self.encoder = Encoder(
            input_dim=input_dim,
            embedding_dim=encoder_embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            rnn_activation=activation,
            dropout_rate=encoder_dropout
        )

        #initialize decoder
        self.decoder = Decoder(
            embedding_dim=decoder_embedding_dim,
            output_dim=out_dim,
            hidden_dim=hidden_dim,
            n_layers = n_layers,
            rnn_activation=activation,
            dropout_rate=decoder_dropout

        )

    def forward(self, X, Y):

        batch_size = X.shape[1]
        seq_len = Y.shape[0]
        out_dim = self.decoder.output_dim

        #initialize outputs
        outputs = torch.zeros(seq_len, batch_size, out_dim).to(self.device)

        #feed the data to the encoder and get the hidden cells and state
        hidden, cell = self.encoder(X)

        #loop over all the outputs. Notice that there might be one extra input to kickastart the sequence
        Y0 = Y[0, :]

        for t in range(1, seq_len):
            Y, hidden, cell = self.decoder(Y0, hidden, cell) #initializing with Y0
            outputs[t, :, :] =  Y
            return outputs
    

class Encoder(nn.Module):

    def __init__(
            self,
            input_dim: int,
            embedding_dim: int = 16,
            hidden_dim: int = 32,
            n_layers: int = 1,
            rnn_activation: str = "LSTM",
            dropout_rate: float = 0.2
    ):

        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim #hidden dim is the dimension of the c vector
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        #make sure assertion layers are legit
        assert rnn_activation in ["LSTM", "GRU"]

        self.activation = rnn_activation

        #create embedding
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim) #linear embedding

        #create rnnfunction
        self.rnn = self.activation(
            self.embedding_dim,
            self.hidden_dim,
            num_layers = self.n_layers,
            dropout = self.dropout_rate
        )

        #Create the dropout function
        self.dropout = nn.Dropout(self.dropout_rate)


    def forward(self, input_data):

        #input_data: [sequence_length, batch_size]
        embedded = self.dropout(self.embedding(input_data)) #embedded: [input_dim, batch_size, embded_dim]
        output, (hidden, cell) = self.rnn(embedded)

        #output: [sequence_length, batch_size, hidden_dim]
        #hidden: [n_layers, batch_size, hidden_dim]
        #cell:  [n_layers, batch_size, hidden_dim]

        return hidden, cell

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

    def __init__(
            self,
            embedding_dim: int,
            output_dim: int,
            hidden_dim: int = 32,
            n_layers: int = 1,
            rnn_activation: str = "LSTM",
            dropout_rate: int = 0.2
    ):

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        self.activation = rnn_activation

        #create linear embedding
        self.embedding = nn.Embedding(output_dim, embedding_dim)

        #create the RNN
        self.rnn = self.activation(
            self.embedding_dim,
            self.hidden_dim,
            num_layers=self.n_layers,
            dropout=self.dropout_rate
        )

        #create the linear map to the output
        self.linear_map = nmn.Linear(self.hidden_dim, self.output_dim)

        #create the dropout function
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input_data, hidden, cell):

        #input_data = [batch_size] (this is the sos prompt)
        input = input_data.unsqueece #[batch_size] -> [1, batch_size]
        embedded = self.dropout(self.embedding(input)) #embedded: [1, embedding_size, batch_size]

        #define the RNN. takes in embedding as input
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        #hidden: [n_layers, batch_size, hidden_dim]
        #cell:  [n_layers, batch_size, hidden_dim]

        #finally compute the output map
        y = self.linear_map(output.squeece(0))
        return y, hidden, cell


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

