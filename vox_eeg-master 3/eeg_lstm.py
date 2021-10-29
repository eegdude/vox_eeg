import numpy as np
import pandas as pd
import mne
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x, hidden):

        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)
        #out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(out)#.view(batch_size, -1)
        out = self.fc(out)

        #out, hidden = self.rnn(x, hidden)
        #out = nn.Sigmoid(out)
        out = self.relu(out)
        out = out.view(batch_size, 2, -1)
        out = out[:,:,-1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden
