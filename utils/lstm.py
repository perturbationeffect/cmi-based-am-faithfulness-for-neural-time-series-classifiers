import torch.nn as nn
from .constants import Constants as c
from .network_utils import *

class LSTM(nn.Module):
    def __init__(self, input_length, in_channels, num_outputs):
        super(LSTM, self).__init__()
        
        self.input_length = input_length
        self.in_channels = in_channels
        self.num_outputs = num_outputs

        if c.normalize_data:
            initial_bn = nn.Identity()
        else:
            initial_bn = nn.BatchNorm1d(in_channels)

        self.bn0 = initial_bn
        self.swapaxes = LSTMSwapAxes()
        self.lstm = nn.LSTM(in_channels, 256, batch_first=True)
        self.out = nn.Linear(256, self.num_outputs)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.bn0(x)
        x = self.swapaxes(x) # lstm expects tensor provided as (batch, sequence, features)
        x, _ = self.lstm(x)
        x = self.out(x[:, -1])
        x = self.softmax(x)
        return x