from collections import OrderedDict
import torch.nn as nn
from .constants import Constants as c
from .network_utils import *

class MLP(nn.Module):
    def __init__(self, input_length, in_channels, num_outputs):
        super(MLP, self).__init__()
        
        self.input_length = input_length
        self.in_channels = in_channels
        self.num_outputs = num_outputs

        if c.normalize_data:
            initial_bn = nn.Identity()
        else:
            initial_bn = nn.BatchNorm1d(in_channels)

        self.layers = nn.Sequential(OrderedDict([
            ('bn0', initial_bn),

            ('input', nn.Linear(self.input_length, 256)),
            ('relu0', nn.ReLU()),
            ('dropout0', nn.Dropout(0.5)),
            
            ('hidden1', nn.Linear(256, 256)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.5)),

            ('hidden2', nn.Linear(256, 128)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.5)),
        ]))

        self.out = nn.Linear(128, self.num_outputs)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = x.view(x.size()[0], -1) # reshape
        x = self.layers(x)
        x = self.out(x)
        x = self.softmax(x)
        return x