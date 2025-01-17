import torch.nn as nn
from collections import OrderedDict
from .constants import Constants as c
from .network_utils import *

class ResNet(nn.Module):
    def __init__(self, input_length, in_channels, num_outputs):
        super(ResNet, self).__init__()

        n_feature_maps = 64

        # If the training data is not normalized we use batch_norm as the first layer
        # otherwise the data just passes through the first layer without any modification
        if c.normalize_data:
            initial_bn = nn.Identity()
        else:
            initial_bn = nn.BatchNorm1d(in_channels)

        self.layers = nn.Sequential(OrderedDict([
            ('bn0', initial_bn),
            
            ('resBlock1', ResidualBlock(   in_channels, n_feature_maps)),
            ('resBlock2', ResidualBlock(n_feature_maps, n_feature_maps*2)),
            ('resBlock3', ResidualBlock(n_feature_maps*2, n_feature_maps*2)),
            ('gap', GlobalAveragePooling())
        ]))
        self.out = nn.Linear(n_feature_maps*2, num_outputs)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layers(x)
        x = self.out(x)
        x = self.softmax(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(in_channels, out_channels, 7, padding=3, stride=1, bias=False)),
            ('bn1', nn.BatchNorm1d(out_channels)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv1d(out_channels, out_channels, 5, padding=2, stride=1, bias=False)),
            ('bn2', nn.BatchNorm1d(out_channels)),
            ('relu3', nn.ReLU()),
            ('conv3', nn.Conv1d(out_channels, out_channels, 3, padding=1, stride=1, bias=False)),
            ('bn3', nn.BatchNorm1d(out_channels)),
        ]))

        self.conv_expand = nn.Conv1d(in_channels, out_channels, 1, padding=0, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        in_x = x
        x = self.layers(x)
        shortcut = self.conv_expand(in_x)
        x = x + shortcut
        x = self.relu(x)
        return x
