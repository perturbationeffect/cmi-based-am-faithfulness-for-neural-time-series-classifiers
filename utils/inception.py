import torch.nn as nn
from collections import OrderedDict
from .constants import Constants as c
from .network_utils import *

# NOTE
# Changes were made to the original inception time from: https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
#   1. Batchnorm was not applied to the shortcut connections, since it hurts performance: http://torch.ch/blog/2016/02/04/resnets.html
#   2. The Kernel sizes were changed from 10, 20, 40 to 11, 21, 41, to simplify padding in pytorch, since pytorch does not have padding=same like tensorflow
#   3. The ReLu activation from the last inception block is removed, and ReLu is applied only after addition

class Inception(nn.Module):
    def __init__(self, input_length: int, in_channels: int, num_outputs: int):
        super(Inception, self).__init__()

        n_feature_maps = 128

        # If the training data is not normalized we use batch_norm as the first layer
        # otherwise the data just passes through the first layer without any modification
        if c.normalize_data:
            initial_bn = nn.Identity()
        else:
            initial_bn = nn.BatchNorm1d(in_channels)

        self.layers = nn.Sequential(OrderedDict([
            ('bn0', initial_bn),
            
            ('residualBlock1', ResidualBlock(   in_channels, n_feature_maps)),
            ('residualBlock2', ResidualBlock(n_feature_maps, n_feature_maps)),
            ('gap', GlobalAveragePooling())
        ]))
        self.out = nn.Linear(n_feature_maps, num_outputs)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        x = self.out(x)
        x = self.softmax(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()

        # ReLu not used in last InceptionBlock since ReLu will be applied after addition
        self.layers = nn.Sequential(OrderedDict([
            ('inceptionBlock1', InceptionBlock(in_channels, out_channels)),
            ('inceptionBlock2', InceptionBlock(out_channels, out_channels)),
            ('inceptionBlock3', InceptionBlock(out_channels, out_channels, use_relu = False)),
        ]))
        
        self.conv_expand = nn.Conv1d(in_channels, out_channels, 1, padding=0, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        in_x = x
        x = self.layers(x)
        shortcut = self.conv_expand(in_x)
        x = x + shortcut
        x = self.relu(x)
        return x



class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_relu: bool = True):
        super(InceptionBlock, self).__init__()

        self.use_relu = use_relu
        bottleneck_size = 32

        self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, 1, padding=0, stride=1)
        n_filters = out_channels // 4
        
        self.conv_s = nn.Conv1d(bottleneck_size, n_filters, 11, padding=5, stride=1, bias=False)
        self.conv_m = nn.Conv1d(bottleneck_size, n_filters, 21, padding=10, stride=1, bias=False)
        self.conv_l = nn.Conv1d(bottleneck_size, n_filters, 41, padding=20, stride=1, bias=False)

        self.max_pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_mp = nn.Conv1d(bottleneck_size, n_filters, 1, padding=0, stride=1, bias=False)

        self.bn = nn.BatchNorm1d(n_filters * 4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # bottleneck
        x = self.bottleneck(x)

        # parallel convolutions
        x = torch.cat([
            self.conv_s(x),
            self.conv_m(x),
            self.conv_l(x),
            self.conv_mp(self.max_pool(x))
        ], dim=1)
        
        x = self.bn(x)

        if self.use_relu:
            x = self.relu(x)
        return x