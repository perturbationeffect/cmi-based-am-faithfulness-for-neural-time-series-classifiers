from tsai.all import *
import torch.nn as nn

# Install TSAI: https://timeseriesai.github.io/tsai/#installation

class ResNetMulti(nn.Module):
    def __init__(self, input_length, in_channels, num_outputs):
        super(ResNetMulti, self).__init__()

        tsai_resnet = ResNet(in_channels, num_outputs)
        softmax = nn.LogSoftmax(dim=1)

        self.classifier = nn.Sequential(tsai_resnet, softmax)
    
    def forward(self, x):
        x = self.classifier(x)
        return x