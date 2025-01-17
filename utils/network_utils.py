import torch
import torch.nn as nn

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        return torch.mean(x, 2)


class LSTMSwapAxes(nn.Module):
    def __init__(self):
        super(LSTMSwapAxes, self).__init__()

    def forward(self, x):
        return torch.swapaxes(x,1,2) # lstm expects tensor provided as (batch, sequence, features)


class Permute_bs_c_len_to_len_bs_c(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.permute(x, (2,0,1))


class Permute_len_bs_dmodel_to_bs_len_dmodel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.permute(x, (1,0,2))

class Max(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(x, 1)[0]