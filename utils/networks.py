from .vit.vit import ViT
from .inception import Inception
from .resnet import ResNet
from .mlp import MLP
from .lstm import LSTM

network_dict = {
    'Inception' : Inception,
    'ResNet' : ResNet,
    'LSTM' : LSTM,
    'MLP' : MLP,
    'ViT' : ViT,
}