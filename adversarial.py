import torch
import torch.nn.functional as F
import torch.nn as nn

# from torchsummary import summary
import torch_geometric.nn as tgnn
from torch_geometric.nn import GCNConv, SGConv, MessagePassing, knn_graph
import torch_geometric as tg

# from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree, get_laplacian, remove_self_loops
import torch_scatter as tscatter
import numpy as np
import random, math, colorama
from tqdm import *
from scaffold import Scaffold
from utils import *
from bf import *
from gat_baseline import BaseDenoiser

scaf = Scaffold()
scaf.debug()
sprint = scaf.print
warn = scaf.warn


class Critic(nn.Module):
    # TODO
    pass

class AdversarialDenoiser(nn.Module):
    def __init__(self, fin, hidden_layers, **kwargs):
        r"""
        Use mirrored structure to adversarially propose noise/denoise
        """
        super().__init__()
        self.hidden_layers = [fin] + hidden_layers
        # hidden_layers shall be symmetric
        self.denoiser = AmaFilter(
            fin=fin, fout=fin, k=32, filter=BilateralFilterv2, activation=True
        )
        self.generator = AmaFilter(
            fin=fin, fout=fin, k=32, filter=BilateralFilterv2, activation=True
        )
        self.critic = Critic()

    def forward(self, data):
        pass

