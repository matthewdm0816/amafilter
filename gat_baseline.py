from re import M
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

# from torchsummary import summary
import torch_geometric.nn as tgnn
from torch_geometric.nn import GCNConv, SGConv, MessagePassing, knn_graph, DataParallel
import torch_geometric as tg
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.utils import degree, get_laplacian, remove_self_loops
from torch_geometric.nn import GATConv, knn_graph
import torch_scatter as tscatter
from tensorboardX import SummaryWriter
import numpy as np

import random, math, colorama, os, json
from collections import defaultdict
from tqdm import *
from itertools import product
from typing import Optional, List
import pretty_errors

from scaffold import Scaffold
from utils import *
from train_bf import process_batch
from dataloader import ADataListLoader, MPEGDataset, MPEGTransform
from bf import MLP


class GATDenoiser(nn.Module):
    r"""
    Baseline GAT as denoiser
    examplar hidden_layers:
    [
        {"f":16, "heads":4}, # i.e. 64
        {"f":64, "heads":2}, # i.e. 128
        {"f":6, "heads":8, "concat":False, "negative_slope":1.0}
        # cancel activation at last layer
    ]
    """

    def __init__(self, fin, hidden_layers: list):
        super().__init__()
        hidden_layers = [{"f": fin, "heads": 1}] + hidden_layers
        self.gats = nn.ModuleList(
            [
                GATConv(
                    in_channels=i["f"] * i["heads"],
                    out_channels=o["f"],
                    heads=o["heads"],
                    concat=o["concat"] if "concat" in o.keys() else True,
                    negative_slope=o["negative_slope"]
                    if "negative_slope" in o.keys()
                    else 0.2,
                )
                for i, o in layers(hidden_layers)
            ]
        )
        self.activation = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.BatchNorm1d(o["f"] * o["heads"]),
                ) if idx != len(hidden_layers) - 2 else nn.Identity()
                for idx, (i, o) in enumerate(layers(hidden_layers))
            ]
        )

    def forward(self, data):
        # print(data)
        target, batch, x = data.y, data.batch, data.x
        for i, (filter, activation) in enumerate(zip(self.gats, self.activation)):
            edge_index = knn_graph(x, k=32, batch=batch, loop=False)
            x = filter(x, edge_index=edge_index)
            x = activation(x)

        loss = mse(x, target)
        return x, loss
