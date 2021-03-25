from re import M
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
# import torch.linalg

# from torchsummary import summary
import torch_geometric.nn as tgnn
from torch_geometric.nn import (
    GCNConv,
    SGConv,
    MessagePassing,
    knn_graph,
    DataParallel,
    GMMConv,
)
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

scaf = Scaffold()
scaf.debug()
sprint = scaf.print
warn = scaf.warn


class BaseDenoiser(nn.Module):
    r"""
    Generic Base Denoiset Architecture
    x=>FILTER=>ACTIVATION(if not output layer)=>...=>y
    """

    def __init__(self, fin, hidden_layers, activation: bool = True):
        super().__init__()
        self.fin, self.hidden_layers = fin, self.process_fin(fin) + hidden_layers
        self.filters = nn.ModuleList(
            [self.get_layer(i, o) for i, o in layers(self.hidden_layers)]
        )
        self.activation = nn.ModuleList(
            [
                self.get_activation(i, o)
                if idx != len(self.hidden_layers) - 2
                else nn.Identity()
                for idx, (i, o) in enumerate(layers(self.hidden_layers))
            ]
        )
        self.has_activation = activation

    def get_layer(self, i, o):
        raise NotImplementedError

    def get_activation(self, i, o):
        raise NotImplementedError

    def process_fin(self, fin):
        return NotImplementedError

    # def calc_filter(x, filter, edge_index):
    #     return filter(x, edge_index=edge_index)

    def forward(self, data):
        # print(data)
        target, batch, x = data.y, data.batch, data.x
        for i, (filter, activation) in enumerate(zip(self.filters, self.activation)):
            edge_index = knn_graph(x, k=32, batch=batch, loop=False)
            x = filter(x, edge_index=edge_index)
            if self.has_activation:
                x = activation(x)

        loss = mse(x, target)
        return x, loss


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

    def __init__(self, fin, hidden_layers: list, activation: bool = True):
        super().__init__()
        hidden_layers = [{"f": fin, "heads": 1}] + hidden_layers
        self.has_activation = activation
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
                )
                if idx != len(hidden_layers) - 2
                else nn.Identity()
                for idx, (i, o) in enumerate(layers(hidden_layers))
            ]
        )

    def forward(self, data):
        # print(data)
        target, batch, x = data.y, data.batch, data.x
        for i, (filter, activation) in enumerate(zip(self.gats, self.activation)):
            edge_index = knn_graph(x, k=32, batch=batch, loop=False)
            x = filter(x, edge_index=edge_index)
            if self.has_activation:
                x = activation(x)

        loss = mse(x, target)
        return x, loss


class MoNetDenoiser(BaseDenoiser):
    def __init__(
        self,
        fin,
        hidden_layers,
        activation: bool = True,
        kernel_size: int = 8,
        separate_gaussians: bool = True,
    ):
        self.kernel_size = kernel_size
        self.separate_gaussians = separate_gaussians
        super().__init__(fin, hidden_layers, activation=activation)

    def get_activation(self, i, o):
        return nn.Sequential(
            nn.PReLU(),
            nn.BatchNorm1d(o),
        )

    def get_layer(self, i, o):
        sprint("Created GMMConv %d => %d" % (i, o))
        return GMMConv(
            in_channels=i,
            out_channels=o,
            dim=i,
            kernel_size=self.kernel_size,
            separate_gaussians=self.separate_gaussians,
        )

    def process_fin(self, fin):
        return [fin]

    def forward(self, data):
        target, batch, x = data.y, data.batch, data.x
        for i, (filter, activation) in enumerate(zip(self.filters, self.activation)):
            edge_index = knn_graph(x, k=32, batch=batch, loop=False)
            row, col = edge_index
            edge_attr = x[row] - x[col]
            # print(edge_attr.shape, edge_index.shape)
            # e_ij = x_i - x_j
            x = filter(x, edge_index=edge_index, edge_attr=edge_attr)
            if self.has_activation:
                x = activation(x)

        loss = mse(x, target)
        return x, loss


class ICADenoiser(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.fin = fin
        self.lin = nn.Linear(fin, fin)

    def whiten(self, x, batch_size: int):
        r"""
        Perform batch whiten, by SVD
        $\Sigma = U\Lambda V^T$ (for square matrices, U=V)
        $\tilde X = U^T \Lambda^{-1/2} X$
        """
        x = x.view(batch_size, -1, self.fin)
        # x ~ [B, N, F]
        u, s, _ = torch.svd(x, some=True)
        # s ~ [B, N], U, V ~ [B, N, N]
        x = x - x.mean(dim=-2)
        norm = torch.bmm(torch.diag_embed(s.pow(-0.5)), torch.transpose(u, -1, -2))
        x = torch.bmm(norm, x)
        return x

    def forward(self, data, batch_size: int):
        target, batch, x = data.y, data.batch, data.x
        x = self.whiten(x, batch_size)
        # TODO
        pass
