"""
Bilateral Filter Models Definition
TODO: 
    1. Try multiple W_ij type
    2. Try different layer stucture
"""

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

scaf = Scaffold()
scaf.debug()
sprint = scaf.print
warn = scaf.warn


class MLP(nn.Module):
    """
    Plain MLP with activation
    """
    def __init__(self, fin, fout, activation=nn.ReLU, dropout=None, batchnorm=True):
        super().__init__()
        if dropout is not None and batchnorm:
            assert isinstance(dropout, float)
            self.net = nn.Sequential(
                nn.Linear(fin, fout),
                nn.BatchNorm1d(fout),
                nn.Dropout(p=dropout),
                activation()
            )
        elif batchnorm:
            self.net = nn.Sequential(
                nn.Linear(fin, fout),
                nn.BatchNorm1d(fout),
                activation()
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(fin, fout),
                activation()
            )

    def forward(self, x):
        return self.net(x)


class Weight(nn.Module):
    def __init__(self, fin, embedding='linear', collate='gaussian', **kwargs):
        """
        embedding: 
            linear ~ Malahanobis Distance 
            MLP
        collate:
            gaussian ~ exp(-t^2)
            exponential ~ exp(-t)
            fractional ~ 1/x^2
        kwargs:
            for linear, fout=Int
            for MLP, hidden_layers=[Int]
        TODO: 
            - Optimize into one Φ(x1, x2) using single MLP/Linear under concat
            - Use more embedding
            - Init embedding, make it separate INITIALLY
        """
        super().__init__()
        try:
            fout = kwargs["fout"]
        except KeyError: 
            fout = fin
        try:
            fout = kwargs["hidden_layers"][-1]
        except KeyError: 
            fout = fin
        self.fout = fout
        self.embedding = embedding
        assert embedding in ['linear', 'MLP']
        if embedding == 'linear':
            self.embedding = nn.Linear(fin, fout)
        elif embedding == 'MLP':
            try:
                hidden_layers = [fin] + kwargs["hidden_layers"]
            except:
                hidden_layers = [fin, fout]
                warn("Weight#hidden_layers not specified, using fout instead")
            self.embedding = nn.Sequential(*[
                MLP(i, o) for i, o in layers(hidden_layers)
            ])
            for i, o in layers(hidden_layers):
                sprint("Created layer (%d, %d)" % (i, o)) 
            # self.embedding = nn.Sequential(mlps)
        
        self.collate = collate
        if collate == 'gaussian':
            self.out = module_wrapper(lambda x:
                torch.exp(-(x) ** 2)
            )
        elif collate == 'exponential':
            self.out = module_wrapper(lambda x:
                torch.exp(-(x))
            )
        elif collate == 'fractional':
            self.out = module_wrapper(lambda x:
                (x) ** (-2)
            )
        
    def forward(self, x1, x2, eps=1e-9):
        """
        Input: x1, x2 ~ N * F
        Output: w ~ N * 1?
        """
        # sprint(x1, x2)
        f1 = self.embedding(x1)
        f2 = self.embedding(x2)
        distance = torch.norm(f1 - f2, dim=1)
        # xs = self.concat([x1, x2], dim=1) 
        # # TODO: make it single

        # increase numerical stability, using eps
        # sprint(distance)
        return self.out(distance) + eps


class BilateralFilter(MessagePassing):
    """
    Bilateral Filter Impl. under PyG
    In: (B * N) * FIN
    X' = D^-1 W X Θ
    """
    def __init__(self, fin, fout):
        super().__init__(aggr='add')
        self.fproj = nn.Linear(fin, fout)
        # self.weight = Weight(fin, embedding='linear', collate='gaussian')
        self.weight = Weight(fout, embedding='MLP', collate='gaussian', hidden_layers=[128, fout])

    def _weighted_degree(self, edge_index, edge_weight, num_nodes):
        """
        edge_index ~ [E, ] of to-index of an edge
        edge_weight ~ [E, ]
        Return ~ [N, ]
        """
        # normalize
        # sprint(edge_index.shape, edge_weight.shape, num_nodes)
        out = torch.zeros((num_nodes, )).to(edge_weight)
        return out.scatter_add_(dim=0, index=edge_index, src=edge_weight)

    def _edge_weight(self, x_i, x_j):
        """
        x_i, x_j ~ E * FIN
        out ~ E * 1
        """
        # TODO: explicitly show embeddings of xs
        return self.weight(x_i, x_j)

    def message(self, x_i, x_j, norm, edge_weight):
        y = norm.view(-1, 1) * edge_weight.view(-1, 1) * x_j
        # selective. clamp
        return torch.clamp(y, -1, 1)

    def forward(self, x, edge_index):
        """
        x ~ N * FIN
        edge_index ~ 2 * E
        """
        x = self.fproj(x) # => N * FOUT

        n_nodes = x.size(0)

        # Compute edge weight
        # print(edge_index, edge_index.shape)
        # print(x.shape)
        # print(edge_index.max(), edge_index.min())
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        # sprint(x_i, x_j)
        # edge_weight: E * FOUT
        edge_weight = self._edge_weight(x_i, x_j)
        # FIXME: Why so small? far distance in initial embeddings
        # sprint(tensorinfo(edge_weight))

        # Compute normalization W = D^{-1}W ~ RW
        # TODO: Variable Norm, Sym/None
        deg = self._weighted_degree(col, edge_weight, n_nodes)
        norm = deg.pow(-1.)
        norm = norm[row] # norm[i] = norm[row[i] ~ indexof(x_i)]
        # sprint(tensorinfo(norm))
        # => E * 1
        # print(norm.shape, edge_weight.shape)
        return self.propagate(edge_index, x=x, norm=norm, edge_weight=edge_weight)


class AmaFilter(nn.Module):
    """
    Combine BFs into denoise module
    TODO:
        - Res/Dense Conn.?
        - Hidden layer adjust
        - Bottleneck?
    """
    def __init__(self, fin=6, fout=6):
        super().__init__()
        # self.filters = nn.ModuleList([
        #     BilateralFilter(fin, 64),
        #     BilateralFilter(64, 128),
        #     BilateralFilter(128, fout)
        # ])
        self.filters = nn.ModuleList([
            BilateralFilter(fin, 128),
            # BilateralFilter(64, 128),
            BilateralFilter(128 + fin, fout)
        ])

        self.nfilters = len(self.filters)

    def forward(self, x, batch=None, k=16):
        for i, filter in enumerate(self.filters):
            # dynamic graph?
            edge_index = knn_graph(x, k=k, batch=batch, loop=False)
            # print(edge_index, edge_index.shape)
            # NOTE: denselinks
            y = filter(x, edge_index)
            x = torch.cat((x, y), dim=-1) if i != self.nfilters - 1 else y
        return x


if __name__ == "__main__":
    colorama.init(autoreset=True)
    """
    Unit test of module_wrapper
    """
    print(colorama.Fore.MAGENTA + "Testing module_wrapper")
    m = module_wrapper(lambda x: x.pow(-2))
    x = torch.randn([100, 10])
    x = m(x)
    print(tensorinfo(x))

    """
    Unit test of MLP
    """
    with torch.no_grad():
        mlp = MLP(100, 10)
        print(colorama.Fore.MAGENTA + "Testing MLP")
        x = torch.randn([100, 100])
        init_weights(mlp)
        y = mlp(x)
        print(tensorinfo(y))
        

    """
    Unit test of Weight
    """
    with torch.no_grad():
        print(colorama.Fore.MAGENTA + "Testing Weight: Linear")
        m = Weight(fin=10, embedding='linear', collate='gaussian')
        x_i = torch.randn([20, 10])
        x_j = torch.randn([20, 10])
        dist = m(x_i, x_j) # E * 1
        print(dist.shape, tensorinfo(dist))

        print(colorama.Fore.MAGENTA + "Testing Weight: MLP")
        m = Weight(fin=10, embedding='MLP', collate='gaussian', hidden_layers=[32, 32])
        x_i = torch.randn([20, 10])
        x_j = torch.randn([20, 10])
        dist = m(x_i, x_j) # E * 1
        print(dist.shape, tensorinfo(dist))

    """
    Unit test of BF
    FIXME: at earliest iterations, weights are big/small => numerical instability, added eps threshold
    """
    with torch.no_grad():
        print(colorama.Fore.MAGENTA + "Testing BF")
        # Compose data
        l = 30
        x = torch.randn([l, 6])
        choices = [list(range(l)) for _ in range(l)]
        for i, c in enumerate(choices):
            c.remove(i)
        edge_index = [
            [i, random.choice(seq)] for i, seq in enumerate(choices)
        ] # of 1%% edges
        print(edge_index)
        edge_index = edge_index + [
            [e[1], e[0]] for e in edge_index
        ] # make symmetric
        edge_index = torch.tensor(edge_index).transpose(0, 1)
        print(edge_index.shape)

        # Model and verification
        model = BilateralFilter(6, 6)
        init_weights(model)
        x = model(x, edge_index)
        print(tensorinfo(x))

    """
    Unit test of Denoiser
    """
    with torch.no_grad():
        print(colorama.Fore.MAGENTA + "Testing denoiser")
        # Compose data
        l = 30
        x = torch.randn([l, 6])
        choices = [list(range(l)) for _ in range(l)]
        for i, c in enumerate(choices):
            c.remove(i)
        edge_index = [
            [i, random.choice(seq)] for i, seq in enumerate(choices)
        ] # of 1%% edges
        edge_index = edge_index + [
            [e[1], e[0]] for e in edge_index
        ] # make symmetric
        edge_index = torch.tensor(edge_index).transpose(0, 1)

        # Model and verification
        model = AmaFilter(6, 6)
        init_weights(model)
        x = model(x)
        print(tensorinfo(x))


