import torch
import torch.nn.functional as F
import torch.nn as nn
# from torchsummary import summary
import torch_geometric.nn as tgnn
from torch_geometric.nn import GCNConv, SGConv, MessagePassing
import torch_geometric as tg
# from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree, get_laplacian, remove_self_loops
import torch_scatter as tscatter
import numpy as np
import random, math, colorama
from tqdm import *
from scaffold import Scaffold
from utils import init_weights, layers, module_wrapper

scaf = Scaffold()
scaf.debug()
sprint = scaf.print


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
            layers = nn.ModuleList([
                MLP(fin, fout) for fin, fout in layers
            ])
        
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
        f1, f2 = self.embedding(x1), self.embedding(x2)
        distance = torch.norm(f1 - f2, dim=1)
        # xs = self.concat([x1, x2], dim=1) 
        # # TODO: make it single

        # increase numerical stability, using eps
        # sprint(distance)
        return self.out(distance) + eps


class BilateralFilter(MessagePassing):
    def __init__(self, fin, fout):
        super().__init__(aggr='add')
        self.fproj = nn.Linear(fin, fout)
        self.weight = Weight(fin, embedding='linear', collate='gaussian')

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

    def message(self, x_i, x_j, norm):
        return norm.view(-1, 1) * x_j

    def forward(self, x, edge_index):
        """
        x ~ N * FIN
        edge_index ~ 2 * E
        """
        x = self.fproj(x) # => N * FOUT

        n_nodes = x.size(0)

        # Compute edge weight
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        # sprint(x_i, x_j)
        edge_weight = self._edge_weight(x_i, x_j)
        # FIXME: Why so small? far distance in initial embeddings
        sprint(edge_weight, edge_weight.max(), edge_weight.median(), edge_weight.min())

        # Compute normalization W = D^{-1}W ~ RW
        # TODO: Variable Norm, Sym/None
        deg = self._weighted_degree(col, edge_weight, n_nodes)
        norm = deg.pow(-1.)
        norm = norm[row] # norm[i] = norm[row[i] ~ indexof(x_i)]
        sprint(norm, norm.max(), norm.median(), norm.min())
        # => E * 1
        
        return self.propagate(edge_index, x=x, norm=norm)


class Denoiser(nn.Module):
    # TODO: Add Denoiser and Unit Tests
    pass

if __name__ == "__main__":
    colorama.init(autoreset=True)
    """
    Unit test of module_wrapper
    """
    print(colorama.Fore.MAGENTA + "Testing module_wrapper")
    m = module_wrapper(lambda x: x.pow(-2))
    x = torch.randn([100, 10])
    x = m(x)
    print(x.max())

    """
    Unit test of Weight
    """
    print(colorama.Fore.MAGENTA + "Testing Weight")
    m = Weight(fin=10, embedding='linear', collate='gaussian')
    x_i = torch.randn([20, 10])
    x_j = torch.randn([20, 10])
    dist = m(x_i, x_j) # E * 1
    print(dist.shape, dist)

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
        print(x.max(), x.median(), x.min())


