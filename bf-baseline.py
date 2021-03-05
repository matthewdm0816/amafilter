"""
Bilateral Filter Baseline: Manual Adjusted Filters
TODO: 
    1. Test on modelnet40
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
# from torchsummary import summary
import torch_geometric.nn as tgnn
from torch_geometric.nn import GCNConv, SGConv, MessagePassing, knn_graph
import torch_geometric as tg
from torch_geometric.datasets import ModelNet
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

class NaiveBilateralFilter(MessagePassing):
    """
    Naive BF.
    $$
        x_i=\gamma x_i + 
        (1-\gamma)\frac{\sum_{v_j\in N_i}x_j \exp(\sigma\|x_j-x_i\|^{2})}
            {\sum_{v_j\in N_i} \exp(\sigma\|x_j-x_i\|^{2})}
    $$
    """
    def __init__(self, fin, sigma=1/0.1, k=16, gamma=0.9):
        super().__init__(aggr='mean')
        self.fin = fin 
        self.sigma = sigma
        self.k = k

    def message(self, x_i, x_j):
        return x_j * torch.exp(-self.sigma * (torch.norm(x_i-x_j) ** 2))

    def update(self, aggr_out, x):
        return aggr_out * (1 - self.gamma) + x * self.gamma

    def forward(self, x, batch):
        edge_index = knn_graph(x, self.k, batch=batch)
        n_nodes = x.size(0)

        return self.propagate(edge_index, x=x, norm=norm)

