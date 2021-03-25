r"""
Bilateral Filter Models Definition
TODO: 
    1. Try multiple W_ij type
    2. Try different layer stucture
    3. Add graph reg. term, i.e. \tau x^T L x
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
                activation(),
            )
        elif batchnorm:
            self.net = nn.Sequential(
                nn.Linear(fin, fout), nn.BatchNorm1d(fout), activation()
            )
        else:
            self.net = nn.Sequential(nn.Linear(fin, fout), activation())

    def forward(self, x):
        return self.net(x)


class Weight(nn.Module):
    def __init__(self, fin, embedding="linear", collate="gaussian", **kwargs):
        r"""
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
        # self.embedding = embedding
        assert embedding in ["linear", "MLP"]
        if embedding == "linear":
            self.embedding = nn.Linear(fin, fout)
        elif embedding == "MLP":
            try:
                hidden_layers = [fin] + kwargs["hidden_layers"]
            except:
                hidden_layers = [fin, fout]
                warn("Weight#hidden_layers not specified, using fout instead")
            self.embedding = nn.Sequential(
                *[MLP(i, o) for i, o in layers(hidden_layers)]
            )
            for i, o in layers(hidden_layers):
                sprint("Created layer (%d, %d)" % (i, o))
            # self.embedding = nn.Sequential(mlps)

        self.collate = collate
        if collate == "gaussian":
            self.out = module_wrapper(lambda x: torch.exp(-((x) ** 2)))
        elif collate == "exponential":
            self.out = module_wrapper(lambda x: torch.exp(-(x)))
        elif collate == "fractional":
            self.out = module_wrapper(lambda x: (x) ** (-2))

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
        super().__init__(aggr="add")
        self.fproj = nn.Linear(fin, fout)
        # self.weight = Weight(fin, embedding='linear', collate='gaussian')
        self.weight = Weight(
            fout, embedding="MLP", collate="gaussian", hidden_layers=[128, fout]
        )
        sprint(
            "Created BilateralFilter with {:d} => {:d}, hidden: {}".format(
                fin, fout, [128]
            )
        )

    def _weighted_degree(self, edge_index, edge_weight, num_nodes):
        """
        edge_index ~ [E, ] of to-index of an edge
        edge_weight ~ [E, ]
        Return ~ [N, ]
        """
        # normalize
        out = torch.zeros((num_nodes,)).to(edge_weight)
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
        # selective clamp
        return torch.clamp(y, -10, 10)

    def forward(self, x, edge_index):
        """
        x ~ N * FIN
        edge_index ~ 2 * E
        """
        x = self.fproj(x)  # => N * FOUT

        n_nodes = x.size(0)

        # Compute edge weight
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        # sprint(x_i, x_j)
        # edge_weight: E * FOUT
        edge_weight = self._edge_weight(x_i, x_j)
        # FIXME: Why so small? far distance in initial embeddings

        # Compute normalization W = D^{-1}W ~ RW
        # TODO: Variable Norm, Sym/None
        deg = self._weighted_degree(col, edge_weight, n_nodes)
        norm = deg.pow(-1.0)
        norm = norm[row]  # norm[i] = norm[row[i] ~ indexof(x_i)]
        # sprint(tensorinfo(norm))
        # => E * 1
        # print(norm.shape, edge_weight.shape)
        return self.propagate(edge_index, x=x, norm=norm, edge_weight=edge_weight)


class Embedding(nn.Module):
    r"""
    Separate embedding layer, to reduce computational redundance in GNN
    I.e. calc. only once embedding of each node
    embedding:
        linear ~ Malahanobis Distance
        MLP
    kwargs:
        for linear, fout=Int
        for MLP, hidden_layers=[Int]
    """

    def __init__(self, fin, embedding, **kwargs):
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
        # self.embedding = embedding
        assert embedding in ["linear", "MLP"]
        if embedding == "linear":
            self.embedding = nn.Linear(fin, fout)
        elif embedding == "MLP":
            try:
                hidden_layers = [fin] + kwargs["hidden_layers"]
            except KeyError:
                hidden_layers = [fin, fout]
                warn("Weight#hidden_layers not specified, using fout instead")
            self.embedding = nn.Sequential(
                *[MLP(i, o) for i, o in layers(hidden_layers)]
            )
            for i, o in layers(hidden_layers):
                sprint("Created layer (%d, %d)" % (i, o))
            # self.embedding = nn.Sequential(mlps)

    def forward(self, x):
        return self.embedding(x)


class BilateralFilterv2(MessagePassing):
    def __init__(self, fin, fout, collate="gaussian"):
        super().__init__(aggr="add")
        # TODO: Alternative simplification
        # self.embedding = Embedding(fout, embedding="MLP", hidden_layers=[256, 128, fout])
        self.fproj = nn.Linear(fin, fout)
        self.embedding = Embedding(fout, embedding="MLP", hidden_layers=[128, fout])
        self.collate = collate
        if collate == "gaussian":
            self.out = module_wrapper(lambda x: torch.exp(-((x) ** 2)))
        elif collate == "exponential":
            self.out = module_wrapper(lambda x: torch.exp(-(x)))
        elif collate == "fractional":
            self.out = module_wrapper(lambda x: (x) ** (-2))
        sprint(
            "Created BilateralFilterv2 {:d} => {:d}, hidden: {}".format(
                fin, fout, [128]
            )
        )

    def _weighted_degree(self, edge_index, edge_weight, num_nodes):
        """
        edge_index ~ [E, ] of to-index of an edge
        edge_weight ~ [E, ]
        Return ~ [N, ]
        """
        # normalize
        # sprint(edge_index.shape, edge_weight.shape, num_nodes)
        # print(edge_index.shape, edge_weight.shape)
        out = torch.zeros((num_nodes,)).to(edge_weight)
        return out.scatter_add_(dim=0, index=edge_index, src=edge_weight)

    def _edge_weight(self, x_i, x_j):
        """
        x_i, x_j ~ E * FIN
        out ~ E * 1
        """
        # TODO: explicitly show embeddings of xs
        return self.out(torch.norm(x_i - x_j, dim=1)) + 1e-9

    def message(self, x_i, x_j, norm, edge_weight):
        y = norm.view(-1, 1) * edge_weight.view(-1, 1) * x_j
        # selective clamp
        return torch.clamp(y, -10, 10)

    def forward(self, x, edge_index):
        """
        x ~ N * FIN
        edge_index ~ 2 * E
        """
        x = self.fproj(x)  # => N * FOUT
        e = self.embedding(x)  # => N * FOUT through embedding layer

        n_nodes = x.size(0)

        # Compute edge weight
        row, col = edge_index
        e_i, e_j = e[row], e[col]
        # edge_weight: E * FOUT
        edge_weight = self._edge_weight(e_i, e_j)

        # Compute normalization W = D^{-1}W ~ Random Walk Laplacian
        # TODO: Variable Norm, Sym/None
        deg = self._weighted_degree(col, edge_weight, n_nodes)
        norm = deg.pow(-1.0)
        norm = norm[row]  # norm[i] = norm[row[i] ~ indexof(x_i)]
        # => E * 1
        return self.propagate(edge_index, x=x, norm=norm, edge_weight=edge_weight)


class GraphRegularizer(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def message(self, x_i, x_j, edge_weight):
        r"""
        x_i, x_j ~ [E, FIN]
        edge_weight ~ [E, 1]
        """
        # batch outer prod.
        xdim = x_i.shape[-1]  # i.e. FIN
        res = torch.einsum("bi,bj->bij", x_i, x_j)
        # print(res.shape)
        return edge_weight.view(-1, 1) * res.view(-1, xdim * xdim)

    def forward(self, x, k, edge_index=None, batch=None):
        r"""
        Calculate graph regularization term
        $R=||X^T L X||_F$
        """
        num_nodes = x.shape[-2]
        xdim = x.shape[-1]
        if edge_index is None:
            edge_index = knn_graph(x, k=k, batch=batch, loop=False)
        lap_index, lap_val = get_laplacian(
            edge_index, normalization="rw", num_nodes=num_nodes
        )
        res = self.propagate(edge_index=lap_index, x=x, edge_weight=lap_val)
        # print(res.shape)  # [B, F * F]
        # Frobenius Norm (intrinstically same)
        return (torch.norm(res, dim=-1, p="fro") ** 2).mean()


class AmaFilter(nn.Module):
    """
    Combine BFs into denoise module
    TODO:
        - Res/Dense Conn.?
        - Hidden layer adjust
        - Bottleneck?
    """

    def __init__(
        self,
        fin=6,
        fout=6,
        k=16,
        filter=BilateralFilter,
        activation: bool = True,
        reg: float = 0.0,
        loss_type: Optional[str] = None,
    ):
        super().__init__()
        self.fin, self.fout, self.k = fin, fout, k
        self.loss_type= loss_type
        hidden_layers = [fin, 64 + fin, 128 + 64 + fin, fout]
        # total = 0
        # for idx, h in enumerate(hidden_layers):
        #     total += h
        #     hidden_layers[idx] = total
        self.filters = nn.ModuleList(
            [
                filter(fin, 64),
                filter(64 + fin, 128),
                filter(64 + fin + 128, fout),
            ]
        )
        self.has_activation = activation
        self.activation = nn.ModuleList(
            [
                nn.Sequential(
                    nn.PReLU(),
                    nn.BatchNorm1d(o),
                )
                if idx != len(hidden_layers) - 2
                else nn.Identity()
                for idx, (i, o) in enumerate(layers(hidden_layers))
            ]
        )

        self.nfilters = len(self.filters)
        if reg >= 1e-6:  # non-zero G. Reg.
            self.reg = GraphRegularizer()
            self.reg_coeff = reg
            print(colorama.Fore.GREEN + "Using reg={:.1E}".format(reg))
        else:
            self.reg = None

    def forward(self, data):
        r"""
        data ~ Data(x, y, [z, ][batch, ])
        """
        # print(data)
        target, batch, x = data.y, data.batch, data.x

        for i, (filter, act) in enumerate(zip(self.filters, self.activation)):
            # dynamic graph? yes!
            edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)
            # print(edge_index.shape)
            # NOTE: denselinks added
            y = filter(x, edge_index)
            x = torch.cat((x, y), dim=-1) if i != self.nfilters - 1 else y
            if self.has_activation:
                x = act(x)
        if self.loss_type == "mse":
            loss = mse(x, target)
        elif self.loss_type == "chamfer":
            loss = chamfer_measure(x, target, batch)
        else: 
            raise NotImplementedError
        # loss = self.loss(x, target)
        mse_loss = mse(x, target)
        if self.reg is not None:
            reg_loss = self.reg(x, k=self.k, batch=batch) * self.reg_coeff
        else:
            reg_loss = torch.tensor([0.0])
        return x, reg_loss + loss, mse_loss


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
        m = Weight(fin=10, embedding="linear", collate="gaussian")
        x_i = torch.randn([20, 10])
        x_j = torch.randn([20, 10])
        dist = m(x_i, x_j)  # E * 1
        print(dist.shape, tensorinfo(dist))

        print(colorama.Fore.MAGENTA + "Testing Weight: MLP")
        m = Weight(fin=10, embedding="MLP", collate="gaussian", hidden_layers=[32, 32])
        x_i = torch.randn([20, 10])
        x_j = torch.randn([20, 10])
        dist = m(x_i, x_j)  # E * 1
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
        ]  # of 1%% edges
        print(edge_index)
        edge_index = edge_index + [[e[1], e[0]] for e in edge_index]  # make symmetric
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
        ]  # of 1%% edges
        edge_index = edge_index + [[e[1], e[0]] for e in edge_index]  # make symmetric
        edge_index = torch.tensor(edge_index).transpose(0, 1)

        # Model and verification
        model = AmaFilter(6, 6)
        init_weights(model)
        x = model(x)
        print(tensorinfo(x))
