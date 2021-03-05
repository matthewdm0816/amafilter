"""
Bilateral Filter Baseline: Manual Adjusted Filters
TODO: 
    0. Implement Naive BF
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
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.utils import degree, get_laplacian, remove_self_loops
import torch_scatter as tscatter
import numpy as np
import random, math, colorama, os, json
from collections import defaultdict
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
    gamma: filter strength
    sigma: kernel variance
    k: total neighbors number in kNN
    $$
        x_i=\gamma x_i + 
        (1-\gamma)\frac{\sum_{v_j\in N_i}x_j \exp(\sigma\|x_j-x_i\|^{2})}
            {\sum_{v_j\in N_i} \exp(\sigma\|x_j-x_i\|^{2})}
    $$
    """
    def __init__(self, fin, sigma=1/0.1, k=16, gamma=0.9):
        super().__init__(aggr='add')
        self.fin = fin 
        self.sigma = sigma
        self.gamma = gamma
        self.k = k

    def message(self, x_i, x_j, edge_weight, norm):
        return x_j * edge_weight.view(-1, 1)

    def update(self, aggr_out, x):
        return aggr_out * (1 - self.gamma) + x * self.gamma

    # direct copy from <class BilateralFilter._weighted_degree>
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

    def forward(self, x, batch):
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)
        n_nodes = x.size(0)

        row, col = edge_index
        x_i, x_j = x[row], x[col]
        edge_weight = torch.exp(-self.sigma * (torch.norm(x_i-x_j, dim=-1) ** 2))

        deg = self._weighted_degree(col, edge_weight, n_nodes)
        norm = deg.pow(-1.)
        norm = norm[row] # norm[i] = norm[row[i] ~ indexof(x_i)]

        return self.propagate(edge_index, x=x, norm=norm, edge_weight=edge_weight)

def evaluate(model, loader, epoch: int):
    """
    NOTE: Need DROP_LAST=TRUE, in case batch length is not uniform
    """
    model.eval()
    total_psnr, total_mse, total_orig_psnr = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader, 0):
            # torch.cuda.empty_cache()
            if parallel:
                reals, bs = parallel_cuda(batch)
                # TODO: parallel add noise
            else:
                batch = batch.to(device)
                reals = batch.pos
                jittered = add_multiplier_noise(reals.detach(), multiplier=5)
            
            orig_mse = mse(jittered, reals)
            orig_psnr = mse_to_psnr(orig_mse)
            out = model(jittered, batch=batch.batch)

            loss = mse(out, reals)
            psnr_loss = mse_to_psnr(loss)
            total_orig_psnr += orig_psnr.detach().item()
            total_psnr += psnr_loss.detach().item()
            total_mse += loss.detach().item()

    total_mse /= len(loader)
    total_psnr /= len(loader)
    total_orig_psnr /= len(loader)
    print(colorama.Fore.MAGENTA + "[%d]MSE: %.3f, PSNR: %.3f, PSNR-ORIG: %.3f" % (epoch, total_mse, total_psnr, total_orig_psnr))
    return total_mse, total_psnr, orig_psnr

if __name__ == '__main__':
    """
    Naive BF Tests
    """
    batch_size = 256
    gpu_id = 7
    # gpu_ids = [0, 1, 2, 7]
    gpu_ids = [7]
    ngpu = len(gpu_ids)
    # os.environ['CUDA_VISIBLE_DEVICES'] = repr(gpu_ids)[1:-1]
    parallel = (ngpu > 1) 
    assert gpu_id in gpu_ids

    device = torch.device("cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu")

    pl_path = 'modelnet40-1024'
    data_path = os.path.join('/data', 'pkurei', pl_path)

    for path in (data_path, ):
        check_dir(path, color=colorama.Fore.CYAN)

    # dataset and dataloader
    samplePoints = 1024
    train_dataset = ModelNet(root=data_path, name='40', train=True,
        pre_transform=transform(samplePoints=samplePoints))
    test_dataset = ModelNet(root=data_path, name='40', train=False,
        pre_transform=transform(samplePoints=samplePoints))

    if parallel: 
        train_loader = DataListLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=16, pin_memory=True)
        test_loader = DataListLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=16, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=16, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=16, pin_memory=True)

    # baseline test
    records = defaultdict(dict)
    max_psnr = -10
    best_sigma, best_gamma = None, None
    for sigma in (0.01, 0.03, 0.05, 0.1, 0.2, 0.3):
        for gamma in (0.3, 0.5, 0.75, 0.9, 0.95, 0.99):
            print(colorama.Fore.MAGENTA + "Testing config - sigma: %.2f, gamma: %.2f" % (sigma, gamma))
            model = NaiveBilateralFilter(fin=3, sigma=1/sigma, gamma=gamma).to(device)
            model.eval()
            
            total_mse, total_psnr, orig_psnr = evaluate(model, train_loader, 0)
            records[sigma][gamma] = (total_mse, total_psnr)

            if total_psnr > max_psnr:
                best_sigma, best_gamma = sigma, gamma
                max_psnr = total_psnr
    
    # record in JSON
    record_str = json.dumps(
        {
            "best_args": (best_sigma, best_gamma), 
            "records": records
        }
    )
    with open('naive-baseline.json', 'w') as f:
        f.write(record_str)
        
    