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
from torch_geometric.nn import GCNConv, SGConv, MessagePassing, knn_graph, DataParallel
import torch_geometric as tg
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.utils import degree, get_laplacian, remove_self_loops
import torch_scatter as tscatter
import numpy as np
import random, math, colorama, os, json
from collections import defaultdict
from tqdm import *
from itertools import product
from scaffold import Scaffold
from utils import *
from train_bf import process_batch
from dataloader import ADataListLoader, MPEGDataset, MPEGTransform

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
    NOTE: Must use >=2 GPUs when wrapped by DataParallel,
    NOTE: otherwise the propagated result (after aggr.) would be nan.
    """

    def __init__(self, fin, sigma=1.0, k=16, gamma=0.9):
        super().__init__(aggr="add")
        self.fin = fin
        self.sigma = sigma
        self.gamma = gamma
        self.k = k

    def message(self, x_i, x_j, edge_weight, norm):
        # print(x_j.shape)
        # sprint(tensorinfo(x_j))
        # sprint(tensorinfo(edge_weight))
        # sprint(tensorinfo(norm))
        return norm.view(-1, 1) * edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out, x):
        # sprint(tensorinfo(aggr_out))
        # sprint(tensorinfo(x))
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
        out = torch.zeros((num_nodes,)).to(edge_weight)
        return out.scatter_add_(dim=0, index=edge_index, src=edge_weight)

    def forward(self, data):
        target, batch, x = data.y, data.batch, data.x
        eps = 1e-6
        # x = x + torch.randn_like(x).to(x) * eps # NOTE: add dither reduced error rate but still exists
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)
        # sprint(x.shape)
        # FIXME: why the edges is less than self.k * bn????
        # but it's ok in BilateralFilter???

        try:
            assert edge_index.shape[1] == x.shape[0] * self.k, "Mismatch knn"
        except AssertionError:
            sprint(tensorinfo(batch))
            # sprint(self.k)
            sprint(edge_index.shape)
            sprint(edge_index)
            row, col = edge_index
            wholeset = set(range(64 * 2048))
            rem = list(wholeset - set(col.tolist()))
            idx = rem[0] / 2048
            print(idx)  # unique
            assert False
        n_nodes = x.size(0)

        row, col = edge_index
        x_i, x_j = x[row], x[col]
        dist = torch.sum((x_i - x_j) ** 2.0, dim=-1)
        # sprint(tensorinfo(dist))
        edge_weight = torch.exp(-self.sigma * dist) + eps
        # sprint(tensorinfo(edge_weight))
        deg = self._weighted_degree(col, edge_weight, n_nodes)
        # sprint(tensorinfo(deg))
        norm = deg.pow(-1.0)
        norm = norm[row]  # norm[i] = norm[row[i] ~ indexof(x_i)]
        # sprint(tensorinfo(norm))

        result = self.propagate(edge_index, x=x, norm=norm, edge_weight=edge_weight)
        # sprint(tensorinfo(result))
        return mse(result, target)


def evaluate(model, loader, epoch: int):
    """
    NOTE: Need DROP_LAST=TRUE, in case batch length is not uniform
    """
    global parallel, dataset_type
    model.eval()
    total_psnr, total_mse, total_orig_mse = 0, 0, 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader, 0), total=len(loader)):
            # print(colorama.Fore.RED + 'Batch %d' % i)
            batch, orig_mse = process_batch(
                batch, parallel=parallel, dataset_type=dataset_type
            )

            orig_psnr = mse_to_psnr(orig_mse)
            loss = model(batch).mean()

            psnr_loss = mse_to_psnr(loss)
            total_psnr += psnr_loss.detach().item()
            total_mse += loss.detach().item()
            total_orig_mse += orig_mse.detach().item()

    total_mse /= len(loader)
    total_psnr /= len(loader)
    total_orig_mse /= len(loader)
    print(
        colorama.Fore.MAGENTA
        + "[%d]MSE: %.3f, PSNR: %.3f, ORIG-MSE: %.3f"
        % (epoch, total_mse, total_psnr, total_orig_mse)
    )
    return total_mse, total_psnr, total_orig_mse


if __name__ == "__main__":
    """
    Naive BF Tests
    """
    gpu_id = 6
    # gpu_ids = [0, 1, 2, 7]
    gpu_ids = [6, 7]
    ngpu = len(gpu_ids)
    batch_size = 512 * ngpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = repr(gpu_ids)[1:-1]
    parallel = (ngpu > 1) or True  # use 1 gpu parallel
    assert gpu_id in gpu_ids

    device = torch.device("cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu")
    dataset_type = "MPEG"

    # model and data path
    if dataset_type == "MN40":
        pl_path = "modelnet40-1024"
        data_path = os.path.join("/data", "pkurei", pl_path)
    elif dataset_type == "MPEG":
        # pl_path = 'pku'
        dataset_name = "data-5.0"
        data_path = os.path.join(dataset_name)
    print(colorama.Fore.RED + "Testing on dataset %s at %s" % (dataset_type, data_path))

    for path in (data_path,):
        check_dir(path, color=colorama.Fore.CYAN)

    # dataset and dataloader
    if dataset_type == "MN40":
        samplePoints = 1024
        train_dataset = ModelNet(
            root=data_path,
            name="40",
            train=True,
            pre_transform=transform(samplePoints=samplePoints),
        )
        test_dataset = ModelNet(
            root=data_path,
            name="40",
            train=False,
            pre_transform=transform(samplePoints=samplePoints),
        )
        if parallel:
            train_loader = DataListLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=16,
                pin_memory=True,
            )
            test_loader = DataListLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=16,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=16,
                pin_memory=True,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=16,
                pin_memory=True,
            )
    elif dataset_type == "MPEG":
        dataset = MPEGDataset(root=data_path, pre_transform=MPEGTransform)
        if parallel:
            train_loader = ADataListLoader(
                dataset,
                training=True,
                test_classes=[],
                batch_size=batch_size,
                shuffle=False,
            )
            test_loader = ADataListLoader(
                dataset,
                training=False,
                test_classes=[0],
                batch_size=batch_size,
                shuffle=True,
            )
        else:
            raise NotImplementedError

    # baseline test
    records = defaultdict(dict)
    max_psnr, min_mse = -10, 1e10
    best_sigma, best_gamma = None, None
    # sigma: kernel std
    # gamma: original image ratio
    for sigma, gamma in tqdm(
        product(
            (0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10),
            (0, 0.1, 0.3, 0.5, 0.75, 0.9, 0.99),
        )
    ):
        print(
            colorama.Fore.MAGENTA
            + "Testing config - sigma: %.3E, gamma: %.3E" % (sigma, gamma)
        )

        model = NaiveBilateralFilter(fin=6, sigma=sigma, gamma=gamma, k=32)
        if parallel:
            model = DataParallel(model, device_ids=gpu_ids, output_device=gpu_id).to(
                device
            )
        else:
            model = model.to(device)
        model.eval()

        total_mse, total_psnr, orig_psnr = evaluate(model, train_loader, 0)
        records[sigma][gamma] = (total_mse, total_psnr)

        if total_psnr > max_psnr:
            best_sigma, best_gamma = sigma, gamma
            max_psnr, min_mse = total_psnr, total_mse

    print(
        colorama.Fore.GREEN
        + "Max PSNR: %.3f, min MSE: %.3f, ORIG-MSE: %.3f@ sigma: %.3f, gamma: %.3f"
        % (max_psnr, min_mse, orig_psnr, best_sigma, best_gamma)
    )
    # record in JSON
    record_str = json.dumps(
        {
            "dataset": data_path,
            "best_args": (best_sigma, best_gamma),
            "records": records,
        }
    )
    with open("naive-baseline-%s.json" % dataset_name, "w") as f:
        f.write(record_str)
