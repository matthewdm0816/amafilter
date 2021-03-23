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
from torch_geometric.nn import DynamicEdgeConv, EdgeConv
import torch_scatter as tscatter
from tensorboardX import SummaryWriter
import numpy as np
import re

import random, math, colorama, os, json
from collections import defaultdict
from tqdm import *
from itertools import product
from typing import Optional, List
import pretty_errors
import importlib

from scaffold import Scaffold
from utils import *
from train_bf import process_batch, train, evaluate, get_data
from dataloader import ADataListLoader, MPEGDataset, MPEGTransform
from bf import MLP
from gat_baseline import GATDenoiser, MoNetDenoiser

scaf = Scaffold()
scaf.debug()
sprint = scaf.print
warn = scaf.warn

samplePoints = 1024
epochs = 1001
milestone_period = 5
use_sbn = True
gpu_id = 0
# gpu_ids = [0, 1, 2, 7]
gpu_ids = [0, 1, 2, 3, 4, 5]
ngpu = len(gpu_ids)
# os.environ['CUDA_VISIBLE_DEVICES'] = repr(gpu_ids)[1:-1]
parallel = ngpu > 1
assert gpu_id in gpu_ids

device = torch.device("cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu")

optimizer_type = "SGD"
assert optimizer_type in ["Adam", "SGD"]
dataset_type = "MPEG"
assert dataset_type in ["MPEG", "MN40"]


class DGCNNFilter(nn.Module):
    def __init__(self, fin, hidden_layers: List[int], activation: bool = True):
        super().__init__()
        self.fin = fin
        hidden_layers = [fin] + hidden_layers
        self.has_activation = activation
        self.mlps = nn.ModuleList([MLP(2 * i, o) for i, o in layers(hidden_layers)])
        for i, o in layers(hidden_layers):
            sprint("Created layer (%d, %d)" % (i, o))
        self.filters = nn.ModuleList([DynamicEdgeConv(mlp, k=32) for mlp in self.mlps])
        self.activation = nn.ModuleList(
            [
                nn.Sequential(nn.ReLU(), nn.BatchNorm1d(o))
                if idx != len(hidden_layers) - 2
                else nn.Identity()
                for idx, (i, o) in enumerate(layers(hidden_layers))
            ]
        )

    def forward(self, data):
        # print(data)
        target, batch, x = data.y, data.batch, data.x
        for i, (filter, activation) in enumerate(zip(self.filters, self.activation)):
            x = filter(x, batch=batch)
            if self.has_activation:
                x = activation(x)

        loss = mse(x, target)
        return x, loss


if __name__ == "__main__":
    timestamp = init_train(parallel, gpu_ids)
    print(colorama.Fore.RED + "Training on dataset %s" % dataset_type)
    if dataset_type == "MN40":
        model_name = "modelnet40-bf-64-128"
        model_path = os.path.join("model", model_name, str(timestamp))
        pl_path = "modelnet40-1024"
        data_path = os.path.join("/data", "pkurei", pl_path)
    elif dataset_type == "MPEG":
        model_name = "mpeg-monet-5.0sgd"
        model_path = os.path.join("model", model_name, str(timestamp))
        # pl_path = 'pku'
        data_path = os.path.join("data-5.0")

    for path in (data_path, model_path):
        check_dir(path, color=colorama.Fore.CYAN)

    model_milestone, optim_milestone, beg_epochs = (
        os.path.join("model", model_name, str(15), "model-latest.save"),
        os.path.join("model", model_name, str(15), "opt-latest.save"),
        20,
    )
    model_milestone, optim_milestone, beg_epochs = (
        None,
        None,
        0,
    )  # comment this if need to load from milestone

    # Select baseline network
    if re.search("dgcnn", model_name) is not None:
        model = DGCNNFilter(fin=6, hidden_layers=[64, 128, 6])
        batch_size = 24 * ngpu  # bs depends on GPUs used
    elif re.search("gat", model_name) is not None:
        model = GATDenoiser(
            fin=6,
            hidden_layers=[
                {"f": 16, "heads": 4},  # i.e. 64
                {"f": 64, "heads": 2},  # i.e. 128
                {"f": 6, "heads": 8, "concat": False, "negative_slope": 0.2}
                # cancel activation at last layer
            ],
        )
        batch_size = 40 * ngpu  # bs depends on GPUs used
    elif re.search("monet", model_name):
        model = MoNetDenoiser(
            fin=6, hidden_layers=[64, 128, 6], kernel_size=8, separate_gaussians=False
        )
        batch_size = 4 * ngpu
        # NOTE: 8 might be unfair.
    else:
        raise NotImplementedError

    if parallel and use_sbn:
        model = parallelize_model(model, device, gpu_ids, gpu_id)
    else:
        model = model.to(device)

    dataset, test_dataset, train_loader, test_loader = get_data(
        dataset_type,
        data_path,
        batch_size=batch_size,
        samplePoints=samplePoints,
        parallel=parallel,
    )
    # print(train_loader)

    writer = SummaryWriter(comment=model_name)

    optimizer, scheduler = get_optimizer(
        model, optimizer_type, (), 0.002, 0.002, beg_epochs
    )

    if model_milestone is not None:
        load_model(model, optimizer, model_milestone, optim_milestone, beg_epochs)
    else:
        init_weights(model)

    batch_cnt = len(train_loader)
    print(
        colorama.Fore.MAGENTA
        + "Begin training with: batch size %d, %d batches in total"
        % (batch_size, batch_cnt)
    )

    for epoch in trange(beg_epochs, epochs + 1):
        train_mse, train_psnr, train_orig_psnr = train(
            model, optimizer, scheduler, train_loader, dataset_type, parallel, epoch
        )
        eval_mse, eval_psnr, test_orig_psnr = evaluate(
            model, test_loader, dataset_type, parallel, epoch
        )

        # save model for each <milestone_period> epochs (e.g. 10 rounds)
        if epoch % milestone_period == 0 and epoch != 0:
            torch.save(
                model.state_dict(), os.path.join(model_path, "model-%d.save" % (epoch))
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(model_path, "opt-%d.save" % (epoch)),
            )
            torch.save(
                model.state_dict(), os.path.join(model_path, "model-latest.save")
            )
            torch.save(
                optimizer.state_dict(), os.path.join(model_path, "opt-latest.save")
            )

        # log to tensorboard
        record_dict = {
            "train_mse": train_mse,
            "train_psnr": train_psnr,
            "test_mse": eval_mse,
            "test_psnr": eval_psnr,
            "train_orig_psnr": train_orig_psnr,
            "test_orig_psnr": test_orig_psnr,
        }

        for key in record_dict:
            if not isinstance(record_dict[key], dict):
                writer.add_scalar(key, record_dict[key], epoch)
            else:
                writer.add_scalars(key, record_dict[key], epoch)
                # add multiple records