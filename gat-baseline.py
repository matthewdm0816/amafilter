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
        {"f":6, "heads":8, "concat":False, "negative_slope"=1.0}
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

    def forward(self, data):
        # print(data)
        target, batch, x = data.y, data.batch, data.x
        for i, filter in enumerate(self.filters):
            edge_index = knn_graph(x, k=32, batch=batch, loop=False)
            x = filter(x, edge_index=edge_index, batch=batch)

        loss = mse(x, target)
        return x, loss

timestamp = init_train(parallel, gpu_ids)
    print(colorama.Fore.RED + "Training on dataset %s" % dataset_type)
    if dataset_type == "MN40":
        model_name = "modelnet40-bf-64-128"
        model_path = os.path.join("model", model_name, str(timestamp))
        pl_path = "modelnet40-1024"
        data_path = os.path.join("/data", "pkurei", pl_path)
    elif dataset_type == "MPEG":
        model_name = "mpeg-dgcnn-5.0v3sgd"
        model_path = os.path.join("model", model_name, str(timestamp))
        # pl_path = 'pku'
        data_path = os.path.join("data-5.0")

    for path in (data_path, model_path):
        check_dir(path, color=colorama.Fore.CYAN)

    dataset, test_dataset, train_loader, test_loader = get_data(
        dataset_type,
        data_path,
        batch_size=batch_size,
        samplePoints=samplePoints,
        parallel=parallel,
    )
    print(train_loader)

    writer = SummaryWriter(comment=model_name)

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

    model = DGCNNFilter(6, hidden_layers=[64, 128, 6])
    if parallel and use_sbn:
        model = parallelize_model(model, device, gpu_ids, gpu_id)
    else:
        model = model.to(device)

    # print(colorama.Fore.RED + "Using optimizer type %s" % optimizer_type)
    # if optimizer_type == "Adam":
    #     optimizer = optim.Adam(
    #         [
    #             {"params": model.parameters(), "initial_lr": 0.002},
    #         ],
    #         lr=0.002,
    #         weight_decay=5e-4,
    #         betas=(0.9, 0.999),
    #     )
    # elif optimizer_type == "SGD":
    #     # Using SGD Nesterov-accelerated with Momentum
    #     optimizer = optim.SGD(
    #         [
    #             {"params": model.parameters(), "initial_lr": 0.002},
    #         ],
    #         lr=0.002,
    #         weight_decay=5e-4,
    #         momentum=0.9,
    #         nesterov=True,
    #     )
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=100, last_epoch=beg_epochs
    # )
    optimizer, scheduler = get_optimizer(
        model, optimizer_type, [], 0.002, 0.002, beg_epochs
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