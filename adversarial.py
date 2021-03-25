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
from bf import *
from gat_baseline import BaseDenoiser
from tensorboardX import SummaryWriter
from train_bf import process_batch

import os, sys, random, math, json

scaf = Scaffold()
scaf.debug()
sprint = scaf.print
warn = scaf.warn


class AdversarialDenoiser(nn.Module):
    def __init__(self, fin, hidden_layers= Optional[list]=None **kwargs):
        r"""
        Use mirrored structure to adversarially propose noise/denoise

        Maybe TODO: shared parameters!
        """
        super().__init__()
        # TODO: Add HL param to AmaFilter
        # self.hidden_layers = [fin] + hidden_layers
        # hidden_layers shall be symmetric
        self.denoiser = AmaFilter(
            fin=fin, fout=fin, k=32, filter=BilateralFilterv2, activation=True
        )
        self.generator = AmaFilter(
            fin=fin, fout=fin, k=32, filter=BilateralFilterv2, activation=True
        )
        self.critic = AmaFilter(
            fin=fin, fout=32, k=32, filter=BilateralFilterv2, activation=True
        )
        self.determine = tgnn.global_add_pool
        self.critic_mlp = MLP(fin=32, fout=1, activation=nn.Sigmoid)

    def critics(self, data):
        target, batch, x = data.y, data.batch, data.x
        x = self.critic(x)
        x = self.determine(x, batch)
        x = self.critic_mlp(x)

    def forward(self, data):
        target, batch, x = data.y, data.batch, data.x

        # Discriminate reals
        y_real = self.critics(data)

        # Generator
        gen, _, gen_mse = self.generator(data)
        gen_data = data.copy()  # NOTE: Must copy
        gen_data.x = gen
        y_gen = self.critics(gen_data)

        # Denoiser
        recon, _, recon_mse = self.denoiser(data)
        recon_data = data.copy()
        recon_data.x = recon
        y_recon = self.critics(recon_data)

        return (
            gen,
            recon,
            gen_mse,
            recon_mse,
            y_real,
            y_gen,
            y_recon,
        )

class AdversarialDenoiserv2(nn.Module):
    def __init__(self, fin, hidden_layers= Optional[list]=None **kwargs):
        r"""
        Use mirrored structure to adversarially propose noise/denoise

        Maybe TODO: shared parameters!
        """
        super().__init__()
        # TODO: Add HL param to AmaFilter
        # self.hidden_layers = [fin] + hidden_layers
        # hidden_layers shall be symmetric
        self.denoiser = AmaFilter(
            fin=fin, fout=fin, k=32, filter=BilateralFilterv2, activation=True
        )
        self.generator = AmaFilter(
            fin=fin, fout=fin, k=32, filter=BilateralFilterv2, activation=True
        )

    def forward_gen(self, data):
        # Generator
        target, batch, x = data.y, data.batch, data.x
        gen, _, gen_mse = self.generator(data)
        gen_data = data.copy()  # NOTE: Must copy
        gen_data.x = gen
        return gen, gen_mse

    def forward_den(self, data):
        # Denoiser
        target, batch, x = data.y, data.batch, data.x
        recon, _, recon_mse = self.denoiser(data)
        recon_data = data.copy()
        recon_data.x = recon
        return recon, recon_mse

    def forward(self, data):
        target, batch, x = data.y, data.batch, data.x

        gen, gen_mse = self.forward_gen(data)
        gen_data = data.copy()
        gen_data.x = gen

        recon, recon_mse = self.forward_den(data)
        


def train_ad(
    model, optimizers, schedulers, loader, dataset_type, parallel: bool, epoch: int
):
    """
    NOTE: Need DROP_LAST=TRUE, in case batch length is not uniform
    """
    # global dataset_type
    model.train()

    # show current lr
    print(colorama.Fore.GREEN + "Current LR: %.3E" % optimizer.param_groups[0]["lr"])

    total_psnr, total_mse, total_orig_psnr = 0, 0, 0
    for i, batch in tqdm(enumerate(loader, 0), total=len(loader)):
        # torch.cuda.empty_cache()
        batch, orig_mse = process_batch(batch, parallel, dataset_type)

        orig_psnr = mse_to_psnr(orig_mse)
        model.zero_grad()
        


        loss.backward()
        optimizer.step()
        # del jittered, reals
        if i % 10 == 0:
            print(
                colorama.Fore.MAGENTA
                + "[%d/%d]MSE: %.3f, LOSS: %.3f, MSE-ORIG: %.3f, PSNR: %.3f, PSNR-ORIG: %.3f"
                % (
                    epoch,
                    i,
                    mse_loss.detach().item(),
                    loss.detach().item(),
                    orig_mse.detach().item(),
                    psnr_loss.detach().item(),
                    orig_psnr.detach().item(),
                )
            )
    scheduler.step()
    total_mse /= len(loader)
    total_psnr /= len(loader)
    total_orig_psnr /= len(loader)
    return total_mse, total_psnr, total_orig_psnr

if __name__ == "__main__":
    r"""
    Test grad backprop on AdversarialDenoiser
    """
    model = AdversarialDenoiser(fin=6)

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
    batch_size = 8 * ngpu

    device = torch.device("cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu")

    optimizer_type = "SGD"
    assert optimizer_type in ["Adam", "SGD"]
    dataset_type = "MPEG"
    assert dataset_type in ["MPEG", "MN40"]
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

    if parallel and use_sbn:
        model = parallelize_model(model, device, gpu_ids, gpu_id)
    else:
        model = model.to(device)

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

    dataset, test_dataset, train_loader, test_loader = get_data(
        dataset_type,
        data_path,
        batch_size=batch_size,
        samplePoints=samplePoints,
        parallel=parallel,
    )
    # print(train_loader)
    

    writer = SummaryWriter(comment=model_name)

    optimizer, scheduler = get_ad_optimizer(
        model, optimizer_type, 0.002, beg_epochs
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
        # TODO: Train AdversarialDenoiser
        # train_mse, train_psnr, train_orig_psnr = train(
        #     model, optimizer, scheduler, train_loader, dataset_type, parallel, epoch
        # )
        # eval_mse, eval_psnr, test_orig_psnr = evaluate(
        #     model, test_loader, dataset_type, parallel, epoch
        # )

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