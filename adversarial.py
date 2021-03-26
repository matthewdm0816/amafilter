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
from train_bf import process_batch, copy_batch

import os, sys, random, math, json

scaf = Scaffold()
scaf.debug()
sprint = scaf.print
warn = scaf.warn


class AdversarialDenoiser(nn.Module):
    def __init__(self, fin, hidden_layers: Optional[list] = None):
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
    def __init__(self, fin, hidden_layers: Optional[list] = None):
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

    def forward(self, data):
        # Generator forward
        target, batch, x = data.y, data.batch, data.x
        recon, _, recon_mse = self.denoiser(data)
        return recon, recon_mse


class AdversarialGeneratorv2(nn.Module):
    def __init__(self, fin, hidden_layers: Optional[list] = None):
        r"""
        Use mirrored structure to adversarially propose noise/denoise

        Maybe TODO: shared parameters!
        """

        super().__init__()
        # TODO: Add HL param to AmaFilter
        # self.hidden_layers = [fin] + hidden_layers
        # hidden_layers shall be symmetric
        self.generator = AmaFilter(
            fin=fin, fout=fin, k=32, filter=BilateralFilterv2, activation=True
        )

    def forward(self, data):
        # Denoiser forward
        target, batch, x = data.y, data.batch, data.x
        gen, _, gen_mse = self.generator(data)
        return gen, gen_mse


class AdversarialGeneratorv3(nn.Module):
    def __init__(
        self, fin, hidden_layers: Optional[list] = None, additional_dim: int = 1
    ):
        r"""
        Use mirrored structure to adversarially propose noise/denoise
        Add random noise seed into noise generation
        Maybe TODO: shared parameters!
        """

        super().__init__()
        # TODO: Add HL param to AmaFilter
        # self.hidden_layers = [fin] + hidden_layers
        # hidden_layers shall be symmetric
        self.additional_dim = additional_dim
        self.generator = AmaFilter(
            fin=fin + additional_dim,
            fout=fin,
            k=32,
            filter=BilateralFilterv2,
            activation=True,
        )

    def forward(self, data):
        # Denoiser forward
        target, batch, x, noise = data.y, data.batch, data.x, data.noise
        data.x = torch.cat([x, noise], dim=-1)
        gen, _, gen_mse = self.generator(data)
        return gen, gen_mse


def train_ad(
    models,
    optimizers,
    schedulers,
    loader,
    dataset_type,
    parallel: bool,
    epoch: int,
    noise_dimensions: Optional[int] = None,
):
    """
    NOTE: Need DROP_LAST=TRUE, in case batch length is not uniform
    """
    # global dataset_type
    for model in models:
        model.train()

    gen_model, den_model = models
    gen_opt, den_opt = optimizers
    gen_scheduler, den_scheduler = schedulers

    # show current lr
    print(
        colorama.Fore.GREEN + "Current LR: %.3E" % optimizers[0].param_groups[0]["lr"]
    )

    total_loss_gen, total_mse_recon_G, total_mse_recon, total_mse = 0, 0, 0, 0
    for i, batch in tqdm(enumerate(loader, 0), total=len(loader)):
        # torch.cuda.empty_cache()
        batch, orig_mse = process_batch(batch, parallel, dataset_type)
        orig_psnr = mse_to_psnr(orig_mse)
        bs = len(batch)
        n_nodes, fout = batch[0].x.shape

        # 1. generate fake PCs
        gen_model.zero_grad()
        den_model.zero_grad()
        reverse_batch = copy_batch(batch)
        for data in reverse_batch:
            data.x, data.y = data.y, data.x
            # print(gen_model.modules.__class__.__name__)
            if noise_dimensions is not None:
                # add noise dimension
                # kind of "Elevation" on feature space
                for data in reverse_batch:
                    data.noise = torch.randn([n_nodes, noise_dimensions])
        x_gen, gen_mse = gen_model(reverse_batch)  # [B * N, F]
        fout = x_gen.shape[-1]
        x_gen = x_gen.view(bs, -1, fout)  # [B, N, F]
        gen_mse = gen_mse.mean()

        # 2. train generator
        # generated noisy PCs
        gen_batch = copy_batch(batch)
        for data, x_gen_data in zip(gen_batch, x_gen):
            # print(x_gen_data.shape)
            data.x = x_gen_data  # replace with generated noisy PC

        # old_gen_batch = copy_batch(gen_batch)  # save for later training in Denoiser
        _, recon_mse = den_model(gen_batch)
        recon_mse = recon_mse.mean()

        # L_gen = MSE_gen - MSE_recon
        # TODO: Try several loss form
        # like exp(L_den)/exp(L_gen)
        loss_gen = gen_mse + torch.log(
            1e-8 + torch.relu(1 - recon_mse / gen_mse)
        )  # using relu to prevent minus in log
        loss_gen.backward()
        gen_opt.step()

        # 3a. train denoiser on pre-generated noisy PC
        gen_model.zero_grad()
        den_model.zero_grad()
        _, mse_recon_G = den_model(batch)
        mse_recon_G = mse_recon_G.mean()

        # 3b. train denoiser on generated noisy PC
        reverse_batch = copy_batch(batch)
        for data in reverse_batch:
            data.x, data.y = data.y, data.x
            if noise_dimensions is not None:
                # add noise dimension
                # kind of "Elevation" on feature space
                # print("here")
                for data in reverse_batch:
                    data.noise = torch.randn([n_nodes, noise_dimensions])
        x_gen, _ = gen_model(reverse_batch)  # [B * N, F]
        fout = x_gen.shape[-1]
        x_gen = x_gen.view(bs, -1, fout)  # [B, N, F]

        gen_batch = copy_batch(batch)
        for data, x_gen_data in zip(gen_batch, x_gen):
            # print(x_gen_data.shape)
            data.x = x_gen_data  # replace with generated noisy PC

        _, mse_recon = den_model(gen_batch)
        mse_recon = mse_recon.mean()

        loss_den = mse_recon_G + mse_recon
        loss_den.backward()
        den_opt.step()

        for scheduler in schedulers:
            scheduler.step()

        # record to STDOUT
        psnr_loss = mse_to_psnr(loss_den / 2)
        # total_orig_mse = orig_mse.detach().mean
        total_loss_gen += loss_gen.detach().item()
        total_mse_recon_G += mse_recon_G.detach().item()
        total_mse_recon += mse_recon.detach().item()
        total_mse += loss_den.detach().item()
        if i % 10 == 0:
            print(
                colorama.Fore.MAGENTA
                + "[%d/%d]L_GEN: %.3f, MSE_G: %.3f, MSE_FAKE: %.3f, MSE-ORIG: %.3f, PSNR: %.3f, PSNR-ORIG: %.3f"
                % (
                    epoch,
                    i,
                    loss_gen.detach().item(),
                    mse_recon_G.detach().item(),
                    mse_recon.detach().item(),
                    orig_mse.detach().item(),
                    psnr_loss.detach().item(),
                    orig_psnr.detach().item(),
                )
            )
    result = (total_loss_gen, total_mse_recon_G, total_mse_recon, total_mse)
    result = [x / len(loader) for x in result]
    return result


def evaluate_ad(
    models,
    loader,
    dataset_type: str,
    parallel: bool,
    epoch: int,
    noise_dimensions: Optional[int] = None,
):
    for model in models:
        model.eval()
    gen_model, den_model = models
    total_loss_gen, total_mse_recon_G, total_mse_recon, total_mse = 0, 0, 0, 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader, 0), total=len(loader)):
            # torch.cuda.empty_cache()
            batch, orig_mse = process_batch(batch, parallel, dataset_type)
            orig_psnr = mse_to_psnr(orig_mse)
            bs = len(batch)
            n_nodes, _ = batch[0].x.shape

            # 1. generate fake PCs
            reverse_batch = copy_batch(batch)
            for data in reverse_batch:
                data.x, data.y = data.y, data.x
                if noise_dimensions is not None:
                    # add noise dimension
                    # kind of "Elevation" on feature space
                    for data in reverse_batch:
                        data.noise = torch.randn([n_nodes, noise_dimensions])
            x_gen, gen_mse = gen_model(reverse_batch)
            fout = x_gen.shape[-1]
            x_gen = x_gen.view(bs, -1, fout)  # [B, N, F]
            gen_mse = gen_mse.mean()

            # 2. test generator
            # make data for denoiser
            gen_batch = copy_batch(batch)
            for data, x_gen_data in zip(gen_batch, x_gen):
                data.x = x_gen_data  # replace with generated noisy PC
            _, recon_mse = den_model(gen_batch)
            recon_mse = recon_mse.mean()

            # L_gen = MSE_gen - MSE_recon
            # TODO: Try several loss form
            # like exp(L_den)/exp(L_gen)
            loss_gen = gen_mse + torch.log(
                1e-8 + torch.relu(1 - recon_mse / gen_mse)
            )  # using relu to prevent minus in log

            # 3a. train denoiser on pre-generated noisy PC
            _, mse_recon_G = den_model(batch)
            mse_recon_G = mse_recon_G.mean()

            # 3b. train denoiser on generated noisy PC
            gen_batch = copy_batch(gen_batch)
            _, mse_recon = den_model(gen_batch)
            mse_recon = mse_recon.mean()
            loss_den = mse_recon_G + mse_recon

            # record to STDOUT
            psnr_loss = mse_to_psnr(loss_den / 2)
            # total_orig_mse = orig_mse.detach().mean
            total_loss_gen += loss_gen.detach().item()
            total_mse_recon_G += mse_recon_G.detach().item()
            total_mse_recon += mse_recon.detach().item()
            total_mse += loss_den.detach().item()
    result = (total_loss_gen, total_mse_recon_G, total_mse_recon, total_mse)
    result = [x / len(loader) for x in result]
    print(
        colorama.Fore.MAGENTA
        + "[%d]L_GEN: %.3f, MSE_G: %.3f, MSE_FAKE: %.3f"
        % (
            epoch,
            result[0],
            result[1],
            result[2],
        )
    )
    return result


if __name__ == "__main__":
    r"""
    Test grad backprop on AdversarialDenoiser
    """
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
    batch_size = 12 * ngpu
    noise_dimensions = 1

    device = torch.device("cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu")

    optimizer_type = "Adam"
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
        model_name = "mpeg-ad-5.0sgd"
        model_path = os.path.join("model", model_name, str(timestamp))
        # pl_path = 'pku'
        data_path = os.path.join("data-5.0")

    for path in (data_path, model_path):
        check_dir(path, color=colorama.Fore.CYAN)

    models = [
        AdversarialGeneratorv3(fin=6, additional_dim=noise_dimensions),
        AdversarialDenoiserv2(fin=6),
    ]

    for i, model in enumerate(models):
        if parallel and use_sbn:
            models[i] = parallelize_model(model, device, gpu_ids, gpu_id)
        else:
            models[i] = model.to(device)
        init_weights(models[i])

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

    writer = SummaryWriter(comment=model_name)

    optimizers, schedulers = get_ad_optimizer(models, optimizer_type, 0.002, beg_epochs)

    batch_cnt = len(train_loader)
    print(
        colorama.Fore.MAGENTA
        + "Begin training with: batch size %d, %d batches in total"
        % (batch_size, batch_cnt)
    )

    for epoch in trange(beg_epochs, epochs + 1):
        train_loss_gen, train_mse_recon_G, train_mse_recon, train_mse = train_ad(
            models,
            optimizers,
            schedulers,
            train_loader,
            dataset_type="MPEG",
            parallel=parallel,
            epoch=epoch,
            noise_dimensions=noise_dimensions,
        )
        test_loss_gen, test_mse_recon_G, test_mse_recon, test_mse = evaluate_ad(
            models,
            test_loader,
            dataset_type="MPEG",
            parallel=parallel,
            epoch=epoch,
            noise_dimensions=noise_dimensions,
        )

        # save model for each <milestone_period> epochs (e.g. 10 rounds)
        if epoch % milestone_period == 0 and epoch != 0:
            torch.save(
                models[0].state_dict(),
                os.path.join(model_path, "model-gen-%d.save" % (epoch)),
            )
            torch.save(
                optimizers[0].state_dict(),
                os.path.join(model_path, "opt-gen-%d.save" % (epoch)),
            )
            torch.save(
                models[1].state_dict(),
                os.path.join(model_path, "model-den-%d.save" % (epoch)),
            )
            torch.save(
                optimizers[1].state_dict(),
                os.path.join(model_path, "opt-den-%d.save" % (epoch)),
            )
            torch.save(
                models[0].state_dict(),
                os.path.join(model_path, "model-gen-latest.save"),
            )
            torch.save(
                optimizers[0].state_dict(),
                os.path.join(model_path, "opt-gen-latest.save"),
            )
            torch.save(
                models[1].state_dict(),
                os.path.join(model_path, "model-den-latest.save"),
            )
            torch.save(
                optimizers[1].state_dict(),
                os.path.join(model_path, "opt-den-latest.save"),
            )

        # log to tensorboard
        record_dict = {
            "train_loss_gen": train_loss_gen,
            "train_mse_recon_G": train_mse_recon_G,
            "train_mse_recon": train_mse_recon,
            "train_mse": train_mse,
            "test_loss_gen": test_loss_gen,
            "test_mse_recon_G": test_mse_recon_G,
            "test_mse_recon": test_mse_recon,
            "test_mse": test_mse,
        }

        for key in record_dict:
            if not isinstance(record_dict[key], dict):
                writer.add_scalar(key, record_dict[key], epoch)
            else:
                writer.add_scalars(key, record_dict[key], epoch)
                # add multiple records