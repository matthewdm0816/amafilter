r"""
Bilateral Filter Training
TODO: 
    1. -Test on modelnet40-
    2. -Implement on MPEG large dataset-
    3. -Implement parallel training-
    4. Learn displacement vector rather than filtered position
    5. -Calculate Std. Dev. => Impl. 10-30 std. jitter-
    6. Use specific channels for loss calc. (i.e. color only)
    7. Impl. further benchmark metrics
    8. Try alternative optimizers(esp. SGD)
    9. -Why MSE mismatch? :NaN data!-
    10. Smaller/faster model
    11. Re-generate dataset
    12. Re-split dataset in in-class fashion
    13. Add attention? 
    14. Patch aggregation (and test)
"""

from tensorboardX import SummaryWriter
import time, random, os, sys, gc, copy, colorama, json, re

colorama.init(autoreset=True)
from tqdm import *
import numpy as np
import pretty_errors
from pprint import pprint

from torch_geometric.nn import DataParallel
from torch_geometric.data import DataListLoader, DataLoader
from torch_geometric.datasets import ModelNet
import torch
import torch.optim as optim

from bf import AmaFilter, BilateralFilterv2, BilateralFilter
from dataloader import MPEGDataset, ADataListLoader, MPEGTransform
from utils import *

# dataset_type = "40"
samplePoints = 1024
epochs = 1001
milestone_period = 5
use_sbn = True
gpu_id = 0
# gpu_ids = [0, 1, 2, 7]
gpu_ids = [0, 1, 2, 3, 4, 5]
ngpu = len(gpu_ids)
batch_size = 24 * ngpu  # bs depends on GPUs used
# os.environ['CUDA_VISIBLE_DEVICES'] = repr(gpu_ids)[1:-1]
parallel = ngpu > 1
assert gpu_id in gpu_ids

device = torch.device("cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu")

optimizer_type = "SGD"
assert optimizer_type in ["Adam", "SGD"]
dataset_type = "MPEG"
assert dataset_type in ["MPEG", "MN40"]
bfilter = BilateralFilterv2


def process_batch(batch, parallel, dataset_type):
    if parallel:
        # FIXME: concat again after one epoch
        # NOTE: must need to make full clone
        result_batch = [data.clone() for data in batch]
    else:
        # make a result_batch for non-parallel runs
        result_batch = batch.clone()
    if dataset_type == "MN40":
        if parallel:
            batch = parallel_cuda(batch, device)
            reals = batch.pos
            jittered = [
                add_multiplier_noise(real.detach(), multiplier=5).to(real)
                for real in reals
            ]
            orig_mse = torch.tensor(
                [mse(jitter, real) for jitter, real in zip(jittered, reals)]
            ).mean()

            # NOTE: concat real/jitter image
            for i, (data, jitter) in enumerate(zip(batch, jittered)):
                result_batch[i].x, result_batch[i].y = jitter, data.pos
        else:
            batch = batch.to(device)
            reals = batch.pos
            jittered = add_multiplier_noise(reals.detach(), multiplier=5)
            orig_mse = mse(jittered, reals)
            # result_batch = batch.clone()
            result_batch.x, result_batch.y = jittered, batch.pos
    elif dataset_type == "MPEG":
        if parallel:  # only paraller loader impl.ed
            batch = parallel_cuda(batch, device)
            # in MPEG, Data(x, y, pos, label) ~ noised C/orig C/noised C-cat-orig P
            orig_mse = torch.tensor([mse(data.x, data.y) for data in batch]).mean()
            for i, data in enumerate(batch):
                # NOTE: don't modify batch itself! it'll change the damn dataset!
                result_batch[i].x = torch.cat([data.x, data.z], dim=-1)
                result_batch[i].y = torch.cat([data.y, data.z], dim=-1)
        else:
            orig_mse = mse(batch.x, batch.y)
            result_batch.x = torch.cat([batch.x, batch.y], dim=-1)
            result_batch.y = torch.cat([batch.y, batch.z], dim=-1)

    # print(result_batch[0].x.shape, result_batch[0].y.shape)
    return result_batch, orig_mse


def train(
    model, optimizer, scheduler, loader, dataset_type, parallel: bool, epoch: int
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
        out, loss = model(batch)
        loss = loss.mean()
        # print(out.shape, loss.shape)

        psnr_loss = mse_to_psnr(loss)
        # total_orig_mse = orig_mse.detach().mean
        total_psnr += psnr_loss.detach().item()
        total_mse += loss.detach().item()
        total_orig_psnr += orig_psnr.detach().item()

        loss.backward()
        optimizer.step()
        # del jittered, reals
        if i % 10 == 0:
            print(
                colorama.Fore.MAGENTA
                + "[%d/%d]MSE: %.3f, MSE-ORIG: %.3f, PSNR: %.3f, PSNR-ORIG: %.3f"
                % (
                    epoch,
                    i,
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


def evaluate(model, loader, dataset_type, parallel: bool, epoch: int):
    """
    NOTE: Need DROP_LAST=TRUE, in case batch length is not uniform
    """
    # global dataset_type
    model.eval()
    total_psnr, total_mse, total_orig_psnr = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader, 0):
            # torch.cuda.empty_cache()
            batch, orig_mse = process_batch(batch, parallel, dataset_type)

            orig_psnr = mse_to_psnr(orig_mse)
            out, loss = model(batch)
            loss = loss.mean()

            # loss = mse(out, reals)
            psnr_loss = mse_to_psnr(loss)
            total_orig_psnr += orig_psnr.detach().item()
            total_psnr += psnr_loss.detach().item()
            total_mse += loss.detach().item()

    total_mse /= len(loader)
    total_psnr /= len(loader)
    total_orig_psnr /= len(loader)
    print(
        colorama.Fore.MAGENTA
        + "[%d]MSE: %.3f, PSNR: %.3f, PSNR-ORIG: %.3f"
        % (epoch, total_mse, total_psnr, total_orig_psnr)
    )
    return total_mse, total_psnr, orig_psnr


if __name__ == "__main__":
    timestamp = init_train(parallel, gpu_ids)

    # model and data path
    print(colorama.Fore.RED + "Training on dataset %s" % dataset_type)
    if dataset_type == "MN40":
        model_name = "modelnet40-bf-64-128"
        model_path = os.path.join("model", model_name, str(timestamp))
        pl_path = "modelnet40-1024"
        data_path = os.path.join("/data", "pkurei", pl_path)
    elif dataset_type == "MPEG":
        model_name = "mpeg-bf-5.0v3sgd+act"
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

    # tensorboard writer
    writer = SummaryWriter(comment=model_name)  # global steps => index of epoch

    # load model or init model
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

    model = get_model(
        dataset_type,
        bfilter,
        device,
        activation=True,
        parallel=parallel,
        use_sbn=True,
        gpu_ids=gpu_ids,
        gpu_id=gpu_id,
    )

    # show named modules
    # for name, param in model.named_parameters():
    #     print(name)
    # exit(0)

    # optimizer & scheduler
    my_list = [
        "module.filters.0.embedding",
        "module.filters.1.embedding",
        "module.filters.2.embedding",
    ]
    optimizer, scheduler = get_optimizer(
        model, optimizer_type, my_list, 0.002, 0.002 * 10, beg_epochs
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
