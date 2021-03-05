"""
Bilateral Filter Training
TODO: 
    1. Test on modelnet40
    2. Implement on MPEG large dataset
    3. Implement parallel training
    4. Learn displacement vector rather than filtered position
    5. Calculate Std. Dev. => Impl. 10-30 std. jitter
"""

from tensorboardX import SummaryWriter
import time, random, os, sys, gc, copy, colorama, json
colorama.init(autoreset=True)
from tqdm import *
import numpy as np

from torch_geometric.nn import DataParallel
from torch_geometric.data import DataListLoader, DataLoader
from torch_geometric.datasets import ModelNet
import torch
import torch.optim as optim

from bf import AmaFilter
from utils import *

dataset_type = '40'
samplePoints = 1024
batch_size = 20
epochs = 1001
milestone_period = 5
use_sbn = True

gpu_id = 0
# gpu_ids = [0, 1, 2, 7]
gpu_ids = [0, 1]
ngpu = len(gpu_ids)
# os.environ['CUDA_VISIBLE_DEVICES'] = repr(gpu_ids)[1:-1]
parallel = (ngpu > 1) 
assert gpu_id in gpu_ids

device = torch.device("cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu")

def train(model, optimizer, scheduler, loader, epoch: int):
    """
    NOTE: Need DROP_LAST=TRUE, in case batch length is not uniform
    """
    model.train()

    # show current lr
    print(colorama.Fore.GREEN + "Current LR: %.5f" % optimizer.param_groups[0]['lr'])
    
    total_psnr, total_mse = 0, 0
    for i, batch in enumerate(loader, 0):
        # torch.cuda.empty_cache()
        if parallel:
            batch, reals, _ = parallel_cuda(batch, device)
            jittered = [
                add_multiplier_noise(real.detach(), multiplier=5)
                for real in reals
            ]
            # TODO: parallel add noise
        else:
            # print(batch)
            batch = batch.to(device)
            reals = batch.pos
            jittered = add_multiplier_noise(reals.detach(), multiplier=5)
        
        orig_mse = mse(jittered, reals)
        orig_psnr = mse_to_psnr(orig_mse)
        model.zero_grad()
        out = model(jittered, batch=batch.batch, k=32)

        loss = mse(out, reals)
        psnr_loss = mse_to_psnr(loss)
        total_psnr += psnr_loss.detach().item()
        total_mse += loss.detach().item()
        
        loss.backward()
        optimizer.step()
        # del jittered, reals
        if i % 10 == 0:
            print(colorama.Fore.MAGENTA + "[%d/%d]MSE: %.3f, MSE-ORIG: %.3f, PSNR: %.3f, PSNR-ORIG: %.3f" % 
                (epoch, i, 
                loss.detach().item(), 
                orig_mse.detach().item(),
                psnr_loss.detach().item(), 
                orig_psnr.detach().item()
                )
            )
    scheduler.step()
    total_mse /= len(loader)
    total_psnr /= len(loader)
    return total_mse, total_psnr

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
                batch, reals, _ = parallel_cuda(batch, device)
                jittered = [
                    add_multiplier_noise(real.detach(), multiplier=5)
                    for real in reals
                ]
                # TODO: parallel add noise
            else:
                batch = batch.to(device)
                reals = batch.pos
                jittered = add_multiplier_noise(reals.detach(), multiplier=5)
            
            orig_mse = mse(jittered, reals)
            orig_psnr = mse_to_psnr(orig_mse)
            out = model(jittered, batch=batch.batch, k=32)

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

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    print(colorama.Fore.MAGENTA + (
        "Running in Single-GPU mode" if not parallel else 
        "Running in Multiple-GPU mode")
    )

    # training identifier
    try:
        with open('timestamp.json', 'r') as f:
            timestamp = json.load(f)["timestamp"] + 1
    except FileNotFoundError:
        # init timestamp
        timestamp = 1
    finally:
        # save timestamp
        with open('timestamp.json', 'w') as f:
            json.dump({
                    "timestamp": timestamp
                },
                f
            )

    # model and data path
    model_name = 'modelnet40-bf'
    model_path = os.path.join('model', model_name, str(timestamp))
    pl_path = 'modelnet40-1024'
    data_path = os.path.join('/data', 'pkurei', pl_path)

    for path in (data_path, model_path):
        check_dir(path, color=colorama.Fore.CYAN)

    # dataset and dataloader
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

    # tensorboard writer
    writer = SummaryWriter(comment=model_name)  # global steps => index of epoch

    # load model or init model
    model_milestone, optim_milestone, beg_epochs = \
        os.path.join('model', model_name, str(15), 'model-latest.save'), \
        os.path.join('model', model_name, str(15), 'opt-latest.save'), \
        20
    # model_milestone, optim_milestone, beg_epochs = None, None, 0 # comment this if need to load from milestone

    # model, optimizer, scheduler declaration
    model = AmaFilter(3, 3).to(device)
    
    # parallelization
    if parallel:
        model = DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_id)
        if use_sbn:
            try:
                from .sync_batchnorm import convert_model
                model = convert_model(model)
                # fix sync-batchnorm
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Sync-BN plugin not found")
    else:   
        model = model.to(device)

    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': 0.002}],
                                lr=0.002, weight_decay=5e-4, betas=(0.9, 0.999))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.65, last_epoch=beg_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, last_epoch=beg_epochs)
    

    if model_milestone is not None:
        load_model(model, optimizer, model_milestone, optim_milestone, beg_epochs)
    else:
        init_weights(model)

    batch_cnt = len(train_loader)
    print(colorama.Fore.MAGENTA + "Begin training with: batch size %d, %d batches in total" % (batch_size, batch_cnt))

    for epoch in trange(beg_epochs, epochs + 1):
        train_mse, train_psnr = train(model, optimizer, scheduler, train_loader, epoch)
        eval_mse, eval_psnr, orig_psnr = evaluate(model, test_loader, epoch)
        
        # save model for each <milestone_period> epochs (e.g. 10 rounds)
        if epoch % milestone_period == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join(model_path, 'model-%d.save' % (epoch)))
            torch.save(optimizer.state_dict(), os.path.join(model_path, 'opt-%d.save' % (epoch)))
            torch.save(model.state_dict(), os.path.join(model_path, 'model-latest.save'))
            torch.save(optimizer.state_dict(), os.path.join(model_path, 'opt-latest.save'))
        
        # log to tensorboard
        record_dict = {
            'train_mse': train_mse,
            'train_psnr': train_psnr,
            'test_mse': eval_mse,
            'test_psnr': eval_psnr,
        }

        for key in record_dict:
            if not isinstance(record_dict[key], dict):
                writer.add_scalar(key, record_dict[key], epoch)
            else: 
                writer.add_scalars(key, record_dict[key], epoch) 
                # add multiple records
    

