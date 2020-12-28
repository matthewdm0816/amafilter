from tensorboardX import SummaryWriter
import time, random, os, sys, gc, copy, colorama, json
colorama.init(autoreset=True)
from tqdm import *
import numpy as np

from torch_geometric.nn import DataParallel
from torch_geometric.data import DataListLoader
import torch
import torch.optim as optim

from bf import BilateralFilter
from utils import *

dataset_type = '40'
samplePoints = 1024
# batch_size = 32  # largest affordable on GTX1080 with 8G VRAM quq, even with sparse
epochs = 1001
milestone_period = 10

gpu_id = 7
# gpu_ids = [0, 1, 2, 7]
gpu_ids = [7]
ngpu = len(gpu_ids)
# os.environ['CUDA_VISIBLE_DEVICES'] = repr(gpu_ids)[1:-1]
parallel = (ngpu > 1) 
assert gpu_id in gpu_ids


model_milestone, optim_milestone, beg_epochs = \
    '/data/pkurei/PointNet/model/modelnet40-dense-gcn-3%%25noise-smooth-label-weighted-instance-rot-10deg-10%%25rescale-ensemble20/2020-12-11-15-20-26-4-346-0/model-latest.save', \
    '/data/pkurei/PointNet/model/modelnet40-dense-gcn-3%%25noise-smooth-label-weighted-instance-rot-10deg-10%%25rescale-ensemble20/2020-12-11-15-20-26-4-346-0/opt-latest.save', \
    30
model_milestone, optim_milestone, beg_epochs = None, None, 0 # comment this if need to load from milestone
    

def train(model, optimizer, scheduler, loader, epoch: int):
    """
    NOTE: Need DROP_LAST=TRUE, in case batch length is not uniform
    """
    model.train()
    total_psnr, total_mse = 0, 0
    for i, batch in enumerate(loader, 0):
        if parallel:
            reals, bs = parallel_cuda(batch)
        else:
            batch = batch.to(device)
            reals, bs = batch.y, batch.y.shape[0]
        
        model.zero_grad()
        out = model(batch)

        loss = mse(out, reals)
        psnr = mse_to_psnr(loss)
        total_psnr += psnr.detach().item()
        total_mse += loss.detach().item()
        
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("[%d/%d]MSE: %.3f, PSNR: %.3f" % (epoch, i, loss.item(), psnr.item()))
    scheduler.step()
    total_mse /= len(loader)
    total_psnr /= len(loader)
    return total_mse, total_psnr

def evaluate(model, loader, epoch: int):
    """
    NOTE: Need DROP_LAST=TRUE, in case batch length is not uniform
    """
    model.evaluate()
    total_psnr, total_mse, orig_psnr = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader, 0):
            if parallel:
                reals, bs = parallel_cuda(batch)
            else:
                batch = batch.to(device)
                reals, bs = batch.y, batch.y.shape[0]
            
            orig_psnr = psnr(batch, reals)
            model.zero_grad()
            out = model(batch)

            loss = mse(out, reals)
            psnr = mse_to_psnr(loss)
            total_psnr += psnr.detach().item()
            total_mse += loss.detach().item()

    total_mse /= len(loader)
    total_psnr /= len(loader)
    orig_psnr /= len(loader)
    print(colorama.Fore.MAGENTA + "[%d]MSE: %.3f, PSNR: %.3f, PSNR0: %.3f" % (epoch, total_mse, total_psnr, orig_psnr))
    return total_mse, total_psnr, orig_psnr

if __name__ == "__main__":
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
    model_path = os.path.join('model', model_name, timestamp)
    pl_path = 'modelnet40-knn'
    data_path = os.path.join('/data1/', pl_path)

    for path in (data_path, model_path):
        check_dir(path, color=colorama.Fore.CYAN)

    # model creation and initialization
    model = None
    # TODO: ...
    

