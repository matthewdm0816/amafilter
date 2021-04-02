# from os import O_PATH
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_geometric as tg
from torch_geometric.data import DataLoader, DataListLoader, InMemoryDataset, Data
from torch_geometric.nn import knn_graph, radius_graph, MessagePassing
import colorama, json
from typing import Optional
from collections import OrderedDict
import pretty_errors
from icecream import ic
from tqdm import tqdm, trange

colorama.init(autoreset=True)


def transform(samplePoints=2500, k=32):
    def f(data):
        data = tg.transforms.NormalizeScale()(data)  # normalize to [-1, 1]
        data = tg.transforms.SamplePoints(samplePoints)(data)
        # data = tg.transforms.KNNGraph(k=k)(data)
        return data

    return f


def check_dir(path, color=None):
    """
    check directory if avaliable
    """
    import os, colorama

    if not os.path.exists(path):
        print("" if color is None else color + "Creating path %s" % path)
        os.makedirs(path, exist_ok=True)


def add_noise(pc, scale=0.01):
    """
    add gaussian noise
    """
    noise = torch.randn(pc.shape).to(pc) * scale
    return pc + noise


def add_multiplier_noise(pc, multiplier=10):
    """
    add gaussian noise according to standard deviation
    """
    std = torch.std(pc)
    # print(std)
    return pc + torch.randn(pc.shape).to(pc) * std * multiplier


def color_jitter(pos, color, scale=0.01):
    """
    wrapped jitterer on pc(pos + color)
    """
    color = add_noise(color)
    return torch.cat(pos, color, dim=-1)  # concat on last dim


def mse(fake, real):
    return F.mse_loss(fake, real, reduction="mean")


def psnr(fake, real, max=1.0):
    # fake, real: N * FIN
    return mse_to_psnr(mse(fake, real), max=max)


def mse_to_psnr(mse, max=1.0):
    return 10 * torch.log10(max ** 2 / mse)


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def load_model(model, optimizer, f: str, optim: str, e: int, evaluate=None):
    loaded = torch.load(f)
    try:
        loaded = loaded.module
        ic(loaded)
    except:
        ic()
        pass
    model.load_state_dict(loaded)
    print("Loaded milestone with epoch %d at %s" % (e, f))
    if optim is not None:
        optimizer.load_state_dict(torch.load(optim))
        print("Loaded milestone optimizer with epoch %d at %s" % (e, optim))
    if evaluate is not None:
        evaluate(model)


class layers:
    def __init__(self, ns):
        self.ns = ns
        self.iter1 = iter(ns)
        self.iter2 = iter(ns)  # iterator of latter element
        next(self.iter2)

    def __iter__(self):
        return self

    def __next__(self):
        return (next(self.iter1), next(self.iter2))


def module_wrapper(f):
    class module(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return f(x)

    return module()


def parallel_cuda(batch, device):
    for i, data in enumerate(batch):
        for key in data.keys:
            if torch.is_tensor(data[key]):
                data[key].to(device)
        batch[i] = data
    return batch


def tensorinfo(t):
    return "%f, %f, %f" % (t.max().item(), t.median().item(), t.min().item())


def get_data(dataset_type, data_path, batch_size=32, samplePoints=1024, parallel=False):
    from dataloader import ADataListLoader, MPEGDataset, MPEGTransform
    from torch_geometric.data import Data, DataLoader, DataListLoader
    from torch_geometric.datasets import ModelNet

    if dataset_type == "MN40":
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
                shuffle=True,
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
        return train_dataset, test_dataset, train_loader, test_loader
    elif dataset_type == "MPEG":
        dataset = MPEGDataset(root=data_path, pre_transform=MPEGTransform)
        if parallel:
            train_loader = ADataListLoader(
                dataset,
                training=True,
                test_classes=[0, 1],
                batch_size=batch_size,
                shuffle=True,
            )
            test_loader = ADataListLoader(
                dataset,
                training=False,
                test_classes=[0, 1],
                batch_size=batch_size,
                shuffle=True,
            )
        else:
            raise NotImplementedError
        return dataset, dataset, train_loader, test_loader


def init_train(parallel, gpu_ids):
    torch.backends.cudnn.benchmark = True
    print(
        colorama.Fore.MAGENTA
        + (
            "Running in Single-GPU mode"
            if not parallel
            else "Running in Multiple-GPU mode with GPU {}".format(gpu_ids)
        )
    )

    # load timestamp
    try:
        with open("timestamp.json", "r") as f:
            timestamp = json.load(f)["timestamp"] + 1
    except FileNotFoundError:
        # init timestamp
        timestamp = 1
    finally:
        # save timestamp
        with open("timestamp.json", "w") as f:
            json.dump({"timestamp": timestamp}, f)

    return timestamp


def parallelize_model(model, device, gpu_ids, gpu_id):
    # parallelization load
    # if parallel:
    #     if use_sbn:
    from torch_geometric.nn import DataParallel

    try:
        # fix sync-batchnorm
        from sync_batchnorm import convert_model

        model = convert_model(model)
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Sync-BN plugin not found")
    # NOTE: DataParallel call MUST after model definition completes
    model = DataParallel(model, device_ids=gpu_ids, output_device=gpu_id).to(device)
    # else:
    #     model = model.to(device)
    print("Parallelized model")
    return model


def get_model(
    dataset_type: str,
    bfilter,
    device,
    batch_size: int,
    activation: bool = True,
    parallel: bool = False,
    use_sbn: bool = True,
    gpu_ids=(0,),
    gpu_id=0,
    reg: float = 0.0,
    loss_type: str = "mse",
    collate: str = "gaussian",
):
    from bf import AmaFilter
    from torch_geometric.nn import DataParallel
    from utils import mse, chamfer_measure

    assert loss_type in ["mse", "chamfer"]

    if dataset_type == "MN40":
        model = AmaFilter(
            3,
            3,
            k=32,
            activation=activation,
            filter=bfilter,
            reg=reg,
            loss_type=loss_type,
            collate=collate,
        )
    elif dataset_type == "MPEG":
        model = AmaFilter(
            6,
            6,
            k=32,
            filter=bfilter,
            activation=activation,
            reg=reg,
            loss_type=loss_type,
            merge_embedding=False,
            collate=collate,
        )
        print(colorama.Fore.MAGENTA + "Using filter type %s" % bfilter.__name__)

    if parallel and use_sbn:
        model = parallelize_model(model, device, gpu_ids, gpu_id)
    else:
        model = model.to(device)
    return model


def get_optimizer(model, optimizer_type, my_list, lr, alt_lr, beg_epochs):
    from torch import optim
    import re

    print(colorama.Fore.RED + "Using optimizer type %s" % optimizer_type)
    if optimizer_type == "Adam":
        optimizer = optim.Adam(
            [
                {"params": model.parameters(), "initial_lr": lr},
                # {"params": model.parameters(), "initial_lr": 0.002}
            ],
            lr=lr,
            weight_decay=5e-4,
            betas=(0.9, 0.999),
        )
    elif optimizer_type == "SGD":
        # Using SGD Nesterov-accelerated with Momentum
        # Selective lr adjustment
        params = list(
            map(
                lambda x: x[1],
                list(
                    filter(
                        lambda kv: any(
                            [
                                re.search(pattern, kv[0]) is not None
                                for pattern in my_list
                            ]
                        ),
                        model.named_parameters(),
                    )
                ),
            )
        )
        base_params = list(
            map(
                lambda x: x[1],
                list(
                    filter(
                        lambda kv: all(
                            [re.search(pattern, kv[0]) is None for pattern in my_list]
                        ),
                        model.named_parameters(),
                    )
                ),
            )
        )
        optimizer = optim.SGD(
            [
                {
                    "params": params,
                    "initial_lr": alt_lr,
                },
                {
                    "params": base_params,
                    "initial_lr": lr,
                },
            ],
            lr=0.002,
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,
        )

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=100, last_epoch=beg_epochs
    # )
    # Cosine annealing with restarts
    # etc.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=1, last_epoch=beg_epochs, eta_min=1e-6
    )
    return optimizer, scheduler


def parse_config(args):
    return (
        args.optimizer,
        args.dataset,
        args.gpus,
        args.gpus[0],
        len(args.gpus),  # ngpu
        len(args.gpus) > 1,  # parallel
        args.total,
        args.model,
        args.path,
        args.regularization,
        args.loss,
        torch.device("cuda:%d" % args.gpus[0] if torch.cuda.is_available() else "cpu"),
        args.batchsize * len(args.gpus),
        args.collate,
    )


def chamfer_measure(fake, real, batch):
    # use code module from https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
    # Using 6D chamfer dist. ~ RGB+XYZ
    from ChamferDistancePytorch.chamfer6D import dist_chamfer_6D

    batch_size = batch.max() + 1
    chamloss = dist_chamfer_6D.chamfer_6DDist()
    d1, d2, _, _ = chamloss(fake.view(batch_size, -1, 6), real.view(batch_size, -1, 6))
    return (d1 + d2).mean()


def get_ad_optimizer(
    models, optimizer_type: str, lr: float = 2e-3, beg_epochs: int = 0
):
    from torch import optim

    print(colorama.Fore.RED + "Using optimizer type %s" % optimizer_type)
    if optimizer_type == "Adam":
        gen, den = models
        gen_opt = optim.Adam(
            [{"params": gen.parameters(), "initial_lr": lr}],
            lr=lr,
            weight_decay=5e-4,
        )
        den_opt = optim.Adam(
            [{"params": den.parameters(), "initial_lr": lr}],
            lr=lr,
            weight_decay=5e-4,
        )
    elif optimizer_type == "SGD":
        # Using SGD Nesterov-accelerated with Momentum
        gen, den = models
        gen_opt = optim.SGD(
            [{"params": gen.parameters(), "initial_lr": lr}],
            lr=lr,
            weight_decay=5e-4,
            momentum=0.9,
        )
        den_opt = optim.SGD(
            [{"params": den.parameters(), "initial_lr": lr}],
            lr=lr,
            weight_decay=5e-4,
            momentum=0.9,
        )
    gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        gen_opt, T_max=100, last_epoch=beg_epochs
    )
    den_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        den_opt, T_max=100, last_epoch=beg_epochs
    )
    return [gen_opt, den_opt], [gen_scheduler, den_scheduler]


class GraphMeaner(MessagePassing):
    # Take Average on single graph
    def __init__(self, *args, **kwargs):
        super().__init__(aggr="mean", *args, **kwargs)

    def message(self, x_j):
        return x_j

    def forward(self, x, pos, batch=None):
        edge_index = radius_graph(pos, r=1e-6, batch=batch, loop=True)
        return self.propagate(edge_index, x=x)


def pre_transform_with_records(data):
    from dataloader import whiten_with_records

    # in-place modify
    data.color, cmean, cstd = whiten_with_records(data.color)
    data.pos, pmean, pstd = whiten_with_records(data.pos)
    return (cmean, pmean), (cstd, pstd)


def process_whole(
    model,
    ply_path: str,
    noise_generator,
    sigma: float = 1.0,
    batch_size: int = 16,
    parallel: bool = False,
    ignore_denoise: bool = False,
    n_patch: int=1000,
    patch_size: int=2048
):
    r"""
    Process a whole PC
    """
    from mpeg_process import process_ply, read_mesh
    from train_bf import process_batch
    from sklearn.cluster import dbscan

    # 1. Read mesh and turn into patches
    # by assumption, a typical PC contains 1000k points
    orig_mesh = read_mesh(ply_path)
    n_pts = orig_mesh.color.shape[0]
    ic(orig_mesh.color.shape)
    print(colorama.Fore.MAGENTA + "Total points: %d @ %s" % (n_pts, ply_path))
    ic("Taking patches of ", n_patch)
    patches = process_ply(ply_path, n_patch=n_patch, k=patch_size)
    # saved patch_index info here
    n_pc = len(patches)
    patches = Data(
        color=torch.stack([d.color for d in patches]),
        pos=torch.stack([d.pos for d in patches]),
        patch_index=torch.stack([d.patch_index for d in patches]),  # [B, N]
    )

    # 1a. record STD+MEAN
    (cmean, pmean), (cstd, pstd) = pre_transform_with_records(patches)
    ic(cmean.shape, cstd.shape) 
    ic(pmean.shape, pstd.shape)  # assumbly [B, 1, F]

    color, pos = patches.color, patches.pos
    noise = noise_generator(color, sigma)
    noisy_color = noise + color
    assert not torch.any(torch.isnan(color)), "NaN detected!"
    # divide into list
    data_list = [Data(x=dx, y=dy, z=dz) for dx, dy, dz in zip(noisy_color, color, pos)]
    if parallel:
        loader = DataListLoader(data_list, batch_size=batch_size, shuffle=False)
    else:
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    # 2. denoise on patches
    opatches = []
    ic(len(loader))
    if not ignore_denoise:
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, total=len(loader)):
                batch, _ = process_batch(batch, parallel=parallel, dataset_type="MPEG")
                # ic(batch, batch.batch)
                if not parallel:
                    batch.batch = batch.batch.long()  # NOTE: why?
                out, *loss = model(batch)
                # compatibility to no mse out nets
                ic(loss)
                if len(loss) == 2:
                    loss, mse_loss = loss
                elif len(loss) == 1:
                    loss, mse_loss = loss[0], loss[0]
                opatches.append(out)  # [B_patch * N, F]
        opatches = torch.cat(opatches, dim=0)  # => [B * N, F]
        f_pc = opatches.shape[-1]
        opatches = opatches.view(n_pc, -1, f_pc)  # => [B, N, F]
    else:
        opatches = torch.randn([n_pc, 2048, 6])

    # 3. rebuild whole PC
    ocolor, opos = opatches[:, :, :3], opatches[:, :, 3:]
    ic(ocolor.shape, opos.shape)
    
    ocolor = ocolor.cpu()

    ocolor = ocolor * cstd + cmean
    noisy_color = noisy_color * cstd + cmean
    # use original position
    # opos = torch.from_numpy(orig_mesh.pos)
    # ocolor = ocolor.view(-1, f_pc)
    # opos = opos.view(-1, f_pc)

    reconstructed = torch.zeros_like(orig_mesh.color)
    reconstructed_cnt = torch.ones([reconstructed.shape[0]]) * 1e-7
    for patch_index, patch_color in zip(patches.patch_index, ocolor):
        reconstructed[patch_index] += patch_color
        reconstructed_cnt[patch_index] += 1.0

    ic(reconstructed_cnt.max(), reconstructed_cnt.min())
    # direct averaging
    # TODO: Use more accurate averaging
    ic(reconstructed.shape, reconstructed_cnt.shape)
    reconstructed = reconstructed / reconstructed_cnt.view(-1, 1)
    # 3b. rebuild noisy PC
    noisy = torch.zeros_like(orig_mesh.color)
    noisy_cnt = torch.ones([noisy.shape[0]]) * 1e-7
    for patch_index, patch_color in zip(patches.patch_index, noisy_color):
        noisy[patch_index] += patch_color
        noisy_cnt[patch_index] += 1.0
    # direct averaging 
    # TODO: Use more accurate averaging
    noisy = noisy / noisy_cnt.view(-1, 1)

    # 4. calc MSE
    mse_error = torch.norm(reconstructed - orig_mesh.color)

    return reconstructed, noisy, orig_mesh, mse_error
