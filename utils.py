import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_geometric as tg
import colorama, json

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
    model.load_state_dict(torch.load(f))
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
    # batchs = [data.to(device) for data in batchs]
    # reals = [data.pos.to(device) for data in batchs]
    # # jittered = [ for real in reals]
    # labels = [data.y for data in batchs] # actually batchs
    # labels = torch.cat(labels).to(device)
    # bs = labels.shape[0]
    # return batchs, reals, labels
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
    return model


def get_model(
    dataset_type: str,
    bfilter,
    device,
    activation: bool = True,
    parallel: bool = False,
    use_sbn: bool = True,
    gpu_ids=(0,),
    gpu_id=0,
):
    from bf import AmaFilter
    from torch_geometric.nn import DataParallel

    if dataset_type == "MN40":
        model = AmaFilter(3, 3, k=32, activation=activation)
    elif dataset_type == "MPEG":
        model = AmaFilter(6, 6, k=32, filter=bfilter, activation=activation)
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
                {"params": model.parameters(), "initial_lr": 0.002},
                # {"params": model.parameters(), "initial_lr": 0.002}
            ],
            lr=0.002,
            weight_decay=5e-4,
            betas=(0.9, 0.999),
        )
    elif optimizer_type == "SGD":
        # Using SGD Nesterov-accelerated with Momentum
        # Selective lr adjustment
        # my_list = [
        #     "module.filters.0.embedding",
        #     "module.filters.1.embedding",
        #     "module.filters.2.embedding",
        # ]
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, last_epoch=beg_epochs
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
    )