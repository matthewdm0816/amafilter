r"""
Dataloaders of datasets: MPEG etc.
TODO:
    1. -Generate various sigma data-
"""
from scaffold import *
from utils import tensorinfo, mse
import torch
import numpy as np
import scipy.io as spio
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import DataListLoader, DataLoader
import torch_geometric.nn as tgnn

import sys
import os
import math
import random
import json
import time
import gc
import colorama
from os.path import join
from tqdm import *
from multiprocessing import Process, Queue, Pool, TimeoutError
from argparse import ArgumentParser
import shutil

colorama.init(autoreset=True)

scaf = Scaffold()
scaf.debug()
sprint = scaf.print
warn = scaf.warn


def whiten(v):
    r"""
    Whiten data to mean 0, std. 1
    NOTE: for std~=0 data, add eps=1e-6
    OPT: could merge into one
    """
    if len(v.shape) == 2: # single PC
        return (v - v.mean(dim=0)) / (v.std(dim=0) + 1e-6)
    elif len(v.shape) == 3: # batch
        return (v - v.mean(dim=-2, keepdim=True)) / (v.std(dim=-2, keepdim=True) + 1e-6)
    else:
        raise ValueError


def remove_ac(v):
    r"""
    Remove AC component from data
    """
    return v - v.mean(dim=0)


def normalize_scale(v):
    r"""
    Normalize to [-1, 1] box
    """
    v = remove_ac(v)
    scale = (1 / v.abs().max()) * 0.999999
    return v * scale


def pointset_diameter(v, sample_times=100):
    r"""
    Calc. diamter of point cloud
    """
    n_pts, fin = v.shape
    eps = 1e-6
    diameter = -1.0
    for _ in range(sample_times):
        index = tgnn.fps(v, ratio=2 / n_pts + eps)
        distance = (v[index][0] - v[index][1]).norm()
        diameter = max(distance, diameter)

    return diameter


def normal_noise(v, sigma=0.1):
    r"""
    Generate noise ~ sigma, for single PC
    """
    noise = torch.randn_like(v) * sigma
    noise = noise.to(v)
    return noise


def sphere_noise(v, sigma=0.1, sample_times=100):
    r"""
    Generate noise ~ sigma * d(PC), for single point cloud!
    d(PC) ~ diameter of point cloud
    """
    diameter = pointset_diameter(v, sample_times)
    # print(diameter)
    noise = torch.randn_like(v) * sigma * diameter
    noise = noise.to(v)
    return noise


class MPEGDataset(InMemoryDataset):
    r"""
    Parse MPEG Dataset
    imporved from jxr
    Composed with a list of Data(x<i.e. noised y>, y, pos(x-z concat), label)
    transform order: pre-T => +noise
    """

    names = [
        "loot",
        "soldier",
        "longdress",
        "redandblack",
        "andrew",
        "david",
        "ricardo",
        "sarah",
    ]
    n_classes = [1520, 1520, 1520, 1520, 3180, 2160, 2160, 2070]

    def __init__(
        self,
        root,
        noise_generator=normal_noise,
        transform=None,
        pre_transform=None,
        sigma=0.1,
        num_workers=8,
    ):
        self.sigma = sigma
        self.noise_generator = noise_generator
        self.num_workers = num_workers
        super(MPEGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [name + ".mat" for name in self.names]

    @property
    def processed_file_names(self):
        return ["dataset.pt"]

    @staticmethod
    def _process_data(data, noise_generator, pre_transform=None, sigma=0.1):
        if pre_transform is not None:
            data = pre_transform(data)
        # add noise to x
        assert not torch.any(torch.isnan(data.y)), "NaN detected!"
        noise = noise_generator(data.y, sigma)
        # print(noise.norm())
        data.x = data.x + noise
        # assert not torch.any(torch.isnan(data.x)), "NaN detected!"
        # print(noise.norm())
        return data

    def process(self):
        print(
            colorama.Fore.YELLOW + "Processing Dataset with σ={:.2E}".format(self.sigma)
        )
        data_list = []

        # parse .mat PCs
        if self.num_workers > 1:
            print(colorama.Fore.GREEN + "Using %d cores..." % self.num_workers)
        for i_name, raw_path in tqdm(
            enumerate(self.raw_paths), total=len(self.raw_paths)
        ):
            raw_data = spio.loadmat(raw_path)
            y = torch.from_numpy(raw_data["colNet"])
            z = torch.from_numpy(raw_data["geoNet"])
            n_pc, n_point, _ = y.shape  # pc/point amounts
            if self.num_workers > 1:
                # parallel process
                with Pool(processes=self.num_workers) as pool:
                    result = pool.starmap(
                        self._process_data,
                        [
                            (
                                Data(
                                    x=y[idx],
                                    y=y[idx],
                                    z=z[idx],
                                    # pos=torch.cat((noise_y[idx], z[idx]), dim=-1),
                                    label=i_name,
                                ),
                                self.noise_generator,
                                self.pre_transform,
                                self.sigma,
                            )
                            for idx in range(n_pc)
                        ],
                    )
                    data_list += result
            else:
                for idx in trange(n_pc):
                    data = Data(
                        x=y[idx],
                        y=y[idx],
                        z=z[idx],
                        # pos=torch.cat((noise_y[idx], z[idx]), dim=-1),
                        label=i_name,
                    )
                    # apply pre_transform: i.e. whiten
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    # add noise to x
                    noise = self.noise_generator(data.y, self.sigma)
                    # print(noise.norm())
                    data.x = data.x + noise
                    # check nan
                    assert not torch.any(torch.isnan(data.x)), "NaN detected @ %d!" % (len(data_list))
                    # print(mse(data.x, data.y))
                    data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class ADataListLoader(DataListLoader):
    """
    DataListLoader exclusive to MPEGDataset
    """

    def __init__(
        self, dataset, batch_size, training=True, test_classes=[], *args, **kwargs
    ):
        names, n_classes = dataset.names, dataset.n_classes
        for test_class in test_classes:
            print(
                colorama.Fore.CYAN
                + "Using classes #%d: %s as test data" % (test_class, names[test_class])
            )
        # NOTE: Assume Dataset is organized in sequential orders!!!
        tmp = 0
        data_list = []
        # remove/include test class
        for i_class, (name, n_class) in enumerate(zip(names, n_classes)):
            if (training and not i_class in test_classes) or (
                not training and i_class in test_classes
            ):
                data_list += dataset[tmp : tmp + n_class]
                tmp += n_class
        print(colorama.Fore.GREEN + "Final dataset: %d PCs" % (len(data_list)))
        super().__init__(data_list, batch_size, *args, **kwargs)


def MPEGTransform(data, func=whiten):
    r"""
    Transformations for MPEG dataset
    Data(x, y, pos, label) | x, y, pos with shape [N, CHANNEL]
    color => mean 0, std. 1.0
    """
    # data.x = whiten(data.x)
    for key in data.keys:
        # process all tensors
        if torch.is_tensor(data[key]) and key != 'kernel_z':
            data[key] = func(data[key])
    return data


def showData(data):
    r"""
    show info of components of Data obj.
    """
    for key in data.keys:
        # process all tensors
        if torch.is_tensor(data[key]):
            sprint(tensorinfo(data[key]))


if __name__ == "__main__":
    # parse args
    parser = ArgumentParser()
    parser.add_argument("-u", "--unit-test", help="run unit-test", action="store_true")
    parser.add_argument(
        "-g", "--generate", help="generate dataset", action="store_true"
    )
    parser.add_argument(
        "-s", "--sigma", help="σ of added gaussian noise", nargs="+", type=float
    )
    args = parser.parse_args()
    assert args.unit_test != args.generate

    if args.unit_test:
        r"""
        Unit Test of dataset/dataloaders
        """
        dataset = MPEGDataset(
            root="data-0.2", sigma=0.2, num_workers=16, pre_transform=MPEGTransform
        )
        train_loader = ADataListLoader(
            dataset,
            training=True,
            test_classes=[0, 1],
            batch_size=16,
            shuffle=True,
            drop_last=False,
            num_workers=8,
        )
        test_loader = ADataListLoader(
            dataset,
            training=False,
            test_classes=[0, 1],
            batch_size=16,
            shuffle=True,
            drop_last=False,
            num_workers=8,
        )
        print("%d batches in total!" % (len(train_loader)))
        for batch in train_loader:
            sprint(batch)
            showData(batch[0])
            break
        for batch in test_loader:
            sprint(batch)
            showData(batch[0])
            break
    elif args.generate:
        """
        Generate according to given σ
        """
        for sigma in args.sigma:
            # copy source raw
            orig_dataset_dir = "data/raw"
            dataset_dir = "data-%.1f" % sigma
            raw_dir = os.path.join(dataset_dir, "raw")
            proc_dir = os.path.join(dataset_dir, 'processed')
            if os.path.exists(raw_dir):
                print(colorama.Fore.RED + "Removing old raw at %s" % raw_dir)
                shutil.rmtree(raw_dir)
            if os.path.exists(proc_dir):
                print(colorama.Fore.RED + "Removing old processed at %s" % proc_dir)
                shutil.rmtree(proc_dir)
            shutil.copytree(orig_dataset_dir, raw_dir)
            # process dataset
            print(colorama.Fore.GREEN + "Setting σ=%.2f" % sigma)
            dataset = MPEGDataset(
                root=dataset_dir,
                sigma=sigma,
                num_workers=1,
                pre_transform=MPEGTransform,
            )
