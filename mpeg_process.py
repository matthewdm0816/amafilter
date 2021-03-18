import torch
import torch.nn.functional as F
from torch.utils import data
import torch_geometric as tg
from torch_geometric.nn import knn_graph, fps, knn
from torch_geometric.data import Data, InMemoryDataset

# from torch_geometric.datasets import InMemoryDataset
import torch_geometric.io as tgio
from torch_geometric.io import read_off, read_ply
from torch_geometric.transforms import SamplePoints, NormalizeScale
import os, sys, time, random
import numpy as np
from plyfile import PlyData, PlyElement
import colorama
import pretty_errors
from typing import Optional, Union
from tqdm import *

from utils import *
from scaffold import Scaffold
from dataloader import *


colorama.init(autoreset=True)
scaf = Scaffold()
scaf.debug()
sprint = scaf.print
warn = scaf.warn

dataset_path = "/data/pkurei/mpeg/dataset"
names = ["longdress", "soldier", "redandblack", "loot"]
ply_paths = ["longdress.ply"]
out_paths = ["longdress.pt"]


def read_mesh(path: str):
    r"""
    Read mesh XYZ/RGB from PLY file
    """
    print(colorama.Fore.RED + "Loading PLY file %s" % path)
    mesh = PlyData.read(path)
    x, y, z = (
        torch.from_numpy(mesh.elements[0].data["x"].copy()),
        torch.from_numpy(mesh.elements[0].data["y"].copy()),
        torch.from_numpy(mesh.elements[0].data["z"].copy()),
    )
    r, g, b = (
        torch.from_numpy(mesh.elements[0].data["red"].copy()),
        torch.from_numpy(mesh.elements[0].data["green"].copy()),
        torch.from_numpy(mesh.elements[0].data["blue"].copy()),
    )
    pos = torch.cat([x, y, z], dim=-1).to(torch.float32)
    color = torch.cat([r, g, b], dim=-1).to(torch.float32)
    # Asserting both [N, 3] shape
    assert pos.shape == color.shape
    return Data(color=color, pos=pos)


def process_ply(
    ply_path: str,
    n_patch: int = 100,
    k: int = 2048,
    down_sample: Union[None, int, float] = None,
    cuda: bool = False,
):
    r"""
    Processes PLY file from path, convert to PC
    :return patch-pos + patch-color + patch-kernel
    """
    mesh_data = read_mesh(ply_path)
    print(colorama.Fore.GREEN + "Loaded PLY file %s" % ply_path)
    pos, color = mesh_data.pos, mesh_data.color
    if cuda:
        pos = pos.to(torch.device("cuda:7"))
        color = color.to(pos)
    # down sampling using uniform method
    n_pts = pos.shape[0]
    if down_sample is not None:
        if isinstance(down_sample, int):
            idx = torch.randperm(n_pts)[:down_sample]
            pos, color = pos[idx], color[idx]
        elif isinstance(down_sample, float):
            idx = torch.randperm(n_pts)[: np.floor(down_sample * n_pts)]
            pos, color = pos[idx], color[idx]
    # select N kernels by FPS sample: [N, ]
    patch_kernel = fps(pos, ratio=n_patch / n_pts, random_start=True)
    # sprint(patch_kernel.shape)
    # select N patches: [N, 2048]
    patches = knn(pos, pos[patch_kernel], k=k, num_workers=32)
    patches = patches[1].reshape(-1, k)
    # sprint(patches.shape)
    data_list = [
        # ([N, 2048, 3], [N, 2048, 3], [N, 3])
        Data(color=color[patch], pos=pos[patch], kernel_pos=pos[pk])
        for patch, pk in zip(patches, patch_kernel)
    ]
    return data_list


def process_dataset(dataset_path: str, names: list, cuda: bool = False):
    import re

    regex = re.compile("\.ply$")  # match all .ply files
    data_list = []
    for name in names:
        folder = os.path.join(dataset_path, name, "Ply")
        for root, dirs, files in os.walk(folder):
            for file in tqdm(files):
                if regex.search(file) is not None:
                    # print(file)
                    res = process_ply(
                        os.path.join(folder, file), n_patch=50, k=2048, cuda=cuda
                    )
                    data_list += res
                    # break # debug: load only one
    # combine into one large data
    return Data(
        y=torch.stack([d.color for d in data_list]),
        z=torch.stack([d.pos for d in data_list]),
        kernel_z=torch.stack([d.kernel_pos for d in data_list]),
    )


class MPEGLargeDataset(InMemoryDataset):
    r"""
    Parse MPEG Dataset: resampled
    imporved from jxr
    Composed with a list of Data(x<i.e. noised y>, y, pos(x-z concat), label)
    transform order: pre-T => +noise
    """

    names = [
        "loot",
        "soldier",
        "longdress",
        "redandblack",
    ]
    n_classes = [3000, 3000, 3000, 3000]

    def __init__(
        self,
        root,
        src,
        noise_generator=normal_noise,
        transform=None,
        pre_transform=None,
        sigma=1,
        num_workers=8,
        cuda=False,
    ):
        self.sigma = sigma
        self.noise_generator = noise_generator
        self.num_workers = num_workers
        # self.root = root
        self.cuda = cuda
        self.src = src
        super(MPEGLargeDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # @property
    # def raw_file_names(self):
    #     return [name + ".mat" for name in self.names]

    @property
    def processed_file_names(self):
        return ["dataset.pt"]

    def process(self):
        print(
            colorama.Fore.YELLOW + "Processing Dataset with σ={:.2E}".format(self.sigma)
        )
        data = process_dataset(self.src, names=self.names, cuda=self.cuda)
        # apply pre_transform: i.e. whiten
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        # add noise to x
        noise = self.noise_generator(data.y, self.sigma)
        data.x = data.y + noise
        # check nan
        assert not torch.any(torch.isnan(data.x)), "NaN detected!"
        # divide into list
        data_list = [
            Data(x=dx, y=dy, z=dz, kernel_z=dkz.reshape(-1))
            for dx, dy, dz, dkz in zip(data.x, data.y, data.z, data.kernel_z)
        ]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    from pprint import pprint

    r"""
    Unit Test of process_ply
    """
    # ply_path = os.path.join(
    #     dataset_path, "longdress", "Ply", "longdress_vox10_1099.ply"
    # )
    # res = process_ply(ply_path, n_patch=10)
    # pprint(res)

    r"""
    Process MPEG Seq. Dataset
    """
    for sigma in (1.0, 5.0, 10.0):
        dataset = MPEGLargeDataset(
            root="/data/pkurei/mpeg/datav2-%.1f" % sigma,
            src=dataset_path,
            noise_generator=normal_noise,
            pre_transform=MPEGTransform,
            cuda=False,
            sigma=sigma,
        )
