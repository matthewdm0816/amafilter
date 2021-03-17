import torch
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.nn import knn_graph, fps, knn
from torch_geometric.data import Data
import torch_geometric.io as tgio
from torch_geometric.io import read_off, read_ply
from torch_geometric.transforms import SamplePoints, NormalizeScale
import os, sys, time, random
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import *
from utils import *
from scaffold import Scaffold
import colorama
import pretty_errors

colorama.init(autoreset=True)
scaf = Scaffold()
scaf.debug()
sprint = scaf.print
warn = scaf.warn

dataset_path = "/data/pkurei/mpeg/dataset"
names = ["longdress", "soldier", "redandblack", "loot"]
ply_paths = ["longdress.ply"]
out_paths = ["longdress.pt"]


def read_mesh(path):
    print(colorama.Fore.MAGENTA + "Loading PLY file %s" % path)
    mesh = PlyData.read(path)
    x, y, z = (
        torch.from_numpy(mesh.elements[0].data["x"]),
        torch.from_numpy(mesh.elements[0].data["y"]),
        torch.from_numpy(mesh.elements[0].data["z"]),
    )
    r, g, b = (
        torch.from_numpy(mesh.elements[0].data["red"]),
        torch.from_numpy(mesh.elements[0].data["green"]),
        torch.from_numpy(mesh.elements[0].data["blue"]),
    )
    pos = torch.cat([x, y, z], dim=-1).to(torch.float32)
    color = torch.cat([r, g, b], dim=-1).to(torch.float32)
    # Asserting both [N, 3] shape
    assert pos.shape == color.shape
    return Data(x=color, pos=pos)


def process_ply(ply_path: str, n_patch: int = 100):
    r"""
    Processes PLY file from path, convert to PC
    :return patch-pos + patch-color + patch-kernel
    """
    mesh_data = read_ply(ply_path)
    pos, color = mesh_data.pos, mesh_data.x
    n_pts = pos.shape[0]
    # select N kernels by FPS sample: [N, ]
    patch_kernel = fps(pos, ratio=n_patch / n_pts + 1e-6)
    # select N patches: [N, 2048]
    patches = knn(pos[patch_kernel], pos, k=2048, num_workers=16)
    data_list = [
        # ([N, 2048, 3], [N, 2048, 3], [N, 3])
        Data(color=color[patch], pos=pos[patch], kernel_pos=pos[pk])
        for patch, pk in zip(patches, patch_kernel)
    ]
    return data_list


if __name__ == "__main__":
    from pprint import pprint

    r"""
    Unit Test of process_ply
    """
    ply_path = os.path.join(
        dataset_path, "longdress", "Ply", "longdress_vox10_1099.ply"
    )
    res = process_ply(ply_path)
    pprint(res)
    pass