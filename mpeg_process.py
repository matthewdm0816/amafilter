import torch
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.io as tgio
from torch_geometric.io import read_off, read_ply
from torch_geometric.transforms import SamplePoints, NormalizeScale
import os, sys, time, random
import numpy as np
from tqdm import *
from utils import *
from scaffold import Scaffold

ply_paths = ["longdress.ply"]
scaf = Scaffold()
scaf.debug()
sprint = scaf.print
warn = scaf.warn

def process_and_save_ply(ply_path):
    mesh_data = read_ply(ply_path)
    pc = SamplePoints(2048 * 100)
