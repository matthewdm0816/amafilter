import torch
import torch.nn.functional as F
import torch.nn as nn

# from torchsummary import summary
import torch_geometric.nn as tgnn
from torch_geometric.nn import GCNConv, SGConv, MessagePassing, knn_graph, DataParallel
import torch_geometric as tg
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.utils import degree, get_laplacian, remove_self_loops
import torch_scatter as tscatter
import numpy as np
import random, math, colorama, os, json
from collections import defaultdict
from tqdm import *
from itertools import product
from scaffold import Scaffold
from utils import *
from train_bf import process_batch
from dataloader import ADataListLoader, MPEGDataset, MPEGTransform

scaf = Scaffold()
scaf.debug()
sprint = scaf.print
warn = scaf.warn