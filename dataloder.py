"""
Dataloaders of datasets: MPEG etc.
"""
import torch
import numpy as np
import scipy.io as spio
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataListLoader, DataLoader

import sys, os, math, random, json, time
import colorama
from os.path import join
colorama.init(autoreset=True)

class MPEGDataset(InMemoryDataset):
    """
    Parse MPEG Dataset
    imporved from EternalHope's
    """
    names = [
        'loot', 'soldier', 'longdress', 'redandblack',
        'andrew', 'david', 'ricardo', 'sarah'
    ]
    def __init__(self, root, training=True, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.training = training
        self.transform = transform
        self.pre_transform = pre_transform
    
    @property
    def raw_file_names(self):
        return [
            name + '.mat' for name in self.names
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def property(self):
        print(colorama.Fore.YELLOW + 'Processing Dataset')
        raw_dir = join(self.root, 'raw')
        processed_dir = join(self.root, 'processed')
        for name in self.names:
            raw_data = spio.loadmat(join(raw_dir, name) + '.mat')
            y = torch.from_array(raw_data['colNet'])
            z = torch.from_array(raw_data['geoNet'])
            n_pc, n_point, _ = y.shape # pc/point amounts


