"""
Dataloaders of datasets: MPEG etc.
"""
import torch
import numpy as np
import scipy.io as spio
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import DataListLoader, DataLoader

import sys, os, math, random, json, time
import colorama
from os.path import join
from tqdm import *
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
    n_class = [1520, 1520, 1520, 1520, 3180, 2160, 2160, 2070]
    def __init__(self, root, training=True, transform=None, pre_transform=None, sigma=0.1):
        super().__init__(root, transform, pre_transform)
        self.sigma = sigma
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
        return ['dataset.pt']

    def property(self):
        print(colorama.Fore.YELLOW + 'Processing Dataset')
        raw_dir = join(self.root, 'raw')
        processed_dir = join(self.root, 'processed')
        data_list = []

        # parse .mat PCs
        for i_name, name in tqdm(enumerate(self.names)):
            raw_data = spio.loadmat(join(raw_dir, name) + '.mat')
            y = torch.from_numpy(raw_data['colNet'])
            z = torch.from_numpy(raw_data['geoNet'])
            n_pc, n_point, _ = y.shape # pc/point amounts
            noise_y = torch.randn_like(y) * self.sigma
            data_list.append([
                Data(x=noise_y[idx], y=y[idx], 
                    pos=torch.cat((noise_y, z), dim=-1), 
                    label=torch.tensor(i_name))
                for idx in n_pc
            ])

        # apply pre-transform
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class ADataListLoader(DataListLoader):
    """
    DataListLoader exclusive to MPEGDataset
    """
    def __init__(self, dataset, batch_size, training=True, test_classes=[], *args, **kwargs):
        names, n_class = dataset.names, dataset.n_class
        for test_class in test_classes:
            print(colorama.Fore.CYAN + 'Using classes #%d: %s as test data' % (test_class, names[test_class]))
        # NOTE: Assume Dataset is organized in sequential orders!!!
        tmp = 0
        data_list = []
        # remove/include test class
        for i_class, (name, n_class) in enumerate(zip(names, n_class)):
            if (training and not i_class in test_classes) \
                or (not training and i_class in test_classes):
                data_list += dataset[tmp: tmp + n_class[i_class]]
                tmp += n_class[i_class]
        print(colorama.Fore.GREEN + 'Final dataset: %d PCs' % (len(data_list)))
        super().__init__(data_list, batch_size, *args, **kwargs)
        
if __name__ == '__main__':
    r"""
    Unit Test of dataset/dataloaders
    """
    dataset = MPEGDataset(root='', training=True, sigma=0.1)



