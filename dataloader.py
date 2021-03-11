"""
Dataloaders of datasets: MPEG etc.
"""
import torch
import numpy as np
import scipy.io as spio
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import DataListLoader, DataLoader

import sys, os, math, random, json, time
import gc
import colorama
from os.path import join
from tqdm import *
colorama.init(autoreset=True)

from utils import tensorinfo
from scaffold import *
scaf = Scaffold()
scaf.debug()
sprint = scaf.print
warn = scaf.warn
class MPEGDataset(InMemoryDataset):
    """
    Parse MPEG Dataset
    imporved from EternalHope's
    """
    names = [
        'loot', 'soldier', 'longdress', 'redandblack',
        'andrew', 'david', 'ricardo', 'sarah'
    ]
    n_classes = [1520, 1520, 1520, 1520, 3180, 2160, 2160, 2070]
    def __init__(self, root, training=True, transform=None, pre_transform=None, sigma=0.1):
        self.sigma = sigma
        self.training = training
        super(MPEGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [
            name + '.mat' for name in self.names
        ]

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def process(self):
        print(colorama.Fore.YELLOW + 'Processing Dataset')
        data_list = []

        # parse .mat PCs
        for i_name, raw_path in tqdm(enumerate(self.raw_paths)):
            gc.collect() # manually calls gc collect
            raw_data = spio.loadmat(raw_path)
            y = torch.from_numpy(raw_data['colNet']) # transfer to gpu to accel?
            z = torch.from_numpy(raw_data['geoNet'])
            # sprint(tensorinfo(y), y.shape, y.element_size() * y.nelement() / 1024 / 1024)
            n_pc, n_point, _ = y.shape # pc/point amounts
            noise_y = torch.randn_like(y).to(y) * self.sigma
            for idx in range(n_pc):
                data_list.append(
                    Data(x=noise_y[idx], y=y[idx], 
                        pos=torch.cat((noise_y[idx], z[idx]), dim=-1), 
                        label=i_name)
                )

        # apply pre-transform
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print(len(data_list))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class ADataListLoader(DataListLoader):
    """
    DataListLoader exclusive to MPEGDataset
    """
    def __init__(self, dataset, batch_size, training=True, test_classes=[], *args, **kwargs):
        names, n_classes = dataset.names, dataset.n_classes
        for test_class in test_classes:
            print(colorama.Fore.CYAN + 'Using classes #%d: %s as test data' % (test_class, names[test_class]))
        # NOTE: Assume Dataset is organized in sequential orders!!!
        tmp = 0
        data_list = []
        # remove/include test class
        for i_class, (name, n_class) in enumerate(zip(names, n_classes)):
            if (training and not i_class in test_classes) \
                or (not training and i_class in test_classes):
                data_list += dataset[tmp: tmp + n_class]
                tmp += n_class
        print(colorama.Fore.GREEN + 'Final dataset: %d PCs' % (len(data_list)))
        super().__init__(data_list, batch_size, *args, **kwargs)
        
if __name__ == '__main__':
    r"""
    Unit Test of dataset/dataloaders
    """
    dataset = MPEGDataset(root='data', training=True, sigma=0.1)
    train_loader = ADataListLoader(dataset, training=True, test_classes=[0, 1], batch_size=16, shuffle=True, drop_last=False, num_workers=8)
    print("%d batches in total!" % (len(train_loader)))
    for batch in train_loader:
        sprint(batch)
        break


