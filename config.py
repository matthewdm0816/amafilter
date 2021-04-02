import yaml, json
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    ic()
    from yaml import Loader, Dumper
import pretty_errors
import sys, os
from icecream import ic
import re
from contextlib import contextmanager

class Configurator():
    def __init__(self, config_path: str, autoparse: bool=True):
        self.config_path = config_path
        if autoparse:
            self.parse()

    def parse(self):
        with open(self.config_path, 'r') as f:
            if re.search('json', self.config_path):
                raise NotImplementedError
            elif re.search('yml', self.config_path):
                data = yaml.load(f, Loader=Loader)
            
        self._data = data
    
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    @property
    def data(self):
        return self._data

    @contextmanager
    def setenv(self):
        yield self.data


def showconf():
    for key in config.keys():
        ic(key, config[key])

if __name__ == "__main__":
    conf = Configurator(config_path='RAW.yml', autoparse=True)
    with conf.setenv() as config:
        showconf()
