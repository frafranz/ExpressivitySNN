#!python3
import numpy as np
import os.path as osp
import sys
import time
import torch
import torch.nn
import torch.autograd


def network_load(path, basename, device):
    net = to_device(torch.load(osp.join(path, basename + "_network.pt"),
                               map_location=device),
                    device)
    net.device = get_default_device()
    return net

def get_default_device():
    # Pick GPU if avialable, else CPU
    if torch.cuda.is_available():
        print("Using GPU, Yay!")
        return torch.device('cuda')
    else:
        print("Using CPU, Meh!")
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    # Wrap a dataloader to move data to device
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


class TIMER:
    def __init__(self, pre=""):
        self.timestmp = time.perf_counter()
        self.pre = pre

    def time(self, label=""):
        if self.timestmp > 0:
            print(f"{self.pre}{label} {(time.perf_counter() - self.timestmp) * 1e3:.0f}ms")
        self.timestmp = time.perf_counter()
