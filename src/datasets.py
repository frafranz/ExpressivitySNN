from itertools import permutations
import numpy as np
import os.path as osp
import torch
from torch.utils.data.dataset import Dataset
import torchvision


class BarsDataset(Dataset):
    class_names = ['horiz', 'vert', 'diag']

    def __init__(self, square_size,
                 early=0.05, late=0.5,
                 noise_level=1e-2,
                 samples_per_class=10,
                 multiply_input_layer=1):
        assert type(multiply_input_layer) == int
        if early is None:
            early = 0.05
        if late is None:
            late = 0.5
        debug = False

        self.__vals = []
        self.__cs = []
        ones = list(np.ones(square_size) + (late - 1.))
        if debug:
            print(ones)
        starter = [ones]
        for _ in range(square_size - 1):
            starter.append(list(np.zeros(square_size) + early))
        if debug:
            print('Starter')
            print(starter)
        horizontals = []
        for h in permutations(starter):
            horizontals.append(list(h))
        horizontals = np.unique(np.array(horizontals), axis=0)
        if debug:
            print('Horizontals')
            print(horizontals)
        verticals = []
        for h in horizontals:
            v = np.transpose(h)
            verticals.append(v)
        verticals = np.array(verticals)
        if debug:
            print('Verticals')
            print(verticals)
        diag = [late - early for _ in range(square_size)]
        first = np.diag(diag) + early
        second = first[::-1]
        diagonals = [first, second]
        if debug:
            print('Diagonals')
            print(diagonals)
        n = 0
        idx = 0
        while n < samples_per_class:
            h = horizontals[idx].flatten()
            h = list(h + np.random.rand(len(h)) * noise_level)
            self.__vals.append(h)
            self.__cs.append(0)
            n += 1
            idx += 1
            if idx >= len(horizontals):
                idx = 0
        n = 0
        idx = 0
        while n < samples_per_class:
            v = verticals[idx].flatten()
            v = list(v + np.random.rand(len(v)) * noise_level)
            self.__vals.append(v)
            self.__cs.append(1)
            n += 1
            idx += 1
            if idx >= len(verticals):
                idx = 0
        n = 0
        idx = 0
        while n < samples_per_class:
            d = diagonals[idx].flatten()
            d = list(d + np.random.rand(len(d)) * noise_level)
            self.__vals.append(d)
            self.__cs.append(2)
            n += 1
            idx += 1
            if idx >= len(diagonals):
                idx = 0

        if multiply_input_layer > 1:
            self.__vals = np.array(self.__vals).repeat(multiply_input_layer, 1)

    def __getitem__(self, index):
        return np.array(self.__vals[index]), self.__cs[index]

    def __len__(self):
        return len(self.__cs)


class FullMnist(Dataset):
    def __init__(self, which='train', early=None, late=None, invert=True, late_at_inf=False):
        if early is None:
            early = 0.15
        if late is None:
            late = 2.

        self.cs = []
        self.vals = []
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        if not np.all([osp.isfile(f"../data/mnist_{which}{i}.npy")
                       for i in ['_label', '']]):
            print("Need to download images...")
            if which == 'train':
                train = True
                start_sample = 0
                end_sample = 50000
            elif which == 'val':
                train = True
                start_sample = 50000
                end_sample = 60000
            elif which == 'test':
                train = False
                start_sample = 0
                end_sample = 10000
            self.data = torchvision.datasets.MNIST('../data/mnist', train=train, download=True, transform=torchvision.transforms.ToTensor())
            loader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False)
            tmp_for_save = []
            for i, elem in enumerate(loader):
                if i >= start_sample and i < end_sample:
                    label = elem[1][0].data.item()
                    self.cs.append(label)
                    dat_flat = elem[0][0][0].flatten().double()
                    dat_flat *= 1./256.
                    tmp_for_save.append(dat_flat)
                    if invert:
                        dat_flat = dat_flat * (-1.) + 1.  # make invert white and black
                    dat_flat = early + dat_flat * (late - early)
                    if late_at_inf:
                        dat_flat[dat_flat == late] = np.inf # i.e. lowest values do not lead to any spikes
                    self.vals.append(dat_flat)
            tmp_for_save = np.array([ii.cpu().detach().numpy() for ii in tmp_for_save])
            np.save(f"../data/mnist_{which}_label.npy",
                    torch.tensor(self.cs).cpu().detach().numpy())
            np.save(f"../data/mnist_{which}.npy",
                    torch.tensor(tmp_for_save).cpu().detach().numpy())
            print("Saved processed images")
        else:
            print("load preprocessed data")
            self.cs = np.load(f"../data/mnist_{which}_label.npy")
            tmp_data = np.load(f"../data/mnist_{which}.npy")
            for i, dat_flat in enumerate(tmp_data):
                if invert:
                    dat_flat = dat_flat * (-1.) + 1.  # make invert white and black (ensured that saved data are not inverted / late at inf)
                dat_flat = early + dat_flat * (late - early)
                if late_at_inf:
                    dat_flat[dat_flat == late] = np.inf
                self.vals.append(dat_flat)

    def __getitem__(self, index):
        return self.vals[index], self.cs[index]

    def __len__(self):
        return len(self.cs)


class CroppedMnist(Dataset):
    def __init__(self, which='train', width_pixel=16, early=None, late=None, invert=True, late_at_inf=False):
        if early is None:
            early = 0.15
        if late is None:
            late = 2.

        self.cs = []
        self.vals = []
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        if not np.all([osp.isfile(f"../data/{width_pixel}x{width_pixel}_mnist_{which}{i}.npy")
                       for i in ['_label', '']]):
            if which == 'train':
                train = True
                start_sample = 0
                end_sample = 50000
            elif which == 'val':
                train = True
                start_sample = 50000
                end_sample = 60000
            elif which == 'test':
                train = False
                start_sample = 0
                end_sample = 10000
            self.data = torchvision.datasets.MNIST(
                '../data/mnist', train=train, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.CenterCrop((24, 24)),
                    torchvision.transforms.Resize((width_pixel, width_pixel)),
                    torchvision.transforms.ToTensor()]))

            loader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False)
            tmp_for_save = []
            for i, elem in enumerate(loader):
                if i >= start_sample and i < end_sample:
                    label = elem[1][0].data.item()
                    self.cs.append(label)
                    dat_flat = elem[0][0][0].flatten().double()
                    dat_flat *= 1./256.
                    tmp_for_save.append(dat_flat)
                    if invert:
                        dat_flat = dat_flat * (-1.) + 1.  # make invert white and black
                    dat_flat = early + dat_flat * (late - early)
                    if late_at_inf:
                        dat_flat[dat_flat == late] = np.inf
                    self.vals.append(dat_flat)
            tmp_for_save = np.array([ii.cpu().detach().numpy() for ii in tmp_for_save])
            np.save(f"../data/{width_pixel}x{width_pixel}_mnist_{which}_label.npy",
                    torch.tensor(self.cs).cpu().detach().numpy())
            np.save(f"../data/{width_pixel}x{width_pixel}_mnist_{which}.npy",
                    torch.tensor(tmp_for_save).cpu().detach().numpy())
            print("Saved processed images")
        else:
            print("load preprocessed data")
            self.cs = np.load(f"../data/{width_pixel}x{width_pixel}_mnist_{which}_label.npy")
            tmp_data = np.load(f"../data/{width_pixel}x{width_pixel}_mnist_{which}.npy")
            for i, dat_flat in enumerate(tmp_data):
                if invert:
                    dat_flat = dat_flat * (-1.) + 1.  # make invert white and black
                dat_flat = early + dat_flat * (late - early)
                if late_at_inf:
                    dat_flat[dat_flat == late] = np.inf
                self.vals.append(dat_flat)

    def __getitem__(self, index):
        return self.vals[index], self.cs[index]

    def __len__(self):
        return len(self.cs)


class YinYangDataset(Dataset):
    def __init__(self, which='train', early=None, late=None,
                 r_small=0.1, r_big=0.5, size=1000, seed=42,
                 multiply_input_layer=1):
        assert type(multiply_input_layer) == int
        if early is None:
            early = 0.15
        if late is None:
            late = 2.

        self.cs = []
        self.vals = []

        try:
            import yin_yang_data_set.dataset
        except (ModuleNotFoundError, ImportError):
            print("Make sure you installed the submodule (github.com/lkriener/yin_yang_data_set)")
            raise
        tmp_dataset = yin_yang_data_set.dataset.YinYangDataset(r_small, r_big, size, seed)
        self.class_names = tmp_dataset.class_names
        loader = torch.utils.data.DataLoader(tmp_dataset, batch_size=1, shuffle=False)

        for i, elem in enumerate(loader):
            self.cs.append(elem[1][0].data.item())

            # extract and multiply (used on hardware)
            vals = elem[0][0].flatten().double().repeat(multiply_input_layer)
            # transfrom values in 0..1 to spiketimes in early..late
            self.vals.append(early + vals * (late - early))

    def __getitem__(self, index):
        return self.vals[index], self.cs[index]

    def __len__(self):
        return len(self.cs)


class XOR(Dataset):
    def __init__(self, which='train', early=None, late=None,
                 r_small=0.1, r_big=0.5, size=1000, seed=42,
                 multiply_input_layer=1):
        assert multiply_input_layer == 1
        if early is None:
            early = 0.15
        if late is None:
            late = 2.        

        self.cs = []
        self.vals = []
        self.class_names = ['False', 'True']

        for i, elem in enumerate(
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1],
             ]
        ):
            self.cs.append(torch.tensor(elem).sum() % 2)
            self.vals.append(
                torch.hstack([torch.tensor(elem),
                              torch.tensor([0, 1])]  # bias
                            ) * (late - early) + early
            )

    def __getitem__(self, index):
        return self.vals[index], self.cs[index]

    def __len__(self):
        return len(self.cs)
