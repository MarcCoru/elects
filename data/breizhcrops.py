import torch
from torch.utils.data import Dataset
import breizhcrops as bzh
import os
import numpy as np

class BreizhCrops(Dataset):
    def __init__(self, partition="train", root="breizhcrops_dataset", sequencelength=70, year=2017):
        assert partition in ["train", "valid", "eval"]
        if partition == "train":
            frh01 = bzh.BreizhCrops("frh01", root=root, transform=lambda x: x, preload_ram=True, year=year)
            frh02 = bzh.BreizhCrops("frh02", root=root, transform=lambda x: x, preload_ram=True, year=year)
            self.ds = torch.utils.data.ConcatDataset([frh01, frh02])
        elif partition == "valid":
            self.ds = bzh.BreizhCrops("frh03", root=root, transform=lambda x: x, preload_ram=True, year=year)
        elif partition == "eval":
            self.ds = bzh.BreizhCrops("frh04", root=root, transform=lambda x: x, preload_ram=True, year=year)

        self.sequencelength = sequencelength

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        X,y,id = self.ds[item]

        # take bands and normalize
        # ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12', '...']
        X = X[:,:13] * 1e-4

        # get length of this sample
        t = X.shape[0]


        if t < self.sequencelength:
            # time series shorter than "sequencelength" will be zero-padded
            npad = self.sequencelength - t
            X = np.pad(X, [(0, npad), (0, 0)], 'constant', constant_values=0)
        elif t > self.sequencelength:
            # time series longer than "sequencelength" will be sub-sampled
            idxs = np.random.choice(t, self.sequencelength, replace=False)
            idxs.sort()
            X = X[idxs]

        X = torch.from_numpy(X).type(torch.FloatTensor)


        return X, y.repeat(self.sequencelength)
