import torch
from torch.utils.data import Dataset
from utils import download, untar

import sys
import requests
import os
import numpy as np

URL = "https://elects.s3.eu-central-1.amazonaws.com/holl.tar.gz"
CLASSES = ["meadow", "summer barley", "corn", "winter wheat", "winter barley", "clover", "winter triticale"]

class BavarianCrops(Dataset):

    def __init__(self, partition, root=os.environ["HOME"], sequencelength=70, return_ids = False):
        assert partition in ["train", "valid", "eval"]
        if not os.path.exists(os.path.join(root,"holl")):
            print(f"no dataset found in {root}/holl. downloading...")
            tardataset = os.path.join(root, "holl.tar.gz")
            download(URL, tardataset)
            untar(tardataset)

        npy_folder = os.path.join(root, "holl", partition)
        self.classweights = np.load(os.path.join(npy_folder, "classweights.npy"), allow_pickle=True)
        self.y = np.load(os.path.join(npy_folder, "y.npy"), allow_pickle=True)
        self.ndims = int(np.load(os.path.join(npy_folder, "ndims.npy"), allow_pickle=True))
        self.sequencelengths = np.load(os.path.join(npy_folder, "sequencelengths.npy"), allow_pickle=True)
        self.ids = np.load(os.path.join(npy_folder, "ids.npy"), allow_pickle=True)
        self.X = np.load(os.path.join(npy_folder, "X.npy"), allow_pickle=True)
        self.return_ids = return_ids

        self.sequencelength=sequencelength

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx]
        y = np.array([self.y[idx]] * X.shape[0])  # repeat y for each entry in x

        # get length of this sample
        t = X.shape[0]


        if t < self.sequencelength:
            # time series shorter than "sequencelength" will be zero-padded
            npad = self.sequencelength - t
            X = np.pad(X, [(0, npad), (0, 0)], 'constant', constant_values=0)
            y = np.pad(y, (0, npad), 'constant', constant_values=0)
        elif t > self.sequencelength:
            # time series longer than "sequencelength" will be sub-sampled
            idxs = np.random.choice(t, self.sequencelength, replace=False)
            idxs.sort()
            X = X[idxs]
            y = y[idxs]

        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)

        if self.return_ids:
            return X, y, self.ids[idx]
        else:
            return X, y
