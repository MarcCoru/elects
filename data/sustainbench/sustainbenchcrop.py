
from data.sustainbench.croptypemapping_dataset import CropTypeMappingDataset, CM_LABELS
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np
import torch

# https://drive.google.com/file/d/1xwaAUL9tZ3LEUwC6ZGOELrshFCGkclM2/view?usp=sharing
# [not adaoted] kenia https://drive.google.com/file/d/1434NDGzuqahT38ZsmmR2wrc7vxcjUi9F/view?usp=sharing

class SustainbenchCrops(Dataset):

    def __init__(self, partition, root="/data/sustainbench/", sequencelength=70, country="ghana", train_test_frac=0.75):

        self.sequencelength = sequencelength

        npy_folder = os.path.join(root, "npy")
        os.makedirs(npy_folder, exist_ok=True)

        X_path = os.path.join(npy_folder, f"{country}_X.npy")
        y_path = os.path.join(npy_folder, f"{country}_y.npy")

        if not (os.path.exists(X_path) or os.path.exists(y_path)):
            # spatiotemporal dataset [D x H x W x T]
            ds = CropTypeMappingDataset(root_dir=os.path.join(root, 'africa_crop_type_mapping_v1.0'), split_scheme=country, download=True)

            self.X = []
            self.y = []
            self.sequencelengths = []
            self.ndims = []
            self.ids = []

            for idx, (X,y,meta) in enumerate(tqdm(ds, total=len(ds))):

                X_s2 = X["s2"]
                # D x H x W x T -> H x W x D x T
                X_s2 = X_s2.permute(1,2,0,3)

                classes = [c for c in y.long().unique() if c > 0]

                for c in classes:

                    mask = y == c

                    # H x W x D x T --average pixels-> D x T
                    X = X_s2[mask].mean(0)

                    # remove temporal padding
                    X = X[:, meta["s2"] > 0]

                    ndims, sequencelength = X.shape

                    self.X.append(X.numpy())
                    self.y.append(int(c))
                    self.ndims.append(ndims)
                    self.sequencelengths.append(sequencelength)
                    self.ids.append(idx)

            T = max(self.sequencelengths)
            X_ = []
            for X in self.X:
                npad = T - X.shape[1]
                X_.append(np.pad(X, [(0, 0), (0, npad)], 'constant', constant_values=0))
            # stack to N x T x D
            self.X = np.stack(X_).transpose(0,2,1)

            np.save(os.path.join(npy_folder, f"{country}_X.npy"), self.X)
            np.save(os.path.join(npy_folder, f"{country}_y.npy"), np.array(self.y))
            np.save(os.path.join(npy_folder, f"{country}_sequencelengths.npy"), np.array(self.sequencelengths))
            np.save(os.path.join(npy_folder, f"{country}_ndims.npy"), np.array(self.ndims))
            np.save(os.path.join(npy_folder, f"{country}_ids.npy"), np.array(self.ids))

        else:
            self.X = np.load(os.path.join(npy_folder, f"{country}_X.npy"), allow_pickle=True)
            self.y = np.load(os.path.join(npy_folder, f"{country}_y.npy"), allow_pickle=True)
            self.ids = np.load(os.path.join(npy_folder, f"{country}_ids.npy"), allow_pickle=True)
            self.sequencelengths = np.load(os.path.join(npy_folder, f"{country}_sequencelengths.npy"), allow_pickle=True)
            self.ndims = np.load(os.path.join(npy_folder, f"{country}_ndims.npy"), allow_pickle=True)

        self.y = self.y - 1

        # remove classes that are not in the official 4 classes...
        mask = [y in CM_LABELS[country] for y in self.y]
        self.X = self.X[mask]
        self.y = self.y[mask]
        self.sequencelengths = self.sequencelengths[mask]
        self.ndims = self.ndims[mask]

        # split train/test
        is_train = np.random.rand(self.X.shape[0]) < train_test_frac

        if partition == "train":
            mask = is_train
        else:
            mask = ~is_train

        self.X = self.X[mask]
        self.y = self.y[mask]
        self.sequencelengths = self.sequencelengths[mask]
        self.ndims = self.ndims[mask]

        self.classids = CM_LABELS[country]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        t = self.sequencelengths[idx]
        X = self.X[idx, :t]
        y = np.array(self.classids.index(self.y[idx]))  # repeat y for each entry in x

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
        y = torch.from_numpy(y).type(torch.LongTensor)

        X = torch.nan_to_num(X)
        X[X>1e20] = 0

        assert not X.isnan().any()

        return X, y.repeat(self.sequencelength)
