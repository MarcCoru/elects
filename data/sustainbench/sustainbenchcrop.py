
from data.sustainbench.croptypemapping_dataset import CropTypeMappingDataset, CM_LABELS
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np
import torch
from datetime import datetime

# https://drive.google.com/file/d/1xwaAUL9tZ3LEUwC6ZGOELrshFCGkclM2/view?usp=sharing
# [not adaoted] kenia https://drive.google.com/file/d/1434NDGzuqahT38ZsmmR2wrc7vxcjUi9F/view?usp=sharing
# https://drive.google.com/drive/folders/1WhVObtFOzYFiXBsbbrEGy1DUtv7ov7wF

class SustainbenchCrops(Dataset):

    def __init__(self, partition, root="/data/sustainbench/", sequencelength=70, country="ghana",
                 use_s2_only=True, average_pixel=False, max_n_pixels=None):
        assert partition in ["train","val","test"]
        self.sequencelength = sequencelength
        self.use_s2_only = use_s2_only

        subfolder = "s2only" if use_s2_only else "full"
        npy_folder = os.path.join(root, "npy", subfolder)
        os.makedirs(npy_folder, exist_ok=True)

        x_file = os.path.join(npy_folder, f"{country}_{partition}_X.npy")
        if not os.path.exists(x_file):
            # spatiotemporal dataset [D x H x W x T]
            print(f"expecting folder {os.path.join(root, 'africa_crop_type_mapping_v1.0')} to exist")
            ds = CropTypeMappingDataset(root_dir=os.path.join(root, 'africa_crop_type_mapping_v1.0'),
                                        split_scheme=country, download=True, partition=partition, resize_planet=True)

            self.X = []
            self.y = []
            self.sequencelengths = []
            self.ndims = []
            self.ids = []
            self.doys = []

            print("preprocessing dataset (iterating through tiles taking s2 data and aggregating pixels of each field)")
            for idx, data in enumerate(tqdm(ds, total=len(ds))):
                X,y,meta = data

                #X_s2 = X["s2"]

                if X["s2"] is None:
                    print(f"skipping idx {idx}")
                    continue

                classes = [c for c in y.long().unique() if c > 0]

                for c in classes:

                    mask = y == c

                    # D x H x W x T --average pixels-> D x T
                    # xs2 = X["s2"][:, mask].mean(1)
                    xs2 = X["s2"][:, mask].permute(0,2,1)

                    xs2 = torch.nan_to_num(xs2)
                    xs2 = torch.clamp(xs2, min=-3, max=3)

                    # remove temporal padding
                    msk = meta["s2"] > 0
                    xs2 = xs2[:, msk]
                    doys_s2 = [datetime.strptime(str(d.numpy()),"%Y%m%d").timetuple().tm_yday for d in meta["s2"][msk]]

                    if not self.use_s2_only:
                        xs1 = X["s1"][:, mask]#.mean(1)

                        # last two bands can contain NANs
                        xs1 = torch.nan_to_num(xs1)
                        xs1 = torch.clamp(xs1, min=-3, max=3)

                        msk = meta["s1"] > 0
                        xs1 = xs1[:, :, msk].permute(0,2,1)
                        doys_s1 = [datetime.strptime(str(d.numpy()),"%Y%m%d").timetuple().tm_yday for d in meta["s1"][msk]]

                        xplanet = X["planet"][:, mask]#.mean(1)
                        xplanet = xplanet[:4] # only take BGR-NIR <- other bands have NANS

                        xplanet = torch.nan_to_num(xplanet)
                        xplanet = torch.clamp(xplanet, min=-3, max=3)

                        msk = meta["planet"] > 0
                        xplanet = xplanet[:, :, msk]
                        doys_planet = [datetime.strptime(str(d.numpy()), "%Y%m%d").timetuple().tm_yday for d in meta["planet"][msk]]

                        t = np.linspace(0,364,365)

                        # interpolate
                        xs2 = np.stack([np.stack([np.interp(t, doys_s2, x) for x in x_px]) for x_px in xs2.permute(0,2,1)])
                        xs1 = np.stack([np.stack([np.interp(t, doys_s1, x) for x in x_px]) for x_px in xs1.permute(0,2,1)])
                        xplanet = np.stack([np.stack([np.interp(t, doys_planet, x) for x in x_px]) for x_px in xplanet])

                        X_timeseries = np.vstack([xs2, xs1, xplanet]).transpose(0,2,1) # make D x T x N_px
                    else:
                        X_timeseries = xs2

                    if average_pixel:
                        X_timeseries = X_timeseries.mean(-1)[:, :, None]

                    if max_n_pixels is not None:
                        idxs = np.random.choice(np.arange(X_timeseries.shape[-1]), size=max_n_pixels)
                        X_timeseries = X_timeseries[:,:,idxs]

                    ndims, sequencelength, npixel = X_timeseries.shape

                    self.doys.append(doys_s2)
                    self.X.append(X_timeseries)
                    self.y.append([int(c)] * npixel)
                    self.ndims.append([ndims] * npixel)
                    self.sequencelengths.append([sequencelength] * npixel)
                    self.ids.append([idx] * npixel)

            self.sequencelengths = np.hstack(self.sequencelengths)
            T = max(self.sequencelengths)
            X_ = []
            for X in self.X:
                npad = T - X.shape[1]
                X_.append(np.pad(X, [(0, 0), (0, npad), (0,0)], 'constant', constant_values=0))
            # stack to N x T x D
            # self.X = np.stack(X_).transpose(0, 2, 1)
            self.X = np.dstack(X_).transpose(2,1,0)

            self.ndims = np.hstack(self.ndims)
            self.ids = np.hstack(self.ids)

            self.y = np.hstack(self.y)
            np.save(os.path.join(npy_folder, f"{country}_{partition}_X.npy"), self.X)
            np.save(os.path.join(npy_folder, f"{country}_{partition}_y.npy"), self.y)
            np.save(os.path.join(npy_folder, f"{country}_{partition}_sequencelengths.npy"), self.sequencelengths)
            np.save(os.path.join(npy_folder, f"{country}_{partition}_ndims.npy"), self.ndims)
            np.save(os.path.join(npy_folder, f"{country}_{partition}_ids.npy"), self.ids)

        else:
            self.X = np.load(os.path.join(npy_folder, f"{country}_{partition}_X.npy"), allow_pickle=True)
            self.y = np.load(os.path.join(npy_folder, f"{country}_{partition}_y.npy"), allow_pickle=True)
            self.ids = np.load(os.path.join(npy_folder, f"{country}_{partition}_ids.npy"), allow_pickle=True)
            self.sequencelengths = np.load(os.path.join(npy_folder, f"{country}_{partition}_sequencelengths.npy"), allow_pickle=True)
            self.ndims = np.load(os.path.join(npy_folder, f"{country}_{partition}_ndims.npy"), allow_pickle=True)

        self.y = self.y - 1

        # remove classes that are not in the official 4 classes...
        mask = [y in CM_LABELS[country] for y in self.y]
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

        # X = torch.nan_to_num(X)
        # X[X>1e20] = 0

        assert not X.isnan().any()

        return X, y.repeat(self.sequencelength)
