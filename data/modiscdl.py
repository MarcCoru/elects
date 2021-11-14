import os
import pandas as pd
import geopandas as gpd
from torch.utils.data import Dataset
import numpy as np
import torch

# match points of shapefile and time series data by coordinate. precision defines rounding precision to match
PRECISION = 3

CLASSES = {"corn":1,
                "Soybeans":5,
                "Winter Wheat": 24,
                "Fallow": 61,
                "Other Hay": 37,
                "Alfalfa": 36,
                "Sorghum": 4,
                "Rice": 3}

CLASS_IDS = list(CLASSES.values())
CLASS_NAMES = list(CLASSES.keys())

class ModisCDL(Dataset):
    def __init__(self, root, partition, sequencelength):

        modis = pd.read_csv(os.path.join(root,"modisCDL.csv"))
        points = gpd.read_file(os.path.join(root, "modispts.shp")).reset_index()

        modis["x"] = modis.longitude.round(PRECISION)
        modis["y"] = modis.latitude.round(PRECISION)
        points["x"] = points.geometry.x.round(PRECISION)
        points["y"] = points.geometry.y.round(PRECISION)

        joined = pd.merge(modis, points,  how='left', left_on=['x','y'], right_on = ['x','y']).dropna().set_index("index")
        joined.index = joined.index.astype(int)

        # gr
        self.Xs = joined.groupby("index").apply(lambda x: x.NDVI.values).values
        self.Ys = joined.groupby("index").first().cropland.astype(int).to_list()
        #self.point_idxs = joined.index.unique().to_list()

        self.sequencelength = sequencelength

    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, item):
        #pt_idx = self.point_idxs[item]
        X = self.Xs[item]
        crop_id = self.Ys[item]

        # make T x 1
        X = np.array(X)[:,None]

        # normalize to reflectance [0, 1]
        X *= 1e-4

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
        y = CLASS_IDS.index(crop_id)
        y = torch.tensor(y).repeat(self.sequencelength)


        return X, y

if __name__ == '__main__':

    root = "/data/modiscdl/"
    ds = ModisCDL(root, partition="train", sequencelength=24)
    X,y = ds[0]

