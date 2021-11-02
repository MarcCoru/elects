print()

import os
import pandas as pd
import geopandas as gpd

root = "/data/modiscrop/"

root = "/ssd2/modiscdl/"

modis = pd.read_csv(os.path.join(root,"modisCDL.csv"))
points = gpd.read_file(os.path.join(root, "modispts.shp")).reset_index()

modis["latplon"] = (modis.latitude + modis.longitude)

precision = 3

modis["x"] = modis.longitude.round(precision)
modis["y"] = modis.latitude.round(precision)
points["x"] = points.geometry.x.round(precision)
points["y"] = points.geometry.y.round(precision)

joined = pd.merge(modis, points,  how='left', left_on=['x','y'], right_on = ['x','y']).dropna().set_index("index")

ndvi = joined.loc[12462.0].NDVI

print()

"""
import netCDF4 as nc
fn = os.path.join(root, "MCD13.A2015.unaccum.nc4")

ds = nc.Dataset(fn)

print()
"""
