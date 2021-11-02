print()

import os

root = "/data/modiscrop/"

import netCDF4 as nc
fn = os.path.join(root, "MCD13.A2015.unaccum.nc4")

ds = nc.Dataset(fn)

print()
