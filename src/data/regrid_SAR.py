"""Regrid irregularly spaced SAR data to regular grid."""

import numpy as np
import xarray as xr
from pyresample.geometry import AreaDefinition, GridDefinition
from pyresample.kd_tree import XArrayResamplerNN

ds = xr.open_dataset(
    "../../data/SAR/S1A_IW_OCN__2SDV_20200113T094154_20200113T094223_030780_0387B2_6706.SAFE/measurement/s1a-iw-ocn-vv-20200113t094155-20200113t094228-030780-0387B2-001.nc"
)
lon, lat = (ds.owiLon, ds.owiLat)
lon_out = np.arange(-60.25, -45, 0.01)
lat_out = np.arange(7.5, 17, 0.01)

area_id = "barbados_east"
description = "ICON DOM01"
proj_id = "icon_dom01"
projection = "EPSG:4326"

lon_out = np.arange(-60.25, -45, 0.01)
lat_out = np.arange(7.5, 17, 0.01)
width = len(lon_out)
height = len(lat_out)
area_extent = (-60.25, 7.5, -45, 17)

area_def = AreaDefinition(
    area_id, description, proj_id, projection, width, height, area_extent
)
grid_def = GridDefinition(lons=lon.values, lats=lat.values)

resampler = XArrayResamplerNN(grid_def, area_def, radius_of_influence=5000)
resampler.get_neighbour_info()
result = resampler.get_sample_from_neighbour_info(
    ds.owiWindSpeed.rename({"owiAzSize": "y", "owiRaSize": "x"})
)

ds_grid = xr.Dataset(
    {"owiWindSpeed": result.rename({"y": "latitude", "x": "longitude"})}
)
