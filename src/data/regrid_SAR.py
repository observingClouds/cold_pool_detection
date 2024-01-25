"""Regrid irregularly spaced SAR data to regular grid."""

import glob

import numpy as np
import xarray as xr
from pyresample.geometry import AreaDefinition, GridDefinition
from pyresample.kd_tree import XArrayResamplerNN


def output_area(lat_min, lat_max, lon_min, lon_max, resolution):
    """Return output area."""
    area_id = "barbados_east"
    description = "ICON DOM01"
    proj_id = "icon_dom01"
    projection = "EPSG:4326"

    lon_out = np.arange(lon_min, lon_max, resolution)
    lat_out = np.arange(lat_min, lat_max, resolution)
    width = len(lon_out)
    height = len(lat_out)
    area_extent = (lon_min, lat_min, lon_max, lat_max)

    return AreaDefinition(
        area_id, description, proj_id, projection, width, height, area_extent
    )


out_def = output_area(7.5, 17, -60.25, -45, 0.01)

file_pattern = "../../data/SAR/*/measurement/*.nc"

# Loop over files of one timestep to merge them in one gridded dataset
dss = []
for file in glob.glob(file_pattern):
    ds = xr.open_dataset(file)

    lon, lat = (ds.owiLon, ds.owiLat)
    in_def = GridDefinition(lons=lon.values, lats=lat.values)

    resampler = XArrayResamplerNN(in_def, out_def, radius_of_influence=5000)
    resampler.get_neighbour_info()
    result = resampler.get_sample_from_neighbour_info(
        ds.owiWindSpeed.rename({"owiAzSize": "y", "owiRaSize": "x"})
    )

    ds_grid = xr.Dataset(
        {"owiWindSpeed": result.rename({"y": "latitude", "x": "longitude"})}
    )

    # Add gridded dataset
    dss.append(ds_grid)


# Merge all gridded datasets
def merge(dss):
    init_val = dss[0].load()
    for ds in dss[1:]:
        for var in ds.data_vars:
            init_val[var].values[~np.isnan(ds[var])] = ds[var].values[~np.isnan(ds[var])]
    return init_val


ds_combined = merge(dss)
