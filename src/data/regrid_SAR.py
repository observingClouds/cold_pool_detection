"""Regrid irregularly spaced SAR data to regular grid."""

import numpy as np
import xarray as xr
from scipy.interpolate import griddata


def regrid_data(input, output):
    """Regrid irregularly spaced SAR data to regular grid.

    Inputs
    ------
    input : tuple
        Tuple of (latitude, longitude, values) arrays.
    output : tuple
        Tuple of 1D (latitude, longitude) arrays the data should be interpolated to.

    Returns
    -------
    tuple: Tuple of (latitude, longitude, values) arrays.
    """
    lat_in, lon_in, val_in = input
    lat_out, lon_out = output

    # assert lat_in == np.sort(lat_in), "Latitude must be ascending"
    # assert lon_in == np.sort(lon_in), "Longitude must be ascending"
    assert np.all(lat_in >= -90) and np.all(
        lat_in <= 90
    ), "Latitude must be between -90 and 90"
    assert np.all(lon_in >= -180) and np.all(
        lon_in <= 180
    ), "Longitude must be between -180 and 180"
    assert np.all(lat_out >= -90) and np.all(
        lat_out <= 90
    ), "Latitude must be between -90 and 90"
    assert np.all(lon_out >= -180) and np.all(
        lon_out <= 180
    ), "Longitude must be between -180 and 180"

    # Flatten the latitude, longitude, and values arrays
    flat_latitude = lat_in.flatten()
    flat_longitude = lon_in.flatten()
    flat_values = val_in.flatten()

    # Create a regular grid of coordinates
    lon_grid, lat_grid = np.meshgrid(lon_out, lat_out)

    # Interpolate the values onto the regular grid
    interpolated_values = griddata(
        (flat_longitude, flat_latitude),
        flat_values,
        (lon_grid, lat_grid),
        method="nearest",
    )

    return lon_out, lat_out, interpolated_values


ds = xr.open_dataset(
    "../../data/SAR/S1A_IW_OCN__2SDV_20200113T094154_20200113T094223_030780_0387B2_6706.SAFE/measurement/s1a-iw-ocn-vv-20200113t094155-20200113t094228-030780-0387B2-001.nc"
)
lon, lat = (ds.owiLon[::-1, ::-1], ds.owiLat[::-1, :])
nan_mask = ~np.isnan(ds.owiWindSpeed)
input = (lat.values[nan_mask], lon.values[nan_mask], ds.owiWindSpeed.values[nan_mask])
lon_out = np.arange(-60.25, -45, 0.1)
lat_out = np.arange(7.5, 17, 0.1)
lon_out, lat_out, d = regrid_data(input, (lat_out, lon_out))
print(np.max(d))
