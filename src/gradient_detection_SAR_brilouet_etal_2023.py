"""Detection of cold pools based on Brilouet et al. (2023)

This script is based on the descriptions given in Brilouet et al. (2023) to detect cold
pools and cold pool fronts in sythetic aperture radar (SAR) images.

References
----------
Brilouet, P.-E., Bouniol, D., Couvreux, F., Ayet, A., Granero-Belinchon, C., Lothon, M.,
    & Mouche, A. (2023). Trade Wind Boundary Layer Turbulence and Shallow Precipitating
    Convection: New Insights Combining SAR Images, Satellite Brightness Temperature, and
    Airborne In Situ Measurements. Geophysical Research Letters, 50(2), e2022GL102180.
    https://doi.org/10.1029/2022GL102180
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import find_objects, gaussian_filter, label


def find_features(mask_a, mask_b):
    """
    >>> mask_a = np.array([False, True, True, False, False, False, True, False])
    >>> mask_b = np.array([False, False, False, True, False, True, False, False])
    >>> find_features(mask_a, mask_b)
    array([(1,6),])
    """
    cont_mask_a = find_objects(label(mask_a)[0])
    cont_mask_b = find_objects(label(mask_b)[0])
    i = 0
    cold_pools = []
    while i < len(cont_mask_a):
        start = cont_mask_a[i][0].start
        if i + 1 >= len(cont_mask_a):
            end = len(cont_mask_a)
        else:
            end = cont_mask_a[i + 1][0].stop
        middle = []
        for b_range in cont_mask_b:
            b_start, b_end = b_range[0].start, b_range[0].stop
            if start <= b_start and end >= b_end:
                middle.extend(np.arange(b_start, b_end + 1))
        if middle is not []:
            if i + 1 >= len(cont_mask_a):
                cold_pools.append((
                    np.arange(cont_mask_a[i][0].start, cont_mask_a[i][0].stop),
                    middle[:-1],
                    end,
                ))
            else:
                cold_pools.append((
                    np.arange(cont_mask_a[i][0].start, cont_mask_a[i][0].stop),
                    middle[:-1],
                    np.arange(cont_mask_a[i + 1][0].start, cont_mask_a[i + 1][0].stop),
                ))
        i += 2
    return cold_pools


def find_continous_mask(mask):
    """
    >>> mask = np.array([False, True, True, False, True, True, True, False])
    >>> find_continous_mask(mask)
    array([(1,2), (3,6)])
    >>> mask = np.array([False, False, True, True, True, True, True, False])
    >>> find_continous_mask(mask)
    array([(2,6)])
    >>> mask = np.array([False, False, False, False, False, False, False, True])
    >>> find_continous_mask(mask)
    array([(7,7)])
    >>> mask = np.array([False, False, False, False, False, False, True, True])
    >>> find_continous_mask(mask)
    array([(6,7)])
    """
    # Find the indices where the mask changes from False to True or True to False
    changes = np.where(mask[:-1] != mask[1:])[0]

    # Initialize the list to store the continuous mask ranges
    ranges = []

    # Iterate through the changes and find the continuous ranges
    for i in range(0, len(changes), 2):
        if changes[i] == 0:
            start = changes[i] + 1
        else:
            start = changes[i]
        if i == len(changes) - 1:
            end = len(mask) - 1
        else:
            end = changes[i + 1]
        ranges.append((start, end))

    return np.array(ranges)


def apply_detection(data, sigma_filter=10, std_threshold=2):
    cold_pool_mask = np.zeros_like(data)
    gaussian = gaussian_filter(data.values, sigma_filter)[0]
    gradient = np.gradient(gaussian, axis=0)
    std = np.nanstd(gradient)
    for lat, _ in enumerate(data.lat):
        grad = gradient[:, lat]
        mask_a = grad > std_threshold * std
        mask_b = grad < 0
        cold_pools = find_features(mask_a, mask_b)
        mask = np.zeros(len(grad))
        mask[:] = np.nan
        for cold_pool in cold_pools:
            mask[cold_pool[0]] = 1
            mask[cold_pool[2]] = 1
            mask[cold_pool[1]] = 0.5
        cold_pool_mask[0, :, lat] = mask
    return cold_pool_mask


if __name__ == "__main__":
    # Get input arguments via command line
    parser = argparse.ArgumentParser(description="Cold pool detection script")

    parser.add_argument("-i", "--input", type=str, help="SAR input file", required=True)
    parser.add_argument(
        "--lat_start", type=float, help="Starting latitude for slice", required=True
    )
    parser.add_argument(
        "--lat_end", type=float, help="Ending latitude for slice", required=True
    )
    parser.add_argument(
        "--lon_start", type=float, help="Starting longitude for slice", required=True
    )
    parser.add_argument(
        "--lon_end", type=float, help="Ending longitude for slice", required=True
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output folder for figures", required=True
    )

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    ds = xr.open_dataset(args.input)

    # SAR image similar to Fig. 3a) of Brilouet et al. (2023)
    lat_slice = slice(args.lat_start, args.lat_end)
    lon_slice = slice(args.lon_start, args.lon_end)
    data = ds.VV.sel(lon=lon_slice, lat=lat_slice)

    fig, ax = plt.subplots(1, 1)
    data.isel(time=0).plot(ax=ax, cmap="grey", vmin=-25, vmax=0, x="lon")
    fig.savefig(args.output + "/SAR_image.png")

    # Figure 3b) of Brilouet et al. (2023)
    cold_pool_mask = apply_detection(data)
    cp_ds = xr.DataArray(
        cold_pool_mask,
        coords=[data.time, data.lon, data.lat],
        dims=["time", "lon", "lat"],
    )
    fig, ax = plt.subplots(1, 1)
    data.isel(time=0).plot(ax=ax, cmap="grey", vmin=-25, vmax=0, x="lon")
    cp_ds.isel(time=0).plot(ax=ax, alpha=0.5, x="lon")
    fig.savefig(args.output + "/SAR_image_w_cold_pools.png")

    # Figure 3c) of Brilouet et al. (2023)
    fig, ax = plt.subplots(1, 1)
    lat_data_slice = data.isel(time=0).sel(lat=9.8, method="nearest")
    lat_data_slice.plot(color="red")
    (cp_ds.isel(time=0).sel(lat=9.8, method="nearest") + lat_data_slice.max()).plot()
    fig.savefig(args.output + "/SAR_backscatter_crosssection.png")
