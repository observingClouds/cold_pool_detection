"""Cold pool detection algorithm following Tompkins (2001)

References
----------
Tompkins, A. M. (2001). Organization of Tropical Convection in Low Vertical Wind
    Shears: The Role of Cold Pools. Journal of the Atmospheric Sciences, 58(13),
    1650–1672. https://doi.org/10.1175/1520-0469(2001)058<1650:OOTCIL>2.0.CO;2
"""

import os

import cartopy.crs as ccrs
import datashader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from datashader.mpl_ext import dsshow


def buoyancy(theta_p, theta_p_mean=None, g=9.81):
    """Bouyancy calculation following Tompkins (2001)"""
    if theta_p_mean is None:
        theta_p_mean = np.mean(theta_p)
    b = g * (theta_p - theta_p_mean) / theta_p_mean
    return b


def theta_p(theta, qv, qc, qr):
    """Density potential temperature calculation following Emmanuel (1994)

    Inputs
    ------
    theta : float
        Potential temperature
    qv : float
        Water vapor mass mixing ratio
    qc : float
        Cloud condensate mass mixing ratio
    qr : float
        Rain water mass mixing ratio
    """
    theta_p = theta * (1 + 0.608 * qv - qc - qr)
    return theta_p


def potential_temperature(T, p, p0=1000, k=0.2854):
    """Potential temperature calculation.

    Inputs
    ------
    T : float
        Temperature
    p : float
        Pressure
    p0 : float
        Reference pressure
    k : float
        R/Cp
    """
    theta = T * (p0 / p) ** k
    return theta


def plot(da, grid, vmin=None, vmax=None, cmap="RdBu_r", dpi=100, fig=None):
    # Lazy loading of output and grid

    central_longitude = 0  # -53.54884554550185
    # central_latitude = 12.28815437976341
    # satellite_height = 8225469.943160511

    projection = ccrs.PlateCarree(
        central_longitude=central_longitude
    )  # , central_latitude=central_latitude, satellite_height=satellite_height)

    coords = projection.transform_points(
        ccrs.Geodetic(),
        np.rad2deg(grid.clon.values),
        np.rad2deg(grid.clat.values),
    )

    if fig is None:
        fig, ax = plt.subplots(subplot_kw={"projection": projection}, dpi=dpi)
        fig.canvas.draw_idle()
        # ax.add_feature(cf.COASTLINE, linewidth=0.8)
    else:
        ax = fig.gca()

    # gl = ax.gridlines(projection, draw_labels=True, alpha=0.35)
    # gl.top_labels = False
    # gl.right_labels = False

    _ = dsshow(
        pd.DataFrame({
            "val": da.values,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }),
        datashader.Point("x", "y"),
        datashader.mean("val"),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        ax=ax,
    )

    # fig.colorbar(artist, label=f"{da.units}", shrink=0.8)
    return fig


if __name__ == "__main__":
    import argparse

    import intake

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time", help="Specify the time in the format 'YYYY-MM-DD HH:MM:SS'"
    )
    parser.add_argument("--output", help="Specify output folder for the plots.")
    args = parser.parse_args()

    time = args.time
    out_dir = args.output

    cat = intake.open_catalog(
        "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
    )
    grid = cat.simulations.grids["6b59890b-99f3-939b-e76a-0a3ad2e43140"].to_dask()
    ds = cat.simulations.ICON.LES_CampaignDomain_control["3D_DOM01"].to_dask()
    data = ds.sel(time=time).isel(height=67).load()
    # b = cold_pool_mask(data["temp"], data["pres"], data["qv"], data["qc"], data["qr"])
    data["theta"] = potential_temperature(data["temp"], data["pres"])
    theta_dp = theta_p(data["theta"], data["qv"], data["qc"], data["qr"])
    # large_scale_mean = data['theta'].rolling(cell=100000, min_periods=1).mean()
    # b = buoyancy(theta_dp, data['theta_coarse'])
    # plot((b > -0.005).astype(int), grid)
    #
    data["theta_coarse"] = xr.DataArray(
        data=np.zeros_like(data["theta"]),
        coords=data["theta"].coords,
        dims=data["theta"].dims,
    )
    theta_dp.attrs["units"] = "K"
    data["theta_coarse"].attrs["units"] = "K"
    pointer = 0
    for block in data["theta"].chunk(cell=100000).data.blocks:
        data["theta_coarse"][pointer : pointer + len(block)] = np.mean(block)
        pointer += len(block)
    b = buoyancy(theta_dp, data["theta_coarse"])
    cold_pool_mask = (b > 0.05).astype(float)
    cold_pool_mask.attrs["units"] = ""
    cold_pool_mask[cold_pool_mask == 1] = np.nan

    # Plotting
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fig = plot(data["temp"], grid, vmin=290, vmax=300)
    fig.savefig(f"{out_dir}/temparature_{time}.png")
    fig = plot(cold_pool_mask, grid, cmap="Greys_r", fig=fig)
    fig.savefig(f"{out_dir}/joint_{time}.png")

    # Plotting cold pool mask only
    fig = plot(cold_pool_mask, grid, cmap="Greys_r")
    fig.savefig(f"{out_dir}/cold_pool_mask_{time}.png")

    # Plotting satellite data
    # to be added

    cat_local = intake.open_catalog(
        "https://github.com/observingClouds/tape_archive_index/raw/main/catalog.yml"
    )
    sat_entry = cat_local["EUREC4A_ICON-LES_control_DOM01_RTTOV_native"]
    sat_entry.storage_options["preffs"]["prefix"] = "/scratch/m/m300408/"
    ds_sat = sat_entry.to_dask()
    ds_sat_sel = ds_sat["synsat_rttov_forward_model_1__abi_ir__goes_16__channel_7"].sel(
        time=time
    )
    fig = plot(ds_sat_sel, grid, vmin=270, vmax=300)
    fig.savefig(f"{out_dir}/satellite_{time}.png")
