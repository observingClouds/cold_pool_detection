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
import intake
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


def large_scale_average(da, avg_length=100000):
    """Calculating the large scale average of a DataArray."""
    da_out = xr.DataArray(
        data=np.zeros_like(da),
        coords=da.coords,
        dims=da.dims,
    )
    pointer = 0
    for block in da.chunk(cell=avg_length).data.blocks:
        da_out[pointer : pointer + len(block)] = np.mean(block)
        pointer += len(block)
    return da_out


def load_datasets():
    # 3D atmospheric variables
    cat = intake.open_catalog(
        "https://raw.githubusercontent.com/observingClouds/eurec4a-intake/ICON-LES_DOM01_synsat_native/catalog.yml"
    )
    hgrid = cat.simulations.grids["6b59890b-99f3-939b-e76a-0a3ad2e43140"].to_dask()
    ds_3D = cat.simulations.ICON.LES_CampaignDomain_control["3D_DOM01"].to_dask()
    ds_sat = cat.simulations.ICON.LES_CampaignDomain_control[
        "rttov_DOM01_native"
    ].to_dask()
    ds_sat = ds_sat["synsat_rttov_forward_model_1__abi_ir__goes_16__channel_7"]

    return ds_3D, ds_sat, hgrid


def load_frame(ds, ds_sat, time, height):
    """Load individual time frame from the datasets.

    Inputs
    ------
    ds : xarray.Dataset
        3D atmospheric variables
    ds_sat : xarray.Dataset
        Synthetic satellite data
    time : str
        Time in the format 'YYYY-MM-DD HH:MM:SS'
    height : int
        Height index to load
    """
    data_3D = ds.sel(time=time, method="nearest").isel(height=height).load()
    data_sat = ds_sat.sel(time=time, method="nearest").load()

    assert data_3D.time == data_sat.time, "Time mismatch between 3D and satellite data."

    return xr.merge([data_3D, data_sat])


def get_cloud_pool_mask(data, avg_length=None, buoyancy_threshold=-0.005, **kwargs):
    data["theta"] = potential_temperature(data["temp"], data["pres"])
    theta_dp = theta_p(data["theta"], data["qv"], data["qc"], data["qr"])

    if avg_length is not None:
        ls_avg = large_scale_average(data["theta"], avg_length=avg_length)
    else:
        ls_avg = None

    b = buoyancy(theta_dp, ls_avg)
    cold_pool_mask = (b > buoyancy_threshold).astype(float)
    cold_pool_mask.attrs["units"] = ""
    cold_pool_mask[cold_pool_mask == 1] = np.nan
    return cold_pool_mask


def plot_classification(cold_pool_mask, grid, time, out_dir=None):
    """Plot classification results."""
    fig = plot(cold_pool_mask, grid, cmap="Greys_r")
    if out_dir is not None:
        fig.savefig(f"{out_dir}/{time}_cold_pool_mask.png", bbox_inches="tight")
    return fig


def plot_input(data, grid, time, out_dir=None):
    """Plot input data."""
    fig = plot(data["temp"], grid, vmin=290, vmax=300)
    if out_dir is not None:
        fig.savefig(f"{out_dir}/{time}_temparature.png", bbox_inches="tight")
    return fig


def plot_validation(data, cold_pool_mask, grid, time, out_dir=None):
    """Plot overlay of classification and input data."""
    fig = plot(data["temp"], grid, vmin=290, vmax=300)
    fig = plot(cold_pool_mask, grid, cmap="Greys_r", fig=fig)
    if out_dir is not None:
        fig.savefig(f"{out_dir}/{time}_joint.png", bbox_inches="tight")
    return fig


def plot_NN_input(
    data,
    grid,
    time,
    out_dir,
    var="synsat_rttov_forward_model_1__abi_ir__goes_16__channel_7",
):
    """Plot satellite data."""
    fig = plot(data[var], grid, vmin=270, vmax=300)
    if out_dir is not None:
        fig.savefig(f"{out_dir}/{time}_satellite.png", bbox_inches="tight")
    return fig


class Classifier:
    """Classification class."""

    def __init__(self, out_dir, params=None):
        """Initializing classifier by loading lazily available datasets."""
        self.params = params if params is not None else {}
        self.out_dir = out_dir
        self.ds_3D, self.ds_sat, self.grid = load_datasets()
        self.times = list(
            set(self.ds_3D.time.values).intersection(set(self.ds_sat.time.values))
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def classify(self, time, height):
        """Classify cold pools at a given time and height."""
        data = load_frame(self.ds_3D, self.ds_sat, time, height)
        cold_pool_mask = get_cloud_pool_mask(data, **self.params)
        plot_classification(cold_pool_mask, self.grid, time, self.out_dir)
        plot_input(data, self.grid, time, self.out_dir)
        plot_validation(data, cold_pool_mask, self.grid, time, self.out_dir)
        plot_NN_input(data, self.grid, time, self.out_dir)
        plt.close("all")


if __name__ == "__main__":
    import argparse

    import dvc.api

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time", help="Specify the time in the format 'YYYY-MM-DD HH:MM:SS'"
    )
    parser.add_argument("--output", help="Specify output folder for the plots.")
    args = parser.parse_args()

    time = args.time
    out_dir = args.output
    params = dvc.api.params_show()

    classifier = Classifier(out_dir, params=params["tompkins_2001"])
    classifier.classify(time, height=67)
