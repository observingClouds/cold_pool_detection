"""Cold pool detection algorithm following Tompkins (2001)

References
----------
Tompkins, A. M. (2001). Organization of Tropical Convection in Low Vertical Wind
    Shears: The Role of Cold Pools. Journal of the Atmospheric Sciences, 58(13),
    1650–1672. https://doi.org/10.1175/1520-0469(2001)058<1650:OOTCIL>2.0.CO;2
"""

import numpy as np


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


def cold_pool_mask(T, p, qv, qc, qr, buoyancy_threshold=-0.005, theta_p_mean=None):
    """Cold pool masking algorithm following Tompkins (2001)

    Inputs
    ------
    T : float
        Temperature
    p : float
        Pressure
    qv : float
        Water vapor mass mixing ratio
    qc : float
        Cloud condensate mass mixing ratio
    qr : float
        Rain water mass mixing ratio
    buoyancy_threshold : float
        Buoyancy threshold for cold pool detection. Default = -0.005 m**2/s
    """
    theta = potential_temperature(T, p)
    theta_dp = theta_p(theta, qv, qc, qr)
    b = buoyancy(theta_dp, theta_p_mean)
    return b  # > buoyancy_threshold


def area_avg(data):
    return data.rolling(cell=10000, min_periods=1).mean()


import cartopy.crs as ccrs
import cartopy.feature as cf
import datashader
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from datashader.mpl_ext import dsshow
from matplotlib import ticker as mticker


def plot(da, grid, vmin=None, vmax=None, cmap="RdBu_r", dpi=100):
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

    fig, ax = plt.subplots(subplot_kw={"projection": projection}, dpi=dpi)
    fig.canvas.draw_idle()
    ax.add_feature(cf.COASTLINE, linewidth=0.8)

    gl = ax.gridlines(projection, draw_labels=True, alpha=0.35)
    gl.top_labels = False
    gl.right_labels = False
    gl.ylocator = mticker.FixedLocator(np.arange(11, 16, 1))
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 0, 1))

    ax.add_patch(
        mpatches.Circle(
            xy=[-57.717, 13.3],
            radius=1,
            edgecolor="grey",
            fill=False,
            transform=ccrs.PlateCarree(),
            zorder=30,
        )
    )

    artist = dsshow(
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

    fig.colorbar(artist, label=f"{da.units}", shrink=0.8)
    return fig


if __name__ == "__main__":
    import intake

    cat = intake.open_catalog(
        "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
    )
    grid = cat.simulations.grids["6b59890b-99f3-939b-e76a-0a3ad2e43140"].to_dask()
    ds = cat.simulations.ICON.LES_CampaignDomain_control["3D_DOM01"].to_dask()
    data = ds.sel(time="2020-01-10 00:00:00").isel(height=67).load()
    b = cold_pool_mask(data["temp"], data["pres"], data["qv"], data["qc"], data["qr"])
    plot((b > -0.005).astype(int), grid)
