"""Script to download SAR images from Google Earth Engine."""

import argparse
import getpass
import os

import asf_search as asf
import ee
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser(description="Download SAR images from ASF")
parser.add_argument(
    "--start", type=str, help="Start date in ISO 8601 format (YYYY-MM-DDThh:mm:ssZ)"
)
parser.add_argument(
    "--end", type=str, help="End date in ISO 8601 format (YYYY-MM-DDThh:mm:ssZ)"
)
parser.add_argument(
    "--product",
    type=str,
    help="Processing level/ Product type",
    default=asf.PRODUCT_TYPE.GRD_HD,
)
parser.add_argument(
    "--output",
    type=str,
    help="Output directory for extracted files",
    default="./data/SAR",
)
args = parser.parse_args()

token = getpass.getpass("EDL Token (login to earthdata.nasa.gov to get personal token):")
token_session = asf.ASFSession().auth_with_token(token)

lats = (7.5, 17)
lons = (-60.25, -45)

aoi = (
    f"POLYGON(({lons[0]} {lats[0]}, {lons[1]} {lats[0]},"
    f" {lons[1]} {lats[1]}, {lons[0]} {lats[1]},"
    f" {lons[0]} {lats[0]}))"
)

opts = {
    "platform": asf.PLATFORM.SENTINEL1,
    "start": args.start,
    "end": args.end,
    "processingLevel": args.product,
    "flightDirection": asf.FLIGHT_DIRECTION.DESCENDING,
    "beamMode": asf.BEAMMODE.IW,
}

results = asf.geo_search(intersectsWith=aoi, **opts)
print(f"{len(results)} results found")
ids = [result.properties["sceneName"] for result in results]

if not os.path.exists(args.output):
    os.makedirs(args.output)

# Download GRD data from Google Earth Engine as Sentinel-1 GRD data
# is already noice corrected there besides other post-processing
# steps
ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")

imgs = ee.ImageCollection(
    [ee.Image("COPERNICUS/S1_GRD/{id}".format(id=id)) for id in ids]
)
geobounds = ee.Geometry.Rectangle(lons[0], lats[0], lons[1], lats[1])

merged_imgs = ee.ImageCollection([imgs.mosaic()])
ds = xr.open_dataset(merged_imgs, engine="ee", scale=0.001, geometry=geobounds)

m = ds[["VV"]]
res = m.where(m != 0, np.nan)
# res.to_netcdf(args.output+'/'+'_'.join(opts.values())+".nc")
res.to_netcdf(args.output + "/" + "test.nc")
