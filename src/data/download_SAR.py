"""Script to download SAR images from ASF.

References:
https://github.com/asfadmin/Discovery-asf_search/blob/master/examples/5-Download.ipynb
"""

import argparse
import getpass
import os

import asf_search as asf

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
    default=asf.PRODUCT_TYPE.OCN,
)
parser.add_argument(
    "--output",
    type=str,
    help="Output directory for extracted files",
    default="./data/SAR",
)
args = parser.parse_args()

token_file = os.path.expanduser("~/.edl_credentials")
if os.path.exists(token_file):
    with open(token_file, "r") as file:
        token = file.read().strip()
else:
    token = getpass.getpass(
        "EDL Token (login to earthdata.nasa.gov to get personal token):"
    )

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

if not os.path.exists(args.output):
    os.makedirs(args.output)

for result_within_zip in results[1:]:
    with result_within_zip.remotezip(session=token_session) as z:
        if args.product == asf.PRODUCT_TYPE.OCN:
            file_paths = [
                file.filename
                for file in z.filelist
                if file.filename.endswith(".grd") or file.filename.endswith(".nc")
            ]
        else:
            file_paths = [file.filename for file in z.filelist]

        print(f"found {len(file_paths)} nc files in zip")

        for file_path in file_paths:
            extract_path = args.output
            z.extract(file_path, path=extract_path)
