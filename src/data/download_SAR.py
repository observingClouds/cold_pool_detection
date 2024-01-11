import getpass
import os

import asf_search as asf

token = getpass.getpass("EDL Token:")
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
    "start": "2020-01-13T00:00:00Z",
    "end": "2020-01-13T23:59:59Z",
    "processingLevel": asf.PRODUCT_TYPE.OCN,
    "flightDirection": asf.FLIGHT_DIRECTION.DESCENDING,
    "beamMode": asf.BEAMMODE.IW,
}

results = asf.geo_search(intersectsWith=aoi, **opts)
print(f"{len(results)} results found")

for result_within_zip in results[1:]:
    with result_within_zip.remotezip(session=token_session) as z:
        file_paths = [
            file.filename for file in z.filelist if file.filename.endswith(".nc")
        ]

        print(f"found {len(file_paths)} nc files in zip")

        for file_path in file_paths:
            extract_path = "./data/SAR"
            os.makedirs(extract_path, exist_ok=True)
            z.extract(file_path, path=extract_path)
