import argparse
import sys

import matplotlib.pyplot as plt
import xarray as xr
from satpy import Scene

sys.path.append(".")
from regrid_SAR import output_area  # noqa: E402

# Create argument parser
parser = argparse.ArgumentParser(description="Regrid GOES data")
parser.add_argument("-f", "--filename", type=str, help="Path to input file")
parser.add_argument("-c", "--channel", type=str, help="Channel to plot")
parser.add_argument("-i", "--image", type=str, help="Output filename of plot")


# Parse command line arguments
args = parser.parse_args()

# Get filename from command line argument
filename = args.filename
channel = args.channel
output_img_fn = args.image

out_def = output_area(7.5, 17, -60.25, -45, 0.01)

input_sat_scene = Scene(reader="abi_l1b", filenames=[filename])
input_sat_scene.load([channel])
output_region_scene = input_sat_scene.resample(out_def)
ds_grid = xr.Dataset({
    "ABI_BT": output_region_scene._datasets[channel].rename(
        {"y": "latitude", "x": "longitude"}
    )
})

# time = output_region_scene._datasets[channel].attrs['start_time']

dpi = 200
fig = plt.figure(figsize=(1525 / dpi, 950 / dpi), dpi=dpi)
ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(ds_grid.ABI_BT, vmin=270, vmax=300, cmap="RdBu_r")
plt.savefig(output_img_fn, dpi=200, pad_inches=0)
