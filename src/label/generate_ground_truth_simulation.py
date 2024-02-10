"""Wrapper to call specified ground truth classification algorithm."""

import argparse

import dvc.api
import tqdm

params = dvc.api.params_show()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir", help="Specify the output directory of classifications"
)
args = parser.parse_args()

output_dir = args.output_dir

if params["ground_truth_simulation"]["method"] == "tompkins_2001":
    from src.tompkins_et_al_2001 import Classifier

    classifier = Classifier(output_dir, params=params["tompkins_2001"])
    hgt = params["tompkins_2001"]["height"]
    times = classifier.times
    for time in tqdm.tqdm(times[:4]):
        dt_time = time.astype("datetime64[s]").item()
        time_str = dt_time.strftime("%Y-%m-%d %H:%M:%S")
        classifier.classify(time_str, height=hgt)
