"""sim_cp_tompkins dataset."""

import os

import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for sim_cp_tompkins dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "label": tfds.features.ClassLabel(names=["environment", "coldpool"]),
                "segmentation_mask": tfds.features.Image(),
            }),
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://github.com/observingClouds/cold_pool_detection",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(
            "/Users/haukeschulz/Documents/GitHub/cold_pool_detection/data/labels.tar.gz"
        )
        return {
            tfds.Split.TRAIN: self._generate_examples(path=data_dir),
        }

    def _open_file(self, path):
        if os.path.isfile(path):
            with open(path, "rb") as file_obj:
                return file_obj

    def _generate_examples(self, path):
        """Yields examples."""
        masks = path.glob("labels/*cold_pool_mask.png")
        for filename in masks:
            filename.parts
            label_fn = str(filename).replace("cold_pool_mask", "satellite")
            yield str(filename), {
                "image": str(filename),
                "label": "environment",
                "segmentation_mask": label_fn,
            }
