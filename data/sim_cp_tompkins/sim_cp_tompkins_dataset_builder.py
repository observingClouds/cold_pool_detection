"""sim_cp_tompkins dataset."""

import os

import numpy as np
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
                "segmentation_mask": tfds.features.LabeledImage(
                    labels=["environment", "coldpool"]
                ),
            }),
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://github.com/observingClouds/cold_pool_detection",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract("data/labels.tar.gz")
        return {
            tfds.Split.TRAIN: self._generate_examples(
                path=data_dir, split=0.5, set="train"
            ),
            tfds.Split.TEST: self._generate_examples(
                path=data_dir, split=0.5, set="test"
            ),
        }

    def _open_file(self, path):
        if os.path.isfile(path):
            with open(path, "rb") as file_obj:
                return file_obj

    def _generate_examples(self, path, split, set):
        """Yields examples."""
        masks = np.array(list(path.glob("labels/*cold_pool_mask.png")))
        np.random.seed(1)
        nb_images = len(masks)
        nb_images_set = int(np.floor(nb_images * split))
        random_idx = np.arange(nb_images)
        np.random.shuffle(random_idx)
        if set == "train":
            ind = sorted(random_idx[:nb_images_set])
        elif set == "test":
            ind = sorted(random_idx[nb_images_set:])
        for filename in masks[ind]:
            filename.parts
            input_fn = str(filename).replace("cold_pool_mask", "satellite")
            if not os.path.exists(input_fn):
                continue
            yield str(filename), {
                "image": input_fn,
                "segmentation_mask": str(filename),
            }
