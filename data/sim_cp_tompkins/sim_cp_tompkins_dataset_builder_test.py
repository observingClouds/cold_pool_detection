"""sim_cp_tompkins dataset."""

import tensorflow_datasets as tfds

from . import sim_cp_tompkins_dataset_builder


class SimCpTompkinsTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for sim_cp_tompkins dataset."""

    # TODO(sim_cp_tompkins):
    DATASET_CLASS = sim_cp_tompkins_dataset_builder.Builder
    SPLITS = {
        "train": 3,  # Number of fake train example
        "test": 1,  # Number of fake test example
    }

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == "__main__":
    tfds.testing.test_main()
