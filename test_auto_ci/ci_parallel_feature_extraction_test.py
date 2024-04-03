import os
import numpy as np

from mirp.deep_learning_preprocessing import deep_learning_preprocessing

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test")


def test_parallel_dl_preprocessing():
    sequential_images = deep_learning_preprocessing(
        output_slices=False,
        crop_size=[20, 50, 50],
        export_images=True,
        write_images=False,
        image_export_format="numpy",
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "nifti", "image"),
        mask_sub_folder=os.path.join("CT", "nifti", "mask")
    )

    parallel_images = deep_learning_preprocessing(
        num_cpus=2,
        output_slices=False,
        crop_size=[20, 50, 50],
        export_images=True,
        write_images=False,
        image_export_format="numpy",
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "nifti", "image"),
        mask_sub_folder=os.path.join("CT", "nifti", "mask")
    )

    for ii in range(len(sequential_images)):
        assert np.array_equal(sequential_images[ii][0], parallel_images[ii][0])
        assert np.array_equal(sequential_images[ii][1], parallel_images[ii][1])
