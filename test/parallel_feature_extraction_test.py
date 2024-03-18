import os

import numpy as np

from mirp.extract_features_and_images import extract_features_and_images
from mirp.deep_learning_preprocessing import deep_learning_preprocessing

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_parallel_feature_extraction():
    sequential_data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        perturbation_translation_fraction=[0.0, 0.5],
        base_feature_families="statistics",
        resegmentation_intensity_range=[-1000.0, 250.0]
    )

    paralell_data = extract_features_and_images(
        num_cpus=2,
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        perturbation_translation_fraction=[0.0, 0.5],
        base_feature_families="statistics",
        resegmentation_intensity_range=[-1000.0, 250.0]
    )

    for ii in range(len(sequential_data)):
        assert sequential_data[ii].equals(paralell_data[ii])


def test_parallel_dl_preprocessing():
    sequential_images = deep_learning_preprocessing(
        output_slices=False,
        crop_size=[20, 50, 50],
        export_images=True,
        write_images=False,
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
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "nifti", "image"),
        mask_sub_folder=os.path.join("CT", "nifti", "mask")
    )

    for ii in range(len(sequential_images)):
        assert np.array_equal(sequential_images[ii][0], parallel_images[ii][0])
        assert np.array_equal(sequential_images[ii][1], parallel_images[ii][1])
