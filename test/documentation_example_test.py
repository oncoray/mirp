import os
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_quick_start():
    from mirp import extract_features

    feature_data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "dicom", "image"),
        mask_sub_folder=os.path.join("CT", "dicom", "mask"),
        roi_name="GTV_Mass_CT",
        new_spacing=1.0,
        resegmentation_intensity_range=[-150.0, 200.0],
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=25.0
    )

    feature_data = pd.concat(feature_data)

    assert len(feature_data) == 3


def test_extract_features_examples():
    from mirp import extract_features

    # Simple example.
    feature_data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=32
    )

    assert len(feature_data[0]) == 1
    assert not feature_data[0]["image_voxel_size_x"][0] == 1.0
    assert not feature_data[0]["image_voxel_size_y"][0] == 1.0
    assert feature_data[0]["image_voxel_size_z"][0] == 3.27

    # Resampling.
    feature_data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "PET", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "PET", "dicom", "mask", "RS.dcm"),
        image_modality="PET",
        new_spacing=3.0,
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=32
    )

    assert len(feature_data[0]) == 1
    assert feature_data[0]["image_voxel_size_x"][0] == 3.0
    assert feature_data[0]["image_voxel_size_y"][0] == 3.0
    assert feature_data[0]["image_voxel_size_z"][0] == 3.0

    # 2D approach.
    feature_data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        image_modality="CT",
        by_slice=True,
        new_spacing=1.0,
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=32
    )

    assert len(feature_data[0]) == 1
    assert feature_data[0]["image_voxel_size_x"][0] == 1.0
    assert feature_data[0]["image_voxel_size_y"][0] == 1.0
    assert feature_data[0]["image_voxel_size_z"][0] == 3.27

    # Fixed bin size method.
    feature_data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        image_modality="CT",
        new_spacing=1.0,
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=25.0
    )

    assert len(feature_data[0]) == 1
    assert feature_data[0]["image_voxel_size_x"][0] == 1.0
    assert feature_data[0]["image_voxel_size_y"][0] == 1.0
    assert feature_data[0]["image_voxel_size_z"][0] == 1.0

    # Fixed bin size method with resegmentation.
    feature_data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        new_spacing=1.0,
        image_modality="CT",
        resegmentation_intensity_range=[-200.0, 200.0],
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=25.0
    )

    assert len(feature_data[0]) == 1
    assert feature_data[0]["image_voxel_size_x"][0] == 1.0
    assert feature_data[0]["image_voxel_size_y"][0] == 1.0
    assert feature_data[0]["image_voxel_size_z"][0] == 1.0

    # Laplacian of Gaussian.
    feature_data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        new_spacing=1.0,
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=25.0,
        filter_kernels="laplacian_of_gaussian",
        laplacian_of_gaussian_sigma=2.0
    )

    assert len(feature_data[0]) == 1
    assert feature_data[0]["image_voxel_size_x"][0] == 1.0
    assert feature_data[0]["image_voxel_size_y"][0] == 1.0
    assert feature_data[0]["image_voxel_size_z"][0] == 1.0

    # Only Laplacian of Gaussian, with extra features.
    feature_data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        new_spacing=1.0,
        base_feature_families="none",
        response_map_feature_families=["statistics", "intensity_histogram"],
        filter_kernels="laplacian_of_gaussian",
        laplacian_of_gaussian_sigma=2.0
    )

    assert len(feature_data[0]) == 1
    assert feature_data[0]["image_voxel_size_x"][0] == 1.0
    assert feature_data[0]["image_voxel_size_y"][0] == 1.0
    assert feature_data[0]["image_voxel_size_z"][0] == 1.0


def test_deeplearning_preprocessing():
    from mirp import deep_learning_preprocessing

    processed_data = deep_learning_preprocessing(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        crop_size=[50, 224, 224]
    )

    image = processed_data[0][0][0]
    mask = processed_data[0][1][0]

    assert np.any(image > -1000.0)
    assert np.any(mask)


def test_image_metadata_extraction():
    from mirp import extract_image_parameters

    image_parameters = extract_image_parameters(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image")
    )

    assert len(image_parameters) == 1


def test_mask_label_extraction():
    from mirp import extract_mask_labels

    mask_labels = extract_mask_labels(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm")
    )

    assert len(mask_labels) == 1
