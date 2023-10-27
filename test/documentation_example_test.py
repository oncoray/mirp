import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


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
