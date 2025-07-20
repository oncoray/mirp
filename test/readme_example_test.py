import os
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_example_feature_extraction():
    from mirp import extract_features

    # Extract from single DICOM stack.
    feature_data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=32
    )
    assert len(feature_data) == 1

    # Extract from multiple DICOM stacks in subfolders.
    feature_data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "dicom", "image"),
        mask_sub_folder=os.path.join("CT", "dicom", "mask"),
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=32
    )
    assert len(feature_data) == 3


def test_example_deep_learning_preprocessing():
    from mirp import deep_learning_preprocessing

    processed_data = deep_learning_preprocessing(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        crop_size=[50, 224, 224]
    )

    image = processed_data[0][0][0]
    mask = processed_data[0][1][0]

    assert np.array_equal(image["image"].shape, (50, 224, 224))
    assert np.array_equal(mask["mask"].shape, (50, 224, 224))
    assert np.any(image["image"] > -1000.0)
    assert np.any(mask["mask"])


def test_example_image_metadata():
    from mirp import extract_image_parameters

    # Extract from single DICOM stack.
    image_parameters = extract_image_parameters(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image")
    )
    assert len(image_parameters) == 1

    # # Extract from multiple DICOM stacks in subfolders.
    image_parameters = extract_image_parameters(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "dicom", "image")
    )
    assert len(image_parameters) == 3


def test_example_retrieve_mask_labels():
    from mirp import extract_mask_labels

    mask_labels = extract_mask_labels(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy")
    )
    assert mask_labels.roi_label.values[0] == 1

    # Multiple _masks.
    mask_labels = extract_mask_labels(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask_sub_folder=os.path.join("CT", "numpy", "mask")
    )

    assert all(x == 1 for x in mask_labels.roi_label.values)


def test_example_use_native_data():
    from mirp import extract_images, extract_features

    results = extract_images(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        image_export_format="native"
    )

    image = results[0][0][0]
    mask = results[0][1][0]

    # Obtain the numpy.ndarray.
    voxel_grid = image.get_voxel_grid()

    # Divide intensities by 2.
    image.set_voxel_grid(voxel_grid=voxel_grid / 2.0)

    features = extract_features(
        image=image,
        mask=mask,
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=32
    )[0]

    assert features.stat_min.values[0] == -42.0
    assert features.stat_max.values[0] == 75.0
