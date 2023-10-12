import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_example_feature_extraction():
    from mirp.extractFeaturesAndImages import extract_features

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


def test_example_image_metadata():
    from mirp.extractImageParameters import extract_image_parameters

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
