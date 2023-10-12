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

    # Extract from multiple DICOM stacks in subfolders.
    feature_data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "dicom", "image"),
        mask_sub_folder=os.path.join("CT", "dicom", "mask"),
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=32
    )

    pass