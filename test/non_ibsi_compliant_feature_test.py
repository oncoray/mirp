import os.path
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

GENERIC_KWARGS = dict([
    ("write_features", False),
    ("export_features", True),
    ("image", os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1", "image")),
    ("mask", os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1", "mask")),
    ("roi_name", "GTV_Mass")
])


def test_morphological_features():
    # These are just functional tests: there is no reference standard.
    from mirp import extract_features

    data_compliant = extract_features(
        base_feature_families="morphology",
        ibsi_compliant=True,
        **GENERIC_KWARGS
    )[0]

    data_non_compliant = extract_features(
        base_feature_families="morphology",
        ibsi_compliant=False,
        **GENERIC_KWARGS
    )[0]

    non_compliant_features = ["morph_max_2d_diam_z", "morph_max_2d_diam_y", "morph_max_2d_diam_x"]
    assert all(x not in data_compliant.columns for x in non_compliant_features)
    assert all(x in data_non_compliant.columns for x in non_compliant_features)


def test_statistics_features():
    # Theses are just functional tests: there is no reference standard.
    from mirp import extract_features

    data_compliant = extract_features(
        base_feature_families="statistics",
        ibsi_compliant=True,
        **GENERIC_KWARGS
    )[0]

    data_non_compliant = extract_features(
        base_feature_families="statistics",
        ibsi_compliant=False,
        **GENERIC_KWARGS
    )[0]

    non_compliant_features = []
    assert all(x not in data_compliant.columns for x in non_compliant_features)
    assert all(x in data_non_compliant.columns for x in non_compliant_features)