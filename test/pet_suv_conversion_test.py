import os
from mirp.extract_features_and_images import extract_features

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_pet_suv_conversion_none():

    data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "mask"),
        pet_suv_conversion="none",
        roi_name="GTV_Mass",
        base_feature_families="statistics"
    )

    feature_data = data[0]

    assert len(feature_data) == 1
    assert 43800.0 < feature_data["stat_max"].values[0] < 43900.0
    assert 930.0 < feature_data["stat_min"].values[0] < 940.


def test_pet_suv_conversion_body_weight():

    data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "mask"),
        pet_suv_conversion="body_weight",
        roi_name="GTV_Mass",
        base_feature_families="statistics"
    )

    feature_data = data[0]

    assert len(feature_data) == 1
    assert 8.0 < feature_data["stat_max"].values[0] < 8.05
    assert 0.15 < feature_data["stat_min"].values[0] < 0.20


def test_pet_suv_conversion_body_surface_area():

    data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "mask"),
        pet_suv_conversion="body_surface_area",
        roi_name="GTV_Mass",
        base_feature_families="statistics"
    )

    feature_data = data[0]

    assert len(feature_data) == 1
    assert 0.195 < feature_data["stat_max"].values[0] < 0.200
    assert 0.004 < feature_data["stat_min"].values[0] < 0.005
