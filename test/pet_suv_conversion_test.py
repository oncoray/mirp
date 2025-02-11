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
    assert 43300.0 < feature_data["stat_max"].values[0] < 43400.0
    assert 920.0 < feature_data["stat_min"].values[0] < 930.0


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
    assert 7.9 < feature_data["stat_max"].values[0] < 8.0
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


def test_pet_suv_conversion_lean_body_mass():

    data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "mask"),
        pet_suv_conversion="lean_body_mass",
        roi_name="GTV_Mass",
        base_feature_families="statistics"
    )

    feature_data = data[0]

    assert len(feature_data) == 1
    assert 5.9 < feature_data["stat_max"].values[0] < 6.1
    assert 0.1 < feature_data["stat_min"].values[0] < 0.2


def test_pet_suv_conversion_lean_body_mass_bmi():

    data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "mask"),
        pet_suv_conversion="lean_body_mass_bmi",
        roi_name="GTV_Mass",
        base_feature_families="statistics"
    )

    feature_data = data[0]

    assert len(feature_data) == 1
    assert 5.8 < feature_data["stat_max"].values[0] < 5.9
    assert 0.1 < feature_data["stat_min"].values[0] < 0.2


def test_pet_suv_conversion_ideal_body_weight():

    data = extract_features(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "mask"),
        pet_suv_conversion="ideal_body_weight",
        roi_name="GTV_Mass",
        base_feature_families="statistics"
    )

    feature_data = data[0]

    assert len(feature_data) == 1
    assert 6.6 < feature_data["stat_max"].values[0] < 6.7
    assert 0.1 < feature_data["stat_min"].values[0] < 0.2
