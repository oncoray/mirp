import os

from mirp.images.mrImage import MRImage
from mirp.masks.baseMask import BaseMask
from mirp.extractFeaturesAndImages import extract_features_and_images

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_basic_mr_t1_feature_extraction():

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1", "mask"),
        roi_name="GTV_Mass",
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert len(feature_data) == 1
    assert 800.0 < feature_data["stat_max"].values[0] < 900.0
    assert 0.0 < feature_data["stat_min"].values[0] < 100.0

    assert isinstance(image, MRImage)
    assert isinstance(mask, BaseMask)


def test_bias_field_correction_t1():

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1", "mask"),
        roi_name="GTV_Mass",
        bias_field_correction=True,
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert len(feature_data) == 1
    assert 800.0 < feature_data["stat_max"].values[0] < 900.0
    assert 0.0 < feature_data["stat_min"].values[0] < 100.0

    assert isinstance(image, MRImage)
    assert isinstance(mask, BaseMask)


def test_basic_mr_t2_feature_extraction():

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T2", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T2", "mask"),
        roi_name="GTV_Mass",
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert len(feature_data) == 1
    assert 800.0 < feature_data["stat_max"].values[0] < 900.0
    assert 0.0 < feature_data["stat_min"].values[0] < 100.0

    assert isinstance(image, MRImage)
    assert isinstance(mask, BaseMask)


def test_bias_field_correction_t2():

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T2", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T2", "mask"),
        roi_name="GTV_Mass",
        bias_field_correction=True,
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert len(feature_data) == 1
    assert 800.0 < feature_data["stat_max"].values[0] < 900.0
    assert 0.0 < feature_data["stat_min"].values[0] < 100.0

    assert isinstance(image, MRImage)
    assert isinstance(mask, BaseMask)