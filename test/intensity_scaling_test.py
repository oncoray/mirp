import os
from mirp import extract_features_and_images
from mirp._images.generic_image import GenericImage
from mirp._images.mr_image import MRImage
from mirp._masks.baseMask import BaseMask

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_intensity_scaling():

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1", "mask"),
        roi_name="GTV_Mass",
        tissue_mask_type="none",
        intensity_normalisation="range",
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert len(feature_data) == 1
    assert 0.3 < feature_data["stat_max"].values[0] <= 0.35
    assert 0.0 <= feature_data["stat_min"].values[0] < 0.10

    assert isinstance(image, MRImage)
    assert isinstance(mask, BaseMask)

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1", "mask"),
        roi_name="GTV_Mass",
        tissue_mask_type="none",
        intensity_normalisation="range",
        intensity_scaling=1000.0,
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert len(feature_data) == 1
    assert 300.0 < feature_data["stat_max"].values[0] <= 350.0
    assert 0.0 <= feature_data["stat_min"].values[0] < 100.0

    assert isinstance(image, MRImage)
    assert isinstance(mask, BaseMask)

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        tissue_mask_type="none",
        intensity_normalisation="range",
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert len(feature_data) == 1
    assert 0.4 < feature_data["stat_max"].values[0] <= 0.45
    assert 0.0 <= feature_data["stat_min"].values[0] < 0.05

    assert isinstance(image, GenericImage)
    assert isinstance(mask, BaseMask)

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        tissue_mask_type="none",
        intensity_normalisation="range",
        intensity_scaling=1000.0,
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert len(feature_data) == 1
    assert 400.0 < feature_data["stat_max"].values[0] <= 450.0
    assert 0.0 <= feature_data["stat_min"].values[0] < 50.0

    assert isinstance(image, GenericImage)
    assert isinstance(mask, BaseMask)

