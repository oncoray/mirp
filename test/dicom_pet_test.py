import os
import pytest

from mirp._images.pet_image import PETImage
from mirp._masks.base_mask import BaseMask
from mirp.extract_features_and_images import extract_features_and_images, extract_images

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_basic_pet_feature_extraction():

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "mask"),
        roi_name="GTV_Mass",
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert len(feature_data) == 1
    assert 7.5 < feature_data["stat_max"].values[0] < 8.5
    assert 0.0 < feature_data["stat_min"].values[0] < 0.5

    assert isinstance(image, PETImage)
    assert isinstance(mask, BaseMask)


def test_pet_image_methods():
    from mirp._images.pet_image import PETImage, GenericImage

    image, mask = extract_images(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "mask"),
        roi_name="GTV_Mass",
        image_export_format="native"
    )[0]
    image: PETImage = image[0]
    assert isinstance(image, PETImage)

    # Normalisation test.
    test_image = image.normalise_intensities(
        normalisation_method="standardisation",
        saturation_range=(-3.0, 3.0)
    )
    assert isinstance(image, PETImage)
    assert not isinstance(test_image, PETImage)
    assert isinstance(test_image, GenericImage)

    # Scaling test.
    test_image = image.scale_intensities(scale=2.0)
    assert isinstance(image, PETImage)
    assert not isinstance(test_image, PETImage)
    assert isinstance(test_image, GenericImage)

    # Lowest intensity.
    assert image.get_default_lowest_intensity() == 0.0


@pytest.mark.skip(reason="digital reference object licensing unclear")
def test_pet_dro():
    import numpy as np

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image="TODO:SET",
        mask="TODO:SET",
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert len(feature_data) == 6

    x_1 = feature_data[feature_data.image_mask_label == "region_1"]
    assert 3.995 < x_1["stat_max"].values[0] < 4.005
    assert 0.635 < x_1["stat_min"].values[0] < 0.645
    assert 1.295 < x_1["stat_mean"].values[0] < 1.505
    assert 0.875 < np.sqrt(x_1["stat_var"].values[0]) < 1.095

    x_2 = feature_data[feature_data.image_mask_label == "region_2"]
    assert 3.995 < x_2["stat_max"].values[0] < 4.005
    assert 3.995 < x_2["stat_min"].values[0] < 4.005
    assert 3.995 < x_2["stat_mean"].values[0] < 4.005
    assert -0.005 < np.sqrt(x_2["stat_var"].values[0]) < 0.005

    x_3 = feature_data[feature_data.image_mask_label == "region_3"]
    assert 4.105 < x_3["stat_max"].values[0] < 4.115
    assert 0.995 < x_3["stat_min"].values[0] < 1.005
    assert 1.015 < x_3["stat_mean"].values[0] < 1.035
    assert 0.235 < np.sqrt(x_3["stat_var"].values[0]) < 0.325

    x_4 = feature_data[feature_data.image_mask_label == "region_4"]
    assert 0.995 < x_4["stat_max"].values[0] < 1.005
    assert -0.115 < x_4["stat_min"].values[0] < -0.105
    assert 0.985 < x_4["stat_mean"].values[0] < 0.995
    assert 0.085 < np.sqrt(x_4["stat_var"].values[0]) < 0.115

    x_5 = feature_data[feature_data.image_mask_label == "region_5"]
    assert 0.895 < x_5["stat_max"].values[0] < 0.905
    assert 0.095 < x_5["stat_min"].values[0] < 0.105
    assert 0.465 < x_5["stat_mean"].values[0] < 0.535
    assert 0.395 < np.sqrt(x_5["stat_var"].values[0]) < 0.405

    x_6 = feature_data[feature_data.image_mask_label == "region_6"]
    assert 0.895 < x_6["stat_max"].values[0] < 0.905
    assert 0.095 < x_6["stat_min"].values[0] < 0.105
    assert 0.475 < x_6["stat_mean"].values[0] < 0.525
    assert 0.395 < np.sqrt(x_6["stat_var"].values[0]) < 0.405

    assert isinstance(image, PETImage)
    assert isinstance(mask, BaseMask)
