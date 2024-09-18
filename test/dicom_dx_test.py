import os
import pytest

from mirp._images.digital_xray_image import DXImage
from mirp._masks.base_mask import BaseMask
from mirp.extract_features_and_images import extract_features_and_images, extract_images

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.ci
def test_basic_dx_feature_extraction():

    data = extract_features_and_images(
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "planar_imaging", "digital_xray", "MSB-02381", "image"),
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert len(feature_data) == 1
    assert feature_data["stat_max"].values[0] == 16383.0
    assert feature_data["stat_min"].values[0] == 0.0

    assert isinstance(image, DXImage)
    assert isinstance(mask, BaseMask)


def test_dx_image_methods():
    image, mask = extract_images(
        image=os.path.join(CURRENT_DIR, "data", "planar_imaging", "digital_xray", "MSB-02381", "image"),
        image_export_format="native"
    )[0]
    image: DXImage = image[0]
    assert isinstance(image, DXImage)

    # Normalisation test.
    test_image = image.normalise_intensities(
        normalisation_method="standardisation",
        saturation_range=(-3.0, 3.0)
    )
    assert isinstance(image, DXImage)
    assert isinstance(test_image, DXImage)

    # Scaling test.
    test_image = image.scale_intensities(scale=2.0)
    assert isinstance(image, DXImage)
    assert isinstance(test_image, DXImage)
