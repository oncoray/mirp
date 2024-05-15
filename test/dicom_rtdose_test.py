import os
import numpy as np

from mirp._images.rtdose_image import RTDoseImage
from mirp._masks.base_mask import BaseMask
from mirp.extract_features_and_images import extract_features_and_images, extract_images

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_basic_rtdose_feature_extraction():

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "rtdose_images", "Pancreas-CT-CB_001", "rtdose"),
        mask=os.path.join(CURRENT_DIR, "data", "rtdose_images", "Pancreas-CT-CB_001", "mask"),
        roi_name="ROI",
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert len(feature_data) == 1
    assert np.around(feature_data["stat_max"].values[0], 0) == 74.0
    assert np.around(feature_data["stat_min"].values[0], 0) == 7.0

    assert isinstance(image, RTDoseImage)
    assert isinstance(mask, BaseMask)


def test_rtdose_image_methods():
    from mirp._images.rtdose_image import RTDoseImage, GenericImage

    image, mask = extract_images(
        image=os.path.join(CURRENT_DIR, "data", "rtdose_images", "Pancreas-CT-CB_001", "rtdose"),
        mask=os.path.join(CURRENT_DIR, "data", "rtdose_images", "Pancreas-CT-CB_001", "mask"),
        roi_name="ROI",
        image_export_format="native"
    )[0]
    image: RTDoseImage = image[0]
    assert isinstance(image, RTDoseImage)

    # Normalisation test.
    test_image = image.normalise_intensities(
        normalisation_method="standardisation",
        saturation_range=(-3.0, 3.0)
    )
    assert isinstance(image, RTDoseImage)
    assert not isinstance(test_image, RTDoseImage)
    assert isinstance(test_image, GenericImage)

    # Scaling test.
    test_image = image.scale_intensities(scale=2.0)
    assert isinstance(image, RTDoseImage)
    assert not isinstance(test_image, RTDoseImage)
    assert isinstance(test_image, GenericImage)

    # Lowest intensity.
    assert image.get_default_lowest_intensity() == 0.0
