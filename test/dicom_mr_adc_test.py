import os

from mirp._images.generic_image import GenericImage
from mirp._images.mr_adc_image import MRADCImage
from mirp.extract_features_and_images import extract_features_and_images, extract_images

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_basic_adc_mr_feature_extraction():

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "adc_images_mr", "SCAN_001", "adc_image"),
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]

    assert len(feature_data) == 1
    assert 9100.0 < feature_data["stat_max"].values[0] < 9200.0
    assert feature_data["stat_min"].values[0] == 0.0

    assert isinstance(image, MRADCImage)


def test_adc_mr_image_methods():
    image, mask = extract_images(
        image=os.path.join(CURRENT_DIR, "data", "adc_images_mr", "SCAN_001", "adc_image"),
        image_export_format="native"
    )[0]
    image: MRADCImage = image[0]
    assert isinstance(image, MRADCImage)

    # Normalisation test.
    test_image = image.normalise_intensities(
        normalisation_method="standardisation",
        saturation_range=(-3.0, 3.0)
    )
    assert isinstance(image, MRADCImage)
    assert not isinstance(test_image, MRADCImage)
    assert isinstance(test_image, GenericImage)

    # Scaling test.
    test_image = image.scale_intensities(scale=2.0)
    assert isinstance(image, MRADCImage)
    assert not isinstance(test_image, MRADCImage)
    assert isinstance(test_image, GenericImage)

    # Lowest intensity.
    assert image.get_default_lowest_intensity() == 0.0
