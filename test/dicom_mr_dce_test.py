import os

from mirp._images.generic_image import GenericImage
from mirp._images.mr_dce_image import MRDCEImage
from mirp.extract_features_and_images import extract_features_and_images, extract_images

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_basic_dce_mr_feature_extraction():
    # Separate DCE images (PE-1)
    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "dce_images_mr", "UCSF-BR-06", "image_dce_pe1"),
        mask=os.path.join(CURRENT_DIR, "data", "dce_images_mr", "UCSF-BR-06", "mask"),
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]

    assert len(feature_data) == 1
    assert 390.0 < feature_data["stat_max"].values[0] < 400.0
    assert feature_data["stat_min"].values[0] == 0.0

    assert isinstance(image, MRDCEImage)


def test_dce_mr_image_methods():
    image, mask = extract_images(
        image=os.path.join(CURRENT_DIR, "data", "dce_images_mr", "UCSF-BR-06", "image_dce_pe1"),
        image_export_format="native"
    )[0]
    image: MRDCEImage = image[0]
    assert isinstance(image, MRDCEImage)

    # Normalisation test.
    test_image = image.normalise_intensities(
        normalisation_method="standardisation",
        saturation_range=(-3.0, 3.0)
    )
    assert isinstance(image, MRDCEImage)
    assert not isinstance(test_image, MRDCEImage)
    assert isinstance(test_image, GenericImage)

    # Scaling test.
    test_image = image.scale_intensities(scale=2.0)
    assert isinstance(image, MRDCEImage)
    assert not isinstance(test_image, MRDCEImage)
    assert isinstance(test_image, GenericImage)
