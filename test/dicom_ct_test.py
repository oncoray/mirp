import os
import pytest

from mirp._images.ct_image import CTImage
from mirp._masks.base_mask import BaseMask
from mirp.extract_features_and_images import extract_features_and_images, extract_images

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.ci
def test_basic_ct_feature_extraction():

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        base_feature_families="statistics",
        resegmentation_intensity_range=[-1000.0, 250.0]
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert len(feature_data) == 1
    assert feature_data["stat_max"].values[0] == 250.0
    assert feature_data["stat_min"].values[0] == -1000.0

    assert isinstance(image, CTImage)
    assert isinstance(mask, BaseMask)


def test_ct_image_methods():
    from mirp._images.ct_image import CTImage, GenericImage

    image, mask = extract_images(
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        image_export_format="native"
    )[0]
    image: CTImage = image[0]
    assert isinstance(image, CTImage)

    # Normalisation test.
    test_image = image.normalise_intensities(
        normalisation_method="standardisation",
        saturation_range=(-3.0, 3.0)
    )
    assert isinstance(image, CTImage)
    assert not isinstance(test_image, CTImage)
    assert isinstance(test_image, GenericImage)

    # Scaling test.
    test_image = image.scale_intensities(scale=2.0)
    assert isinstance(image, CTImage)
    assert not isinstance(test_image, CTImage)
    assert isinstance(test_image, GenericImage)

    # Lowest intensity.
    assert image.get_default_lowest_intensity() == -1000.0
