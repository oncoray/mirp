import os
import pytest

from mirp._images.ct_image import CTImage
from mirp._masks.base_mask import BaseMask
from mirp.extract_features_and_images import extract_images, extract_features

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.mark.ci
def test_import_native_single_image():

    data = extract_images(
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1"
    )

    image = data[0][0][0]
    mask = data[0][1][0]

    assert isinstance(image, CTImage)
    assert isinstance(mask, BaseMask)

    data = extract_features(
        image = image,
        mask = mask,
        base_feature_families = "statistics"
    )
    pass