import os

from mirp.images.ctImage import CTImage
from mirp.masks.baseMask import BaseMask
from mirp.extractFeaturesAndImages import extract_features_and_images

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


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
        resegmentation_method="range",
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