import os

from mirp.images.rtdoseImage import RTDoseImage
from mirp.masks.baseMask import BaseMask
from mirp.extractFeaturesAndImages import extract_features_and_images

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_basic_rtdose_feature_extraction():

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "rtdose_images", "VS-SEG-001", "rtdose"),
        mask=os.path.join(CURRENT_DIR, "data", "rtdose_images", "VS-SEG-001", "mask"),
        roi_name="TV",
        base_feature_families="statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert len(feature_data) == 1
    assert feature_data["stat_max"].values[0] == 250.0
    assert feature_data["stat_min"].values[0] == -1000.0

    assert isinstance(image, RTDoseImage)
    assert isinstance(mask, BaseMask)
