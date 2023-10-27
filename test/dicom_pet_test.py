import os

from mirp.images.petImage import PETImage
from mirp.masks.baseMask import BaseMask
from mirp.extractFeaturesAndImages import extract_features_and_images

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
    assert 7.5 < feature_data["stat_max"].values[0] < 8.0
    assert 0.0 < feature_data["stat_min"].values[0] < 0.5

    assert isinstance(image, PETImage)
    assert isinstance(mask, BaseMask)
