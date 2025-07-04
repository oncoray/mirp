import os

import numpy as np
import pytest

from mirp._images.ct_image import CTImage
from mirp._masks.base_mask import BaseMask
from mirp.extract_features_and_images import extract_features_and_images

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.mark.ci
def test_import_native_single_image():

    data = extract_features_and_images(
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        base_feature_families = "statistics"
    )

    feature_data = data[0][0]
    image = data[0][1][0]
    mask = data[0][2][0]

    assert isinstance(image, CTImage)
    assert isinstance(mask, BaseMask)

    data = extract_features_and_images(
        image_export_format="native",
        image = image,
        mask = mask,
        base_feature_families = "statistics"
    )

    new_feature_data = data[0][0]
    new_image = data[0][1][0]
    new_mask = data[0][2][0]

    assert isinstance(image, CTImage)
    assert isinstance(mask, BaseMask)
    assert np.array_equal(image.get_voxel_grid(), new_image.get_voxel_grid())
    assert np.array_equal(mask.roi.get_voxel_grid(), new_mask.roi.get_voxel_grid())
    assert feature_data.equals(new_feature_data)

def test_import_native_multiple_images():

    data = extract_features_and_images(
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "nifti", "image"),
        mask_sub_folder=os.path.join("CT", "nifti", "mask"),
        image_modality="CT",
        base_feature_families = "statistics"
    )

    feature_data = [x[0] for x in data]
    image = [x[1][0] for x in data]
    mask = [x[2][0] for x in data]

    assert all(isinstance(x, CTImage) for x in image)
    assert all(isinstance(x, BaseMask) for x in mask)

    data = extract_features_and_images(
        image_export_format="native",
        image = image,
        mask = mask,
        base_feature_families = "statistics"
    )

    new_feature_data = [x[0] for x in data]
    new_image = [x[1][0] for x in data]
    new_mask = [x[2][0] for x in data]

    for ii, old_feature_data in enumerate(feature_data):
        assert old_feature_data.equals(new_feature_data[ii])
    for ii, old_image in enumerate(image):
        assert np.array_equal(old_image.get_voxel_grid(), new_image[ii].get_voxel_grid)
    for ii, old_mask in enumerate(mask):
        assert np.array_equal(old_mask.roi.get_voxel_grid(), new_mask[ii].roi.get_voxel_grid)