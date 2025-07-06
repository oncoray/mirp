import os

import numpy as np
import pytest

from mirp._images.ct_image import CTImage
from mirp._masks.base_mask import BaseMask
from mirp import extract_features_and_images, extract_mask_labels, extract_image_parameters

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
    assert all(x.sample_name in ["STS_001", "STS_002", "STS_003"] for x in image)
    assert all(x.sample_name in ["STS_001", "STS_002", "STS_003"] for x in mask)

    data = extract_features_and_images(
        image_export_format="native",
        image = image,
        mask = mask,
        base_feature_families = "statistics"
    )

    new_feature_data = [x[0] for x in data]
    new_image = [x[1][0] for x in data]
    new_mask = [x[2][0] for x in data]

    assert all(x.sample_name in ["STS_001", "STS_002", "STS_003"] for x in new_image)
    assert all(x.sample_name in ["STS_001", "STS_002", "STS_003"] for x in new_mask)

    for ii, old_feature_data in enumerate(feature_data):
        assert old_feature_data.equals(new_feature_data[ii])
    for ii, old_image in enumerate(image):
        assert np.array_equal(old_image.get_voxel_grid(), new_image[ii].get_voxel_grid())
    for ii, old_mask in enumerate(mask):
        assert np.array_equal(old_mask.roi.get_voxel_grid(), new_mask[ii].roi.get_voxel_grid())


def test_extract_labels_native_multiple():
    data = extract_features_and_images(
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "dicom", "image"),
        mask_sub_folder=os.path.join("CT", "dicom", "mask"),
        image_modality="CT",
        base_feature_families="statistics"
    )

    image = [x[1][0] for x in data]
    mask = [x[2][0] for x in data]

    assert all(isinstance(x, CTImage) for x in image)
    assert all(isinstance(x, BaseMask) for x in mask)
    assert all(x.sample_name in ["STS_001", "STS_002", "STS_003"] for x in image)
    assert all(x.sample_name in ["STS_001", "STS_002", "STS_003"] for x in mask)

    roi_labels = extract_mask_labels(
        mask = mask
    )
    assert len(roi_labels["roi_label"]) == 3
    assert all(roi_label == "GTV_Mass_CT" for roi_label in roi_labels["roi_label"])


def test_extract_metadata_parameter_native_multiple_dicom():
    data = extract_features_and_images(
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "dicom", "image"),
        mask_sub_folder=os.path.join("CT", "dicom", "mask"),
        image_modality="CT",
        base_feature_families="statistics"
    )

    image = [x[1][0] for x in data]
    mask = [x[2][0] for x in data]

    assert all(isinstance(x, CTImage) for x in image)
    assert all(isinstance(x, BaseMask) for x in mask)
    assert all(x.sample_name in ["STS_001", "STS_002", "STS_003"] for x in image)
    assert all(x.sample_name in ["STS_001", "STS_002", "STS_003"] for x in mask)

    metadata = extract_image_parameters(
        image=image
    )

    assert np.array_equal(metadata["kvp"].values, np.array([140.0, 140.0, 120.0]))
    assert len(metadata["kvp"]) == 3


def test_extract_metadata_parameter_native_multiple_non_dicom():
    data = extract_features_and_images(
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "nifti", "image"),
        mask_sub_folder=os.path.join("CT", "nifti", "mask"),
        image_modality="CT",
        base_feature_families="statistics"
    )

    image = [x[1][0] for x in data]
    mask = [x[2][0] for x in data]

    assert all(isinstance(x, CTImage) for x in image)
    assert all(isinstance(x, BaseMask) for x in mask)
    assert all(x.sample_name in ["STS_001", "STS_002", "STS_003"] for x in image)
    assert all(x.sample_name in ["STS_001", "STS_002", "STS_003"] for x in mask)

    metadata = extract_image_parameters(
        image=image
    )

    assert all(x == "nifti" for x in metadata["file_type"].values)
    assert len(metadata["file_type"]) == 3
