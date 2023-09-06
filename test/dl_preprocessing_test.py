import pytest
import os.path

from mirp.settings.settingsClass import SettingsClass
from mirp.deepLearningPreprocessing import deep_learning_preprocessing

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_extract_image_crop():
    import numpy as np

    # Configure settings.
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none"
    )

    # No cropping.
    data = deep_learning_preprocessing(
        output_slices=False,
        crop_size=None,
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        settings=settings
    )

    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert mask.shape == (60, 201, 204)

    # Split into splices (with mask present).
    data = deep_learning_preprocessing(
        output_slices=True,
        crop_size=None,
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        settings=settings
    )

    images = data[0][0]
    masks = data[0][1]

    assert len(images) == 26
    assert all(image.shape == (1, 201, 204) for image in images)
    assert len(masks) == 26
    assert all(mask.shape == (1, 201, 204) for mask in masks)

    # Crop to size.
    data = deep_learning_preprocessing(
        output_slices=False,
        crop_size=[20, 50, 50],
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        settings=settings
    )

    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (20, 50, 50)
    assert mask.shape == (20, 50, 50)

    # Crop to size in-plane.
    data = deep_learning_preprocessing(
        output_slices=False,
        crop_size=[50, 50],
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        settings=settings
    )

    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 50, 50)
    assert mask.shape == (60, 50, 50)

    # Split into splices (with mask present) and crop
    data = deep_learning_preprocessing(
        output_slices=True,
        crop_size=[50, 50],
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        settings=settings
    )

    images = data[0][0]
    masks = data[0][1]

    assert len(images) == 26
    assert all(image.shape == (1, 50, 50) for image in images)
    assert len(masks) == 26
    assert all(mask.shape == (1, 50, 50) for mask in masks)

def test_normalisation_saturation():
    ...