import os.path

import numpy as np
import pytest

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
from mirp.importData.importMask import import_mask

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_single_image_import():

    # Read a Nifti image directly.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageITKFile)

    # Read a DICOM image stack.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageDicomFileStack)
    assert mask_list[0].sample_name == "STS_001"

    # Read a numpy image directly.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageNumpyFile)

    # Read a numpy stack.
    mask_list = import_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageNumpyFileStack)

    # Read a Nifti image for a specific sample.
    mask_list = import_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "nifti", "image"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageITKFile)
    assert mask_list[0].sample_name == "STS_001"

    # Read a DICOM image stack for a specific sample.
    mask_list = import_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "dicom", "image"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageDicomFileStack)
    assert mask_list[0].sample_name == "STS_001"
    assert mask_list[0].modality == "ct"

    # Read a numpy image for a specific sample.
    mask_list = import_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "numpy", "image"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageNumpyFile)
    assert mask_list[0].sample_name == "STS_001"

    # Read a numpy image stack for a specific sample.
    mask_list = import_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "numpy_slice", "image"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageNumpyFileStack)
    assert mask_list[0].sample_name == "STS_001"

    # Read a Nifti image by specifying the image name.
    mask_list = import_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti"),
        image_name="image")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageITKFile)

    # Read a numpy file by specifying the image name.
    mask_list = import_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy"),
        image_name="*image")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageNumpyFile)

    # Read a numpy stack by specifying the image name.
    mask_list = import_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice"),
        image_name="*image")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageNumpyFileStack)

    # Read a DICOM image stack by specifying the modality, the sample name and the file type.
    mask_list = import_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_modality="rtstruct",
        image_file_type="dicom")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageDicomFileStack)
    assert mask_list[0].sample_name == "STS_001"
    assert mask_list[0].modality == "rtstruct"