import os.path

import numpy as np
import pytest

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
from mirp.importData.importMask import import_mask

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_single_image_import():

    # Read a Nifti image directly.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageITKFile)

    # Read a DICOM image stack.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageDicomFileStack)
    assert mask_list[0].sample_name == "STS_001"

    # Read a numpy image directly.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageNumpyFile)

    # Read a numpy stack.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "mask"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageNumpyFileStack)

    # Read a Nifti image for a specific sample.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        mask_sub_folder=os.path.join("CT", "nifti", "mask"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageITKFile)
    assert mask_list[0].sample_name == "STS_001"

    # Read a DICOM image stack for a specific sample.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        mask_sub_folder=os.path.join("CT", "dicom", "mask"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageDicomFileStack)
    assert mask_list[0].sample_name == "STS_001"
    assert mask_list[0].modality == "ct"

    # Read a numpy image for a specific sample.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        mask_sub_folder=os.path.join("CT", "numpy", "mask"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageNumpyFile)
    assert mask_list[0].sample_name == "STS_001"

    # Read a numpy image stack for a specific sample.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        mask_sub_folder=os.path.join("CT", "numpy_slice", "mask"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageNumpyFileStack)
    assert mask_list[0].sample_name == "STS_001"

    # Read a Nifti image by specifying the image name.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti"),
        mask_name="mask")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageITKFile)

    # Read a numpy file by specifying the image name.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy"),
        mask_name="*mask")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageNumpyFile)

    # Read a numpy stack by specifying the image name.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice"),
        mask_name="*mask")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageNumpyFileStack)

    # Read a DICOM image stack by specifying the modality, the sample name and the file type.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        mask_modality="rtstruct",
        mask_file_type="dicom")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], ImageDicomFileStack)
    assert mask_list[0].sample_name == "STS_001"
    assert mask_list[0].modality == "rtstruct"
