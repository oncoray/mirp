import os.path

import numpy as np
import pytest

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
from mirp.importData.importMask import import_mask
from mirp.importData.imageITKFile import MaskITKFile
from mirp.importData.imageDicomFileRTSTRUCT import MaskDicomFileRTSTRUCT
from mirp.importData.imageNumpyFile import MaskNumpyFile
from mirp.importData.imageNumpyFileStack import MaskNumpyFileStack


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_single_mask_import():

    # Read a Nifti image directly.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskITKFile)
    assert mask_list[0].modality == "generic_mask"

    # Read a DICOM RTSTRUCT file directly.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskDicomFileRTSTRUCT)
    assert mask_list[0].sample_name == "STS_001"
    assert mask_list[0].modality == "rtstruct"

    # Read a numpy image directly.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskNumpyFile)
    assert mask_list[0].modality == "generic_mask"

    # Read a numpy stack.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "mask"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskNumpyFileStack)
    assert mask_list[0].modality == "generic_mask"

    # Read a Nifti image for a specific sample.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        mask_sub_folder=os.path.join("CT", "nifti", "mask"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskITKFile)
    assert mask_list[0].sample_name == "STS_001"
    assert mask_list[0].modality == "generic_mask"

    # Read a DICOM RTSTRUCT file for a specific sample.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        mask_sub_folder=os.path.join("CT", "dicom", "mask"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskDicomFileRTSTRUCT)
    assert mask_list[0].sample_name == "STS_001"
    assert mask_list[0].modality == "rtstruct"

    # Read a numpy image for a specific sample.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        mask_sub_folder=os.path.join("CT", "numpy", "mask"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskNumpyFile)
    assert mask_list[0].sample_name == "STS_001"
    assert mask_list[0].modality == "generic_mask"

    # Read a numpy image stack for a specific sample.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        mask_sub_folder=os.path.join("CT", "numpy_slice", "mask"))
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskNumpyFileStack)
    assert mask_list[0].sample_name == "STS_001"
    assert mask_list[0].modality == "generic_mask"

    # Read a Nifti image by specifying the image name.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti"),
        mask_name="mask")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskITKFile)
    assert mask_list[0].modality == "generic_mask"

    # Read a numpy file by specifying the image name.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy"),
        mask_name="*mask")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskNumpyFile)
    assert mask_list[0].modality == "generic_mask"

    # Read a numpy stack by specifying the image name.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice"),
        mask_name="*mask")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskNumpyFileStack)
    assert mask_list[0].modality == "generic_mask"
