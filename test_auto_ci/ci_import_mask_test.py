import os.path

import pytest

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
from mirp.data_import.import_mask import import_mask
from mirp._data_import.itk_file import MaskITKFile
from mirp._data_import.dicom_file_rtstruct import MaskDicomFileRTSTRUCT
from mirp._data_import.numpy_file import MaskNumpyFile
from mirp._data_import.numpy_file_stack import MaskNumpyFileStack

CURRENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test")


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


def test_multiple_mask_import():
    # Read Nifti _masks directly.
    mask_list = import_mask([
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "nifti", "mask", "mask.nii.gz"),
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "nifti", "mask", "mask.nii.gz")
    ])
    assert len(mask_list) == 2
    assert all(isinstance(mask_object, MaskITKFile) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)

    # Read DICOM RTSTRUCT files.
    mask_list = import_mask([
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "dicom", "mask", "RS.dcm"),
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "dicom", "mask", "RS.dcm")
    ])
    assert len(mask_list) == 2
    assert all(isinstance(mask_object, MaskDicomFileRTSTRUCT) for mask_object in mask_list)
    assert all(mask_object.modality == "rtstruct" for mask_object in mask_list)
    assert {mask_list[0].sample_name, mask_list[1].sample_name} == {"STS_002", "STS_003"}

    # Read a numpy mask directly.
    mask_list = import_mask([
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "numpy", "mask", "STS_002_mask.npy"),
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "numpy", "mask", "STS_003_mask.npy")
    ])
    assert len(mask_list) == 2
    assert all(isinstance(mask_object, MaskNumpyFile) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)

    # Read a numpy stack.
    mask_list = import_mask([
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "numpy_slice", "mask"),
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "numpy_slice", "mask")
    ])
    assert len(mask_list) == 2
    assert all(isinstance(mask_object, MaskNumpyFileStack) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)

    # Read Nifti _masks for specific samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        mask_sub_folder=os.path.join("CT", "nifti", "mask"))
    assert len(mask_list) == 2
    assert all(isinstance(mask_object, MaskITKFile) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)
    assert {mask_list[0].sample_name, mask_list[1].sample_name} == {"STS_002", "STS_003"}

    # Read DICOM RTSTRUCT files for specific samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        mask_sub_folder=os.path.join("CT", "dicom", "mask"))
    assert len(mask_list) == 2
    assert all(isinstance(mask_object, MaskDicomFileRTSTRUCT) for mask_object in mask_list)
    assert all(mask_object.modality == "rtstruct" for mask_object in mask_list)
    assert {mask_list[0].sample_name, mask_list[1].sample_name} == {"STS_002", "STS_003"}

    # Read numpy _masks for specific samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        mask_sub_folder=os.path.join("CT", "numpy", "mask"))
    assert len(mask_list) == 2
    assert all(isinstance(mask_object, MaskNumpyFile) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)
    assert {mask_list[0].sample_name, mask_list[1].sample_name} == {"STS_002", "STS_003"}

    # Read numpy mask stacks for specific samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        mask_sub_folder=os.path.join("CT", "numpy_slice", "mask"))
    assert len(mask_list) == 2
    assert all(isinstance(mask_object, MaskNumpyFileStack) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)
    assert {mask_list[0].sample_name, mask_list[1].sample_name} == {"STS_002", "STS_003"}

    # Read Nifti _masks for all samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask_sub_folder=os.path.join("CT", "nifti", "mask"))
    assert len(mask_list) == 3
    assert all(isinstance(mask_object, MaskITKFile) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)

    # Read DICOM RTSTRUCT files for all samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask_sub_folder=os.path.join("CT", "dicom", "mask"))
    assert len(mask_list) == 3
    assert all(isinstance(mask_object, MaskDicomFileRTSTRUCT) for mask_object in mask_list)
    assert all(mask_object.modality == "rtstruct" for mask_object in mask_list)
    assert {mask_list[0].sample_name, mask_list[1].sample_name, mask_list[2].sample_name} == \
           {"STS_001", "STS_002", "STS_003"}

    # Read numpy _masks for all samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask_sub_folder=os.path.join("CT", "numpy", "mask"))
    assert len(mask_list) == 3
    assert all(isinstance(mask_object, MaskNumpyFile) for mask_object in mask_list)

    # Read numpy image stacks for all samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask_sub_folder=os.path.join("CT", "numpy_slice", "mask"))
    assert len(mask_list) == 3
    assert all(isinstance(mask_object, MaskNumpyFileStack) for mask_object in mask_list)


def test_single_mask_import_flat():
    # Read a Nifti mask directly.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
        sample_name="STS_001",
        mask_name="#_CT_mask")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskITKFile)
    assert mask_list[0].modality == "generic_mask"
    assert mask_list[0].sample_name == "STS_001"

    # Read a DICOM RTSTRUCT _masks.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "dicom"),
        sample_name="STS_001",
        mask_name="*_CT_RS",
        mask_modality="rtstruct")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskDicomFileRTSTRUCT)
    assert mask_list[0].sample_name == "STS_001"
    assert mask_list[0].modality == "rtstruct"

    # Read a numpy mask directly.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy"),
        sample_name="STS_001",
        mask_name="CT_#_mask")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskNumpyFile)
    assert mask_list[0].sample_name == "STS_001"

    # Read a numpy mask.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy_slice"),
        sample_name="STS_001",
        mask_name="CT_#_*_mask"
    )
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskNumpyFileStack)
    assert mask_list[0].sample_name == "STS_001"

    # Configurations that produce errors.
    with pytest.raises(ValueError):
        _ = import_mask(
            os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
            sample_name="STS_001")


def test_multiple_mask_import_flat():

    # Read Nifti _masks for specific samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
        sample_name=["STS_002", "STS_003"],
        mask_name="#_CT_mask"
    )
    assert len(mask_list) == 2
    assert all(isinstance(mask_object, MaskITKFile) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)
    assert set(mask_object.sample_name for mask_object in mask_list) == {"STS_002", "STS_003"}

    # Read DICOM RTSTRUCT files for a specific samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "dicom"),
        sample_name=["STS_002", "STS_003"],
        mask_name="#_CT_RS"
    )
    assert len(mask_list) == 2
    assert all(isinstance(mask_object, MaskDicomFileRTSTRUCT) for mask_object in mask_list)
    assert set(mask_object.sample_name for mask_object in mask_list) == {"STS_002", "STS_003"}
    assert all(mask_object.modality == "rtstruct" for mask_object in mask_list)

    # Read numpy _masks for specific samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy"),
        sample_name=["STS_002", "STS_003"],
        mask_name="CT_#_mask"
    )
    assert len(mask_list) == 2
    assert all(isinstance(mask_object, MaskNumpyFile) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)
    assert set(mask_object.sample_name for mask_object in mask_list) == {"STS_002", "STS_003"}

    # Read numpy mask stacks for specific samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy_slice"),
        sample_name=["STS_002", "STS_003"],
        mask_name="CT_#_*_mask"
    )
    assert len(mask_list) == 2
    assert all(isinstance(mask_object, MaskNumpyFileStack) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)
    assert set(mask_object.sample_name for mask_object in mask_list) == {"STS_002", "STS_003"}

    # Read Nifti _masks for all samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
        mask_name="#_CT_image")
    assert len(mask_list) == 3
    assert all(isinstance(mask_object, MaskITKFile) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)
    assert set(mask_object.sample_name for mask_object in mask_list) == {"STS_001", "STS_002", "STS_003"}

    # Read DICOM RTSTRUCT files for all samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "dicom"),
        mask_name="#_CT_*"
    )
    assert len(mask_list) == 3
    assert all(isinstance(mask_object, MaskDicomFileRTSTRUCT) for mask_object in mask_list)
    assert set(mask_object.sample_name for mask_object in mask_list) == {"STS_001", "STS_002", "STS_003"}
    assert all(mask_object.modality == "rtstruct" for mask_object in mask_list)

    # Read numpy _masks for all samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy"),
        mask_name="CT_#_mask"
    )
    assert len(mask_list) == 3
    assert all(isinstance(mask_object, MaskNumpyFile) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)
    assert set(mask_object.sample_name for mask_object in mask_list) == {"STS_001", "STS_002", "STS_003"}

    # Read numpy mask stacks for all samples.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy_slice"),
        mask_name="CT_#_*_mask"
    )
    assert len(mask_list) == 3
    assert all(isinstance(mask_object, MaskNumpyFileStack) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)
    assert set(mask_object.sample_name for mask_object in mask_list) == {"STS_001", "STS_002", "STS_003"}

    # Read Nifti _masks for all samples without specifying the sample name in the image name.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
        mask_name="*CT_mask")
    assert len(mask_list) == 3
    assert all(isinstance(mask_object, MaskITKFile) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)

    # Read DICOM RTSTRUCT files for all samples without specifying the image name.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "dicom"),
        mask_modality="rtstruct"
    )
    assert len(mask_list) == 9
    assert all(isinstance(mask_object, MaskDicomFileRTSTRUCT) for mask_object in mask_list)
    assert set(mask_object.sample_name for mask_object in mask_list) == {"STS_001", "STS_002", "STS_003"}
    assert all(mask_object.modality == "rtstruct" for mask_object in mask_list)

    # Read numpy _masks for all samples without specifying the sample name in the image name.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy"),
        mask_name="CT_*_mask"
    )
    assert len(mask_list) == 3
    assert all(isinstance(mask_object, MaskNumpyFile) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)

    # Read numpy mask stacks for all samples without specifying the sample name in the image name.
    mask_list = import_mask(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy_slice"),
        mask_name="CT_*_mask"
    )
    assert len(mask_list) == 3
    assert all(isinstance(mask_object, MaskNumpyFileStack) for mask_object in mask_list)
    assert all(mask_object.modality == "generic_mask" for mask_object in mask_list)


def test_mask_import_flat_poor_naming():
    """
    Tests whether we can select files if their naming convention is poor, e.g. sample_1, sample_11, sample_111.
    :return:
    """
    # Test correctness when all names are provided.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti_poor_names"),
        sample_name=["STS_1", "STS_11", "STS_111"],
        mask_name="#_*_mask")
    assert len(mask_list) == 3
    assert all(isinstance(mask_object, MaskITKFile) for mask_object in mask_list)
    assert set(mask_object.sample_name for mask_object in mask_list) == {"STS_1", "STS_11", "STS_111"}

    # Test correctness when no names are provided, but the naming structure is clear.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti_poor_names"),
        mask_name="#_PET_mask")
    assert len(mask_list) == 3
    assert all(isinstance(mask_object, MaskITKFile) for mask_object in mask_list)
    assert set(mask_object.sample_name for mask_object in mask_list) == {"STS_1", "STS_11", "STS_111"}

    # Test correctness when no names are provided.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti_poor_names"),
        mask_name="*mask")
    assert len(mask_list) == 3
    assert all(isinstance(mask_object, MaskITKFile) for mask_object in mask_list)
