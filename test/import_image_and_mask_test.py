import copy
import os.path
import pytest

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
from mirp.data_import.import_image_and_mask import import_image_and_mask
from mirp._data_import.itk_file import ImageITKFile, MaskITKFile
from mirp._data_import.dicom_file_stack import ImageDicomFileStack
from mirp._data_import.dicom_file_rtstruct import MaskDicomFileRTSTRUCT
from mirp._data_import.numpy_file import ImageNumpyFile, MaskNumpyFile
from mirp._data_import.numpy_file_stack import ImageNumpyFileStack, MaskNumpyFileStack


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_single_image_and_mask_import():

    # Read a Nifti image and mask set directly.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"))
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageITKFile)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskITKFile)
    assert image_list[0].associated_masks[0].modality == "generic_mask"

    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=[
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"),
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "nifti", "mask", "mask.nii.gz")
        ]
    )
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageITKFile)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskITKFile)
    assert image_list[0].associated_masks[0].file_path == os.path.join(
        CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz")
    assert image_list[0].associated_masks[0].modality == "generic_mask"

    # Read a DICOM image stack and mask directly.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm")
    )
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageDicomFileStack)
    assert image_list[0].modality == "ct"
    assert image_list[0].sample_name == "STS_001"
    assert len(image_list[0].associated_masks) == 1
    assert image_list[0].associated_masks[0].file_path == os.path.join(
        CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm")
    assert image_list[0].associated_masks[0].modality == "rtstruct"

    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=[
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "dicom", "mask", "RS.dcm")
        ]
    )
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageDicomFileStack)
    assert image_list[0].modality == "ct"
    assert image_list[0].sample_name == "STS_001"
    assert len(image_list[0].associated_masks) == 1
    assert image_list[0].associated_masks[0].file_path == os.path.join(
        CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm")
    assert image_list[0].associated_masks[0].modality == "rtstruct"

    # Read a numpy image and mask directly.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy")
    )
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFile)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskNumpyFile)
    assert image_list[0].associated_masks[0].modality == "generic_mask"

    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"),
        mask=[
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy"),
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "numpy", "mask", "STS_002_mask.npy")
        ]
    )
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFile)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskNumpyFile)
    assert image_list[0].associated_masks[0].modality == "generic_mask"

    # Read a numpy image and mask stack.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "mask")
    )
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFileStack)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskNumpyFileStack)
    assert image_list[0].associated_masks[0].modality == "generic_mask"

    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"),
        mask=[
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "mask"),
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "numpy_slice", "mask")
        ]
    )
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFileStack)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskNumpyFileStack)
    assert image_list[0].associated_masks[0].modality == "generic_mask"

    # Read a Nifti image and mask for a specific sample.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "nifti", "image"),
        mask_sub_folder=os.path.join("CT", "nifti", "mask"))
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageITKFile)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskITKFile)
    assert image_list[0].associated_masks[0].modality == "generic_mask"

    # Read DICOM image and mask file for a specific sample.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "dicom", "image"),
        mask_sub_folder=os.path.join("CT", "dicom", "mask"))
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageDicomFileStack)
    assert image_list[0].modality == "ct"
    assert image_list[0].sample_name == "STS_001"
    assert len(image_list[0].associated_masks) == 1
    assert image_list[0].associated_masks[0].file_path == os.path.join(
        CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm")
    assert image_list[0].associated_masks[0].modality == "rtstruct"

    # Read a numpy image and mask for a specific sample.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "numpy", "image"),
        mask_sub_folder=os.path.join("CT", "numpy", "mask"))
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFile)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskNumpyFile)
    assert image_list[0].associated_masks[0].modality == "generic_mask"

    # Read a numpy image and mask stack for a specific sample.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "numpy_slice", "image"),
        mask_sub_folder=os.path.join("CT", "numpy_slice", "mask"))
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFileStack)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskNumpyFileStack)
    assert image_list[0].associated_masks[0].modality == "generic_mask"

    # Read a Nifti image by specifying the image name.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti"),
        image_name="image",
        mask_name="mask")
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageITKFile)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskITKFile)
    assert image_list[0].associated_masks[0].modality == "generic_mask"

    # Read a numpy image and mask files by specifying the image name.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy"),
        image_name="*image",
        mask_name="*mask")
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFile)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskNumpyFile)
    assert image_list[0].associated_masks[0].modality == "generic_mask"

    # Read a numpy stack by specifying the image name.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice"),
        image_name="*image",
        mask_name="*mask")
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFileStack)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskNumpyFileStack)
    assert image_list[0].associated_masks[0].modality == "generic_mask"


def test_multiple_image_and_mask_import():
    # Read Nifti _images and _masks directly.
    image_list = import_image_and_mask(
        image=[
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "nifti", "image", "image.nii.gz"),
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "nifti", "image", "image.nii.gz")
        ],
        mask=[
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "nifti", "mask", "mask.nii.gz"),
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "nifti", "mask", "mask.nii.gz")
        ])
    assert len(image_list) == 2
    assert all(isinstance(image, ImageITKFile) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskITKFile) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)

    # Read DICOM _images and _masks.
    image_list = import_image_and_mask(
        image=[
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "dicom", "image"),
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "dicom", "image")
        ],
        mask=[
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "dicom", "mask", "RS.dcm"),
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "dicom", "mask", "RS.dcm")
        ])
    assert len(image_list) == 2
    assert all(isinstance(image, ImageDicomFileStack) for image in image_list)
    assert all(image.modality == "ct" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskDicomFileRTSTRUCT) for image in image_list)
    assert all(image.associated_masks[0].modality == "rtstruct" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read numpy _images and _masks directly.
    image_list = import_image_and_mask(
        image=[
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "numpy", "image", "STS_002_image.npy"),
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "numpy", "image", "STS_003_image.npy")
        ],
        mask=[
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "numpy", "mask", "STS_002_mask.npy"),
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "numpy", "mask", "STS_003_mask.npy")
        ])
    assert len(image_list) == 2
    assert all(isinstance(image, ImageNumpyFile) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskNumpyFile) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)

    # Read numpy and image stacks.
    image_list = import_image_and_mask(
        image=[
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "numpy_slice", "image"),
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "numpy_slice", "image")
        ],
        mask=[
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "numpy_slice", "mask"),
            os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "numpy_slice", "mask")
        ])
    assert len(image_list) == 2
    assert all(isinstance(image, ImageNumpyFileStack) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskNumpyFileStack) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)

    # Read Nifti _images and _masks for specific samples.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        image_sub_folder=os.path.join("CT", "nifti", "image"),
        mask_sub_folder=os.path.join("CT", "nifti", "mask"))
    assert len(image_list) == 2
    assert all(isinstance(image, ImageITKFile) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskITKFile) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read DICOM _images and _masks for specific samples.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        image_sub_folder=os.path.join("CT", "dicom", "image"),
        mask_sub_folder=os.path.join("CT", "dicom", "mask"))
    assert len(image_list) == 2
    assert all(isinstance(image, ImageDicomFileStack) for image in image_list)
    assert all(image.modality == "ct" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskDicomFileRTSTRUCT) for image in image_list)
    assert all(image.associated_masks[0].modality == "rtstruct" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read numpy _images and _masks for specific samples.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        image_sub_folder=os.path.join("CT", "numpy", "image"),
        mask_sub_folder=os.path.join("CT", "numpy", "mask"))
    assert len(image_list) == 2
    assert all(isinstance(image, ImageNumpyFile) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskNumpyFile) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read numpy image and mask stacks for specific samples.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        image_sub_folder=os.path.join("CT", "numpy_slice", "image"),
        mask_sub_folder=os.path.join("CT", "numpy_slice", "mask"))
    assert len(image_list) == 2
    assert all(isinstance(image, ImageNumpyFileStack) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskNumpyFileStack) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read Nifti image and _masks for all samples.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "nifti", "image"),
        mask_sub_folder=os.path.join("CT", "nifti", "mask"))
    assert len(image_list) == 3
    assert all(isinstance(image, ImageITKFile) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskITKFile) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read DICOM _images and _masks for all samples.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "dicom", "image"),
        mask_sub_folder=os.path.join("CT", "dicom", "mask")
    )
    assert len(image_list) == 3
    assert all(isinstance(image, ImageDicomFileStack) for image in image_list)
    assert all(image.modality == "ct" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskDicomFileRTSTRUCT) for image in image_list)
    assert all(image.associated_masks[0].modality == "rtstruct" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read numpy _images and _masks for all samples.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "numpy", "image"),
        mask_sub_folder=os.path.join("CT", "numpy", "mask"))
    assert len(image_list) == 3
    assert all(isinstance(image, ImageNumpyFile) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskNumpyFile) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read numpy image and mask stacks for all samples.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "numpy_slice", "image"),
        mask_sub_folder=os.path.join("CT", "numpy_slice", "mask"))
    assert len(image_list) == 3
    assert all(isinstance(image, ImageNumpyFileStack) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskNumpyFileStack) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)


def test_single_image_and_mask_import_flat():
    # Read a Nifti image and mask from a flat directory.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
        sample_name="STS_001",
        image_name="#_CT_image",
        mask_name="#_CT_mask")
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageITKFile)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskITKFile)
    assert image_list[0].associated_masks[0].modality == "generic_mask"
    assert image_list[0].sample_name == image_list[0].associated_masks[0].sample_name

    # Read a DICOM image and mask from a flat directory.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "dicom"),
        sample_name="STS_001",
        image_modality="ct",
        mask_name="#_CT_RS",
        mask_modality="rtstruct")
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageDicomFileStack)
    assert image_list[0].modality == "ct"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskDicomFileRTSTRUCT)
    assert image_list[0].associated_masks[0].modality == "rtstruct"
    assert image_list[0].sample_name == image_list[0].associated_masks[0].sample_name

    # Read a numpy image and mask from a flat directory.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy"),
        sample_name="STS_001",
        image_name="CT_#_image",
        mask_name="CT_#_mask"
    )
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFile)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskNumpyFile)
    assert image_list[0].associated_masks[0].modality == "generic_mask"
    assert image_list[0].sample_name == image_list[0].associated_masks[0].sample_name

    # Read a numpy image and mask from a flat directory.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy_slice"),
        sample_name="STS_001",
        image_name="CT_#_*_image",
        mask_name="CT_#_*_mask"
    )
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFileStack)
    assert image_list[0].modality == "generic"
    assert len(image_list[0].associated_masks) == 1
    assert isinstance(image_list[0].associated_masks[0], MaskNumpyFileStack)
    assert image_list[0].associated_masks[0].modality == "generic_mask"
    assert image_list[0].sample_name == image_list[0].associated_masks[0].sample_name

    # Configurations that produce errors.
    with pytest.raises(ValueError):
        _ = import_image_and_mask(
            image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
            sample_name="STS_001")


def test_multiple_image_and_mask_import_flat():

    # Read Nifti _images and _masks for specific samples in a flat directory.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
        sample_name=["STS_002", "STS_003"],
        image_name="#_CT_image",
        mask_name="#_CT_mask"
    )
    assert len(image_list) == 2
    assert all(isinstance(image, ImageITKFile) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskITKFile) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read DICOM _images and _masks for a specific samples in a flat directory.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "dicom"),
        sample_name=["STS_002", "STS_003"],
        image_modality="ct",
        mask_name="#_CT_RS",
        mask_modality="rtstruct"
    )
    assert len(image_list) == 2
    assert all(isinstance(image, ImageDicomFileStack) for image in image_list)
    assert all(image.modality == "ct" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskDicomFileRTSTRUCT) for image in image_list)
    assert all(image.associated_masks[0].modality == "rtstruct" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read numpy _images and _masks for specific samples in a flat directory.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy"),
        sample_name=["STS_002", "STS_003"],
        image_name="CT_#_image",
        mask_name="CT_#_mask"
    )
    assert len(image_list) == 2
    assert all(isinstance(image, ImageNumpyFile) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskNumpyFile) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read numpy image and mask stacks for specific samples in a flat directory.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy_slice"),
        sample_name=["STS_002", "STS_003"],
        image_name="CT_#_*_image",
        mask_name="CT_#_*_mask"
    )
    assert len(image_list) == 2
    assert all(isinstance(image, ImageNumpyFileStack) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskNumpyFileStack) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read Nifti _images and _masks for all samples in a flat directory.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
        image_name="#_CT_image",
        mask_name="#_CT_mask"
    )
    assert len(image_list) == 3
    assert all(isinstance(image, ImageITKFile) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskITKFile) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read DICOM _images and _masks for all samples in a flat directory.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "dicom"),
        image_modality="ct",
        mask_name="#_CT_RS",
        mask_modality="rtstruct"
    )
    assert len(image_list) == 3
    assert all(isinstance(image, ImageDicomFileStack) for image in image_list)
    assert all(image.modality == "ct" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskDicomFileRTSTRUCT) for image in image_list)
    assert all(image.associated_masks[0].modality == "rtstruct" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read numpy _images and _masks for all samples in a flat directory.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy"),
        image_name="CT_#_image",
        mask_name="CT_#_mask"
    )
    assert len(image_list) == 3
    assert all(isinstance(image, ImageNumpyFile) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskNumpyFile) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)

    # Read numpy _images and _masks for all samples in a flat directory.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy_slice"),
        image_name="CT_#_*_image",
        mask_name="CT_#_*_mask"
    )
    assert len(image_list) == 3
    assert all(isinstance(image, ImageNumpyFileStack) for image in image_list)
    assert all(image.modality == "generic" for image in image_list)
    assert all(len(image.associated_masks) == 1 for image in image_list)
    assert all(isinstance(image.associated_masks[0], MaskNumpyFileStack) for image in image_list)
    assert all(image.associated_masks[0].modality == "generic_mask" for image in image_list)
    assert all(image.sample_name == image.associated_masks[0].sample_name for image in image_list)


def test_failure_multiple_image_and_mask_import():
    # DICOM stack and _masks for all samples, but with incorrect instructions.
    # No matching file type.
    with pytest.raises(ValueError) as exception_info:
        image_list = import_image_and_mask(
            image=os.path.join(CURRENT_DIR, "data", "sts_images"),
            image_file_type="nifti",
            image_sub_folder=os.path.join("CT", "dicom", "image"),
            mask_sub_folder=os.path.join("CT", "dicom", "mask")
        )
    assert "did not contain any supported image files" in str(exception_info.value)

    # DICOM stack and _masks for all samples, but with incorrect instructions.
    # No matching image modality.
    with pytest.raises(ValueError) as exception_info:
        image_list = import_image_and_mask(
            image=os.path.join(CURRENT_DIR, "data", "sts_images"),
            image_modality="pet",
            image_sub_folder=os.path.join("CT", "dicom", "image"),
            mask_sub_folder=os.path.join("CT", "dicom", "mask")
        )
    assert "No images were found" in str(exception_info.value)

    # DICOM stack and _masks for all samples, but with incorrect instructions.
    # No matching mask modality.
    with pytest.raises(ValueError) as exception_info:
        image_list = import_image_and_mask(
            image=os.path.join(CURRENT_DIR, "data", "sts_images"),
            mask_modality="seg",
            image_sub_folder=os.path.join("CT", "dicom", "image"),
            mask_sub_folder=os.path.join("CT", "dicom", "mask")
        )
    assert "No masks were found" in str(exception_info.value)

    # DICOM stack and _masks for all samples, but with incorrect instructions.
    # Wrong image_name.
    with pytest.raises(ValueError) as exception_info:
        image_list = import_image_and_mask(
            image=os.path.join(CURRENT_DIR, "data", "sts_images"),
            image_name="false_image",
            image_sub_folder=os.path.join("CT", "dicom", "image"),
            mask_sub_folder=os.path.join("CT", "dicom", "mask")
        )
    assert "not contain any supported image files" in str(exception_info.value)
    assert "that contain the name pattern (false_image)" in str(exception_info.value)

    # DICOM stack and _masks for all samples, but with incorrect instructions.
    # Wrong mask_name.
    with pytest.raises(ValueError) as exception_info:
        image_list = import_image_and_mask(
            image=os.path.join(CURRENT_DIR, "data", "sts_images"),
            mask_name="false_mask",
            image_sub_folder=os.path.join("CT", "dicom", "image"),
            mask_sub_folder=os.path.join("CT", "dicom", "mask")
        )
    assert "not contain any supported mask files" in str(exception_info.value)
    assert "that contain the name pattern (false_mask)" in str(exception_info.value)

    # Read Nifti image and _masks for all samples, but with incorrect instructions.
    # No matching sample name.
    with pytest.raises(ValueError) as exception_info:
        image_list = import_image_and_mask(
            image=os.path.join(CURRENT_DIR, "data", "sts_images"),
            image_sub_folder=os.path.join("CT", "nifti", "image"),
            sample_name="false_sample_name",
            mask_sub_folder=os.path.join("CT", "nifti", "mask")
        )
    assert "could not be linked to a sample name for checking" in str(exception_info.value)

    # Read Nifti image and _masks for all samples, but with incorrect instructions.
    # Wrong image_name.
    with pytest.raises(ValueError) as exception_info:
        image_list = import_image_and_mask(
            image=os.path.join(CURRENT_DIR, "data", "sts_images"),
            image_sub_folder=os.path.join("CT", "nifti", "image"),
            image_name="false_image",
            mask_sub_folder=os.path.join("CT", "nifti", "mask")
        )
    assert "not contain any supported image files" in str(exception_info.value)
    assert "that contain the name pattern (false_image)" in str(exception_info.value)

    # Read Nifti image and _masks for all samples, but with incorrect instructions.
    # Wrong mask_name.
    with pytest.raises(ValueError) as exception_info:
        image_list = import_image_and_mask(
            image=os.path.join(CURRENT_DIR, "data", "sts_images"),
            image_sub_folder=os.path.join("CT", "nifti", "image"),
            mask_name="false_mask",
            mask_sub_folder=os.path.join("CT", "nifti", "mask")
        )
    assert "not contain any supported mask files" in str(exception_info.value)
    assert "that contain the name pattern (false_mask)" in str(exception_info.value)


def test_failure_multiple_image_and_mask_import_data_xml():
    # Read the data settings xml file, and update path to image and mask.
    from xml.etree import ElementTree as ElemTree
    from mirp import get_data_xml

    target_dir = os.path.join(CURRENT_DIR, "data", "temp")
    target_file = os.path.join(target_dir, "data.xml")

    # Start with a clean slate.
    if os.path.exists(target_file):
        os.remove(target_file)

    get_data_xml(target_dir=target_dir)

    # Load xml.
    tree = ElemTree.parse(target_file)
    paths_branch = tree.getroot()

    # Set basic data in xml file.
    for element in paths_branch.iter("image"):
        element.text = str(os.path.join(CURRENT_DIR, "data", "sts_images"))
    for element in paths_branch.iter("image_sub_folder"):
        element.text = str(r"CT/dicom/image")
    for element in paths_branch.iter("mask"):
        element.text = str(os.path.join(CURRENT_DIR, "data", "sts_images"))
    for element in paths_branch.iter("mask_sub_folder"):
        element.text = str(r"CT/dicom/mask")

    # DICOM stack and _masks for all samples, but with incorrect instructions.
    # No matching file type.
    false_file_type_tree = copy.deepcopy(tree)
    for element in false_file_type_tree.getroot().iter("image_file_type"):
        element.text = "nifti"
    false_file_type_tree.write(target_file)

    with pytest.raises(ValueError) as exception_info:
        image_list = import_image_and_mask(
            image=target_file
        )
    assert "did not contain any supported image files" in str(exception_info.value)

    # DICOM stack and _masks for all samples, but with incorrect instructions.
    # No matching image modality.
    false_image_modality_tree = copy.deepcopy(tree)
    for element in false_image_modality_tree.getroot().iter("image_modality"):
        element.text = "pet"
    false_image_modality_tree.write(target_file)

    with pytest.raises(ValueError) as exception_info:
        image_list = import_image_and_mask(
            image=target_file
        )
    assert "No images were found" in str(exception_info.value)

    # DICOM stack and _masks for all samples, but with incorrect instructions.
    # No matching mask modality.
    false_mask_modality_tree = copy.deepcopy(tree)
    for element in false_mask_modality_tree.getroot().iter("mask_modality"):
        element.text = "seg"
    false_mask_modality_tree.write(target_file)

    with pytest.raises(ValueError) as exception_info:
        image_list = import_image_and_mask(
            image=target_file
        )
    assert "No masks were found" in str(exception_info.value)

    # DICOM stack and _masks for all samples, but with incorrect instructions.
    # Wrong image_name.
    false_image_name_tree = copy.deepcopy(tree)
    for element in false_image_name_tree.getroot().iter("image_name"):
        element.text = "false_image"
    false_image_name_tree.write(target_file)

    with pytest.raises(ValueError) as exception_info:
        image_list = import_image_and_mask(
            image=target_file
        )
    assert "not contain any supported image files" in str(exception_info.value)
    assert "that contain the name pattern (false_image)" in str(exception_info.value)

    # DICOM stack and _masks for all samples, but with incorrect instructions.
    # Wrong mask_name.
    false_mask_name_tree = copy.deepcopy(tree)
    for element in false_mask_name_tree.getroot().iter("mask_name"):
        element.text = "false_mask"
    false_mask_name_tree.write(target_file)
    with pytest.raises(ValueError) as exception_info:
        image_list = import_image_and_mask(
            image=target_file
        )
    assert "not contain any supported mask files" in str(exception_info.value)
    assert "that contain the name pattern (false_mask)" in str(exception_info.value)
