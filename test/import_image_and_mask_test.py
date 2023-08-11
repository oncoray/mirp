import os.path

import pytest

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
from mirp.importData.importImageAndMask import import_image_and_mask
from mirp.importData.imageITKFile import ImageITKFile, MaskITKFile
from mirp.importData.imageDicomFile import ImageDicomFile
from mirp.importData.imageDicomFileStack import ImageDicomFileStack
from mirp.importData.imageDicomFileRTSTRUCT import MaskDicomFileRTSTRUCT
from mirp.importData.imageNumpyFile import ImageNumpyFile, MaskNumpyFile
from mirp.importData.imageNumpyFileStack import ImageNumpyFileStack, MaskNumpyFileStack


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
    # Read Nifti images and masks directly.
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

    # Read DICOM images and masks.
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

    # Read numpy images and masks directly.
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

    # Read Nifti images and masks for specific samples.
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

    # Read DICOM images and masks for specific samples.
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

    # Read numpy images and masks for specific samples.
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

    # Read Nifti masks for all samples.
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
    assert mask_list[0].sample_name == "STS_001"
    assert mask_list[1].sample_name == "STS_002"
    assert mask_list[2].sample_name == "STS_003"

    # Read numpy masks for all samples.
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


def test_single_image_and_mask_import_flat():
    # Read a Nifti mask directly.
    mask_list = import_mask(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
        sample_name="STS_001",
        mask_name="#_CT_mask")
    assert len(mask_list) == 1
    assert isinstance(mask_list[0], MaskITKFile)
    assert mask_list[0].modality == "generic_mask"
    assert mask_list[0].sample_name == "STS_001"

    # Read a DICOM RTSTRUCT masks.
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


def test_multiple_image_and_mask_import_flat():

    # Read Nifti masks for specific samples.
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

    # Read numpy masks for specific samples.
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

    # Read Nifti masks for all samples.
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

    # Read numpy masks for all samples.
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

    # Read Nifti masks for all samples without specifying the sample name in the image name.
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

    # Read numpy masks for all samples without specifying the sample name in the image name.
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
