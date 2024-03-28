import os.path
import numpy as np

from mirp.data_import.import_image_and_mask import import_image_and_mask
from mirp._data_import.read_data import read_image_and_masks
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._images.ct_image import CTImage
from mirp._images.pet_image import PETImage
from mirp._images.mr_image import MRImage

CURRENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test")


def test_read_itk_image_and_mask():
    # Simple test.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz")
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)

    # With roi name specified.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"),
        roi_name="1"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)

    # With roi name not appearing.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"),
        roi_name="2"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 0

    # Multiple roi names of which one is present.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"),
        roi_name=["1", "2", "3"]
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)

    # Multiple roi names, with dictionary to set labels.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"),
        roi_name={"1": "gtv", "2": "some_roi", "3": "another_roi"}
    )

    image, roi_list = read_image_and_masks(image=image_list[0], to_numpy=False)
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "gtv"


def test_read_numpy_image_and_mask():
    # Simple test.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy")
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)

    # With roi name specified.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy"),
        roi_name="1"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)

    # With roi name not appearing.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy"),
        roi_name="2"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 0

    # Multiple roi names of which one is present.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy"),
        roi_name=["1", "2", "3"]
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)

    # Multiple roi names, with dictionary to set labels.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy"),
        roi_name={"1": "gtv", "2": "some_roi", "3": "another_roi"}
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "gtv"


def test_read_numpy_image_and_mask_stack():

    # Simple test.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "mask")
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)

    # With roi name specified.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "mask"),
        roi_name="1"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)

    # With roi name not appearing.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "mask"),
        roi_name="2"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 0

    # Multiple roi names of which one is present.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "mask"),
        roi_name=["1", "2", "3"]
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)

    # Multiple roi names, with dictionary to set labels.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "mask"),
        roi_name={"1": "gtv", "2": "some_roi", "3": "another_roi"}
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "gtv"


def test_read_numpy_image_and_mask_online():
    """
    Test reading numpy arrays that are provided directly as input.
    """

    # Simple test.
    image = np.load(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"))
    mask = np.load(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy"))

    image_list = import_image_and_mask(
        image=image,
        mask=mask
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)

    # With roi name specified.
    image = np.load(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"))
    mask = np.load(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy"))

    image_list = import_image_and_mask(
        image=image,
        mask=mask,
        roi_name="1"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)


def test_read_dicom_image_and_mask_stack():
    # Simple test.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm")
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "GTV_Mass_CT"

    # With roi name specified.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        roi_name="GTV_Mass_CT"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "GTV_Mass_CT"

    # With roi name not appearing.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        roi_name="some_roi"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 0

    # Multiple roi names of which one is present.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        roi_name=["GTV_Mass_CT", "some_roi", "another_roi"]
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "GTV_Mass_CT"

    # Multiple roi names, with dictionary to set labels.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        roi_name={"GTV_Mass_CT": "gtv", "2": "some_roi", "3": "another_roi"}
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, GenericImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "gtv"


def test_read_dicom_image_and_mask_modality_specific():
    # Read CT image.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm")
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, CTImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "GTV_Mass_CT"

    # Read PET image.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "PET", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "PET", "dicom", "mask", "RS.dcm")
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, PETImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "GTV_Mass_PET"

    # Read T1-weighted MR image.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "MR_T1", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "MR_T1", "dicom", "mask", "RS.dcm")
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, MRImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "GTV_Mass_MR_T1"


def test_read_generic_image_and_mask_modality_specific():
    # Read CT image.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"),
        image_modality="CT"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, CTImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "region_1"

    # Read PET image.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "PET", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "PET", "nifti", "mask", "mask.nii.gz"),
        image_modality="PET"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, PETImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "region_1"

    # Read T1-weighted MR image.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "MR_T1", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "MR_T1", "nifti", "mask", "mask.nii.gz"),
        image_modality="MR"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, MRImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "region_1"


def test_read_dicom_image_and_mask_data_xml():
    # Read the data settings xml file, and update path to image and mask.
    from xml.etree import ElementTree as ElemTree

    # Remove temporary data xml file if it exists.
    if os.path.exists(os.path.join(CURRENT_DIR, "data", "configuration_files", "temp_test_config_data.xml")):
        os.remove(os.path.join(CURRENT_DIR, "data", "configuration_files", "temp_test_config_data.xml"))

    # Load xml.
    tree = ElemTree.parse(os.path.join(CURRENT_DIR, "data", "configuration_files", "test_config_data.xml"))
    paths_branch = tree.getroot()

    # Update paths in xml file.
    for image in paths_branch.iter("image"):
        image.text = str(os.path.join(CURRENT_DIR, "data", "sts_images"))
    for mask in paths_branch.iter("mask"):
        mask.text = str(os.path.join(CURRENT_DIR, "data", "sts_images"))

    # Save as temporary xml file.
    tree.write(os.path.join(CURRENT_DIR, "data", "configuration_files", "temp_test_config_data.xml"))

    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "configuration_files", "temp_test_config_data.xml")
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, PETImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "GTV_Mass_PET"
