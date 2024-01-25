import os.path
import numpy as np

from mirp.importData.importImageAndMask import import_image_and_mask
from mirp.importData.readData import read_image_and_masks
from mirp.images.genericImage import GenericImage
from mirp.masks.baseMask import BaseMask
from mirp.images.ctImage import CTImage
from mirp.images.petImage import PETImage
from mirp.images.mrImage import MRImage


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def test_read_dicom_image_and_rtstruct_mask_stack():
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


def test_read_dicom_image_and_seg_mask_stack():
    # Simple test.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "ct_images_seg", "CRLM-CT-1004", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ct_images_seg", "CRLM-CT-1004", "mask", "mask.dcm")
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, CTImage)
    assert len(roi_list) == 6
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "Liver"

    # With roi name specified.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "ct_images_seg", "CRLM-CT-1004", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ct_images_seg", "CRLM-CT-1004", "mask", "mask.dcm"),
        roi_name="Liver"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, CTImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "Liver"

    # With roi name not appearing.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "ct_images_seg", "CRLM-CT-1004", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ct_images_seg", "CRLM-CT-1004", "mask", "mask.dcm"),
        roi_name="some_roi"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, CTImage)
    assert len(roi_list) == 0

    # Multiple roi names of which one is present.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "ct_images_seg", "CRLM-CT-1004", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ct_images_seg", "CRLM-CT-1004", "mask", "mask.dcm"),
        roi_name=["Liver", "some_roi", "another_roi"]
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, CTImage)
    assert len(roi_list) == 1
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "Liver"

    # Multiple roi names, with dictionary to set labels.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "ct_images_seg", "CRLM-CT-1004", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ct_images_seg", "CRLM-CT-1004", "mask", "mask.dcm"),
        roi_name={"Tumor_1": "lesion_1", "Tumor_2": "lesion_2", "3": "another_roi"}
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, CTImage)
    assert len(roi_list) == 2
    assert all(isinstance(roi, BaseMask) for roi in roi_list)
    assert roi_list[0].roi_name == "lesion_1"
    assert roi_list[1].roi_name == "lesion_2"


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
