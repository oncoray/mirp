import os.path

from mirp.importData.importImageAndMask import import_image_and_mask
from mirp.importData.readData import read_image_and_masks
from mirp.imageClass import ImageClass
from mirp.roiClass import RoiClass


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_read_itk_image_and_mask():
    # Simple test.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz")
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)

    # With roi name specified.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"),
        roi_name="1"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)

    # With roi name not appearing.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"),
        roi_name="2"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 0

    # Multiple roi names of which one is present.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"),
        roi_name=["1", "2", "3"]
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)

    # Multiple roi names, with dictionary to set labels.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"),
        roi_name={"1": "gtv", "2": "some_roi", "3": "another_roi"}
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)
    assert roi_list[0].name == "gtv"


def test_read_numpy_image_and_mask():
    # Simple test.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy")
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)

    # With roi name specified.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy"),
        roi_name="1"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)

    # With roi name not appearing.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy"),
        roi_name="2"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 0

    # Multiple roi names of which one is present.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy"),
        roi_name=["1", "2", "3"]
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)

    # Multiple roi names, with dictionary to set labels.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy"),
        roi_name={"1": "gtv", "2": "some_roi", "3": "another_roi"}
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)
    assert roi_list[0].name == "gtv"


def test_read_numpy_image_and_mask_stack():

    # Simple test.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "mask")
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)

    # With roi name specified.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "mask"),
        roi_name="1"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)

    # With roi name not appearing.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "mask"),
        roi_name="2"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 0

    # Multiple roi names of which one is present.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "mask"),
        roi_name=["1", "2", "3"]
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)

    # Multiple roi names, with dictionary to set labels.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "mask"),
        roi_name={"1": "gtv", "2": "some_roi", "3": "another_roi"}
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)
    assert roi_list[0].name == "gtv"


def test_read_dicom_image_and_mask_stack():
    # Simple test.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm")
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)
    assert roi_list[0].name == "GTV_Mass_CT"

    # With roi name specified.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        roi_name="GTV_Mass_CT"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)
    assert roi_list[0].name == "GTV_Mass_CT"

    # With roi name not appearing.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        roi_name="some_roi"
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 0

    # Multiple roi names of which one is present.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        roi_name=["GTV_Mass_CT", "some_roi", "another_roi"]
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)
    assert roi_list[0].name == "GTV_Mass_CT"

    # Multiple roi names, with dictionary to set labels.
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm"),
        roi_name={"GTV_Mass_CT": "gtv", "2": "some_roi", "3": "another_roi"}
    )

    image, roi_list = read_image_and_masks(image=image_list[0])
    assert isinstance(image, ImageClass)
    assert len(roi_list) == 1
    assert all(isinstance(roi, RoiClass) for roi in roi_list)
    assert roi_list[0].name == "gtv"
