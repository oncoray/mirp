import os.path
import pytest

from mirp.importData.importImageAndMask import import_image_and_mask
from mirp.importData.readData import read_image_and_masks
from mirp.imageClass import ImageClass
from mirp.roiClass import RoiClass


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_read_itk_image_and_mask():
    image_list = import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz")
    )

    image, roi_list = read_image_and_masks(image=image_list[0])

    assert isinstance(image, ImageClass)
    assert all(isinstance(roi, RoiClass) for roi in roi_list)
