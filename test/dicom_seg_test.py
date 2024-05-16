import os
import numpy as np

from mirp.extract_features_and_images import extract_images

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_dicom_segmentation_mask():
    from mirp._images.ct_image import CTImage, GenericImage

    image, seg_mask = extract_images(
        image=os.path.join(CURRENT_DIR, "data", "ct_images_seg_nsclc", "LUNG1-001", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ct_images_seg_nsclc", "LUNG1-001", "mask_seg"),
        roi_name="Neoplasm, Primary",
        image_export_format="native"
    )[0]
    image: CTImage = image[0]
    assert isinstance(image, CTImage)

    image, rtstruct_mask = extract_images(
        image=os.path.join(CURRENT_DIR, "data", "ct_images_seg_nsclc", "LUNG1-001", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ct_images_seg_nsclc", "LUNG1-001", "mask_rtstruct"),
        roi_name="GTV-1",
        image_export_format="native"
    )[0]
    image: CTImage = image[0]
    assert isinstance(image, CTImage)

    assert np.array_equal(seg_mask[0].roi.get_voxel_grid(), rtstruct_mask[0].roi.get_voxel_grid())
