import os
import numpy as np

from mirp.extractMaskLabels import extract_mask_labels


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_extract_mask_labels_generic():
    # Single mask.
    mask_labels = extract_mask_labels(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy")
    )
    assert mask_labels.roi_label.values[0] == 1

    # Multiple masks.
    mask_labels = extract_mask_labels(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask_sub_folder=os.path.join("CT", "numpy", "mask")
    )

    assert all(x == 1 for x in mask_labels.roi_label.values)

    mask = np.zeros((10, 10), dtype=int)
    mask[1, 1] = 1
    mask[2, 2] = 2
    mask[3, 3] = 3
    mask_labels = extract_mask_labels(mask=mask)

    assert all(x in [1, 2, 3] for x in mask_labels.roi_label.values)


def test_extract_mask_labels_rtstruct():
    # Single mask.
    mask_labels = extract_mask_labels(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm")
    )
    assert mask_labels.roi_label.values[0] == "GTV_Mass_CT"

    # Multiple masks.
    mask_labels = extract_mask_labels(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask_sub_folder=os.path.join("CT", "dicom", "mask")
    )
    assert all(x == "GTV_Mass_CT" for x in mask_labels.roi_label.values)


def test_extract_mask_labels_seg():
    # Single mask.
    mask_labels = extract_mask_labels(
        mask=os.path.join(CURRENT_DIR, "data", "ct_images_seg", "CRLM-CT-1004", "mask", "mask.dcm")
    )
    assert mask_labels.roi_label.values

    # Multiple masks
    mask_labels = extract_mask_labels(
        mask=os.path.join(CURRENT_DIR, "data", "ct_images_seg"),
        mask_sub_folder="mask"
    )

    assert mask_labels.roi_label.values
