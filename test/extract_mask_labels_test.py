import os
import pytest
import numpy as np

from mirp.extract_mask_labels import extract_mask_labels


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.ci
def test_extract_mask_labels_generic():
    # Single mask.
    mask_labels = extract_mask_labels(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "mask", "STS_001_mask.npy")
    )
    assert mask_labels.roi_label.values[0] == 1

    # Multiple _masks.
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


@pytest.mark.ci
def test_extract_mask_labels_rtstruct():
    # Single mask.
    mask_labels = extract_mask_labels(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask", "RS.dcm")
    )
    assert mask_labels.roi_label.values[0] == "GTV_Mass_CT"

    # Multiple _masks.
    mask_labels = extract_mask_labels(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask_sub_folder=os.path.join("CT", "dicom", "mask")
    )

    assert all(x == "GTV_Mass_CT" for x in mask_labels.roi_label.values)

    # TODO: Mask with multiple labels
    ...


@pytest.mark.ci
def test_extract_mask_labels_seg():
    # Single mask.
    mask_labels = extract_mask_labels(
        mask=os.path.join(CURRENT_DIR, "data", "ct_images_seg", "CRLM-CT-1004", "mask", "mask.dcm")
    )
    assert mask_labels.roi_label.values[0] == "Liver"

    # Multiple masks.
    mask_labels = extract_mask_labels(
        mask=os.path.join(CURRENT_DIR, "data", "ct_images_seg"),
        mask_sub_folder=os.path.join("mask")
    )
    assert len([label for label in mask_labels.roi_label.values if label == "Liver"]) == 2


def test_extract_mask_labels_rtstruct_to_file(tmp_path):
    import pandas as pd

    extract_mask_labels(
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask_sub_folder=os.path.join("CT", "dicom", "mask"),
        write_dir=tmp_path
    )

    mask_labels = pd.read_csv(os.path.join(tmp_path, "mask_labels.csv"))
    assert len(mask_labels) == 3
