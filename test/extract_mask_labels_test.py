import os
from mirp.extractMaskLabels import extract_mask_labels


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


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
