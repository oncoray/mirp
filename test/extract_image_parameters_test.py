import os

import pandas as pd
import pytest
from mirp.extract_image_parameters import extract_image_parameters


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.ci
def test_extract_image_parameters_default():
    # Read single image.
    image_parameters = extract_image_parameters(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz")
    )
    assert all(x in image_parameters.columns for x in ["modality", "spacing_z", "spacing_y", "spacing_x"])
    assert len(image_parameters) == 1

    # Read multiple images.
    image_parameters = extract_image_parameters(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "nifti", "image")
    )
    assert all(x in image_parameters.columns for x in ["modality", "spacing_z", "spacing_y", "spacing_x"])
    assert len(image_parameters) == 3


@pytest.mark.ci
def test_extract_image_parameters_dicom():
    # Read a single CT image.
    image_parameters = extract_image_parameters(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image")
    )
    assert len(image_parameters) == 1

    # Read multiple CT images.
    image_parameters = extract_image_parameters(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "dicom", "image")
    )
    assert len(image_parameters) == 3

    # Read a single PET image.
    image_parameters = extract_image_parameters(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "PET", "dicom", "image")
    )
    assert len(image_parameters) == 1

    # Read multiple PET images.
    image_parameters = extract_image_parameters(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("PET", "dicom", "image")
    )
    assert len(image_parameters) == 3

    # Read a single T1-weighted MR image.
    image_parameters = extract_image_parameters(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "MR_T1", "dicom", "image")
    )
    assert len(image_parameters) == 1

    # Read multiple T1-weighted MR images.
    image_parameters = extract_image_parameters(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("MR_T1", "dicom", "image")
    )
    assert len(image_parameters) == 3

    # Read a single ADC image.
    image_parameters = extract_image_parameters(
        image=os.path.join(CURRENT_DIR, "data", "adc_images_mr", "SCAN_001", "adc_image"),
    )
    assert len(image_parameters) == 1

    # Read multiple DICOM images.
    image_parameters = extract_image_parameters(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_file_type="dicom"
    )
    assert len(image_parameters) == 9

    # Read single RTDOSE image.
    image_parameters = extract_image_parameters(
        image=os.path.join(CURRENT_DIR, "data", "rtdose_images", "Pancreas-CT-CB_001", "rtdose")
    )
    assert len(image_parameters) == 1


def test_extract_image_parameters_dicom_to_file(tmp_path):

    extract_image_parameters(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "dicom", "image"),
        write_dir=tmp_path
    )

    image_parameters = pd.read_csv(os.path.join(tmp_path, "image_metadata.csv"))
    assert len(image_parameters) == 3
