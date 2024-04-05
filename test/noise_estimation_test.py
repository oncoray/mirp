import os
from mirp import extract_images
from mirp._images.ct_image import CTImage

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_rank_noise_estimation_method():
    data = extract_images(
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1"
    )

    image: CTImage = data[0][0][0]
    noise_estimate = image.estimate_noise(method="rank")

    assert noise_estimate < 3.0


def test_ikeda_noise_estimation_method():
    data = extract_images(
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1"
    )

    image: CTImage = data[0][0][0]
    noise_estimate = image.estimate_noise(method="ikeda")

    assert noise_estimate < 15.0


def test_chang_noise_estimation_method():
    data = extract_images(
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1"
    )

    image: CTImage = data[0][0][0]
    noise_estimate = image.estimate_noise(method="chang")

    assert noise_estimate < 15.0


def test_immerkaer_noise_estimation_method():
    data = extract_images(
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1"
    )

    image: CTImage = data[0][0][0]
    noise_estimate = image.estimate_noise(method="immerkaer")

    assert noise_estimate < 3.0


def test_zwanenburg_noise_estimation_method():
    data = extract_images(
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1"
    )

    image: CTImage = data[0][0][0]
    noise_estimate = image.estimate_noise(method="zwanenburg")

    assert noise_estimate < 15.0
