import os
import numpy as np

from mirp import extract_images
from mirp._images.ct_image import CTImage

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_median_denoiser_method():
    # 3D variant
    data = extract_images(
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        image_denoise_method="median",
        image_denoiser_median_size=5
    )

    image_3d: CTImage = data[0][0][0]

    # Default value is 12.
    noise_estimate_3d = image_3d.estimate_noise()

    assert noise_estimate_3d < 12.0
    assert np.min(image_3d.get_voxel_grid()) == -1000.0
    assert np.max(image_3d.get_voxel_grid()) == 714.0

    # 2D variant
    data = extract_images(
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        by_slice = True,
        image_denoise_method="median",
        image_denoiser_median_size=5
    )

    image_2d: CTImage = data[0][0][0]

    # Default value is 12.
    noise_estimate_2d = image_2d.estimate_noise()

    assert noise_estimate_2d < 12.0
    assert np.min(image_2d.get_voxel_grid()) == -1000.0
    assert np.max(image_2d.get_voxel_grid()) == 907.0

    # 3D operation takes more voxels into account and should suppress noise more strongly.
    assert noise_estimate_3d < noise_estimate_2d


def test_gaussian_denoiser_method():
    # 3D variant
    data = extract_images(
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        image_denoise_method="gaussian",
        image_denoiser_gaussian_sigma=1.5
    )

    image_3d: CTImage = data[0][0][0]

    # Default value is 12.
    noise_estimate_3d = image_3d.estimate_noise()

    assert noise_estimate_3d < 12.0
    assert np.min(image_3d.get_voxel_grid()) == -1000.0
    assert np.max(image_3d.get_voxel_grid()) == 805.0

    # 2D variant
    data = extract_images(
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        by_slice = True,
        image_denoise_method="gaussian",
        image_denoiser_gaussian_sigma=1.5
    )

    image_2d: CTImage = data[0][0][0]

    # Default value is 12.
    noise_estimate_2d = image_2d.estimate_noise()

    assert noise_estimate_2d < 12.0
    assert np.min(image_2d.get_voxel_grid()) == -1000.0
    assert np.max(image_2d.get_voxel_grid()) == 840.0

    # 3D operation takes more voxels into account and should suppress noise more strongly.
    assert noise_estimate_3d < noise_estimate_2d

