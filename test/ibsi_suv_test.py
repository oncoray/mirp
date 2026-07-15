import os.path
import numpy as np
import pytest

from mirp import extract_images
from mirp._images.pet_image import PETImage
from mirp._masks.base_mask import BaseMask
from mirp.data_import.import_image_and_mask import import_image_and_mask
from mirp._data_import.dicom_file_rtstruct import MaskDicomFileRTSTRUCT

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_import_suv_dro():
    # Iterate over DRO in the  ibsi_suv/DRO directory to test if the DRO can be read with their associated masks.
    for dro in os.scandir(os.path.join(CURRENT_DIR, "data", "ibsi_suv", "DRO")):
        if not dro.is_dir():
            continue

        # Import without reading / processing.
        image = import_image_and_mask(
            image=os.path.join(CURRENT_DIR, dro.path, "PT"),
            mask=os.path.join(CURRENT_DIR, dro.path, "RS"),
        )[0]

        assert image.modality == "pt"
        assert len(image.associated_masks) == 1
        assert isinstance(image.associated_masks[0], MaskDicomFileRTSTRUCT)


def test_read_suv_dro():

    available_dro = [
        "DRO_0_0",
        "DRO_1_0",
        "DRO_2_0",
        "DRO_2_1_0",
        "DRO_2_1_1",
        "DRO_2_1_2",
        "DRO_2_2_0",
        "DRO_2_2_1",
        "DRO_2_2_2",
        "DRO_2_3",
        "DRO_2_4",
        "DRO_2_5",
        "DRO_2_6_0",
        "DRO_2_6_1",
        "DRO_2_6_2",
        "DRO_3_0",
        "DRO_3_1",
        "DRO_3_2_0",
        "DRO_3_2_1",
        "DRO_3_2_2",
        "DRO_3_2_3",
        "DRO_3_3_0",
        "DRO_3_3_1",
        "DRO_3_4_0",
        "DRO_3_4_1",
        "DRO_3_4_2",
        "DRO_3_4_3",
        "DRO_3_5_0",
        "DRO_3_5_1",
        "DRO_3_5_2",
        "DRO_3_5_3",
        "DRO_4_0",
        "DRO_4_1",
        "DRO_4_2",
        "DRO_4_3",
        "DRO_4_4",
        "DRO_4_5",
        "DRO_5_0",
        "DRO_7_0_0",
        "DRO_7_2_0",
        "DRO_7_3_0",
        "DRO_7_3_1"
    ]

    for dro in available_dro:
        custom_kwargs = dict([])

        image, mask = extract_images(
            image_export_format="native",
            image=os.path.join(CURRENT_DIR, "data", "ibsi_suv", "DRO", dro, "PT"),
            mask=os.path.join(CURRENT_DIR, "data", "ibsi_suv", "DRO", dro, "RS"),
            roi_name="DRO_mask",
            **custom_kwargs
        )[0]
        image = image[0]
        mask = mask[0]

        print(f"{dro}: {np.around(np.min(image.get_voxel_grid()[mask.roi.get_voxel_grid()]), 5):.5f},"
              f"{np.around(np.median(image.get_voxel_grid()[mask.roi.get_voxel_grid()]), 5):.5f},"
              f"{np.around(np.max(image.get_voxel_grid()[mask.roi.get_voxel_grid()]), 5):.5f}")

        assert isinstance(image, PETImage)
        assert isinstance(mask, BaseMask)

        assert 0.99 < np.median(image.get_voxel_grid()[mask.roi.get_voxel_grid()]) < 1.01
        assert 3.99 < np.max(image.get_voxel_grid()[mask.roi.get_voxel_grid()]) < 4.01
        assert 0.19 < np.min(image.get_voxel_grid()[mask.roi.get_voxel_grid()]) < 0.21


def test_error_suv_dro():

    available_dro = [
        "DRO_error_2_0",
        "DRO_error_2_1",
        "DRO_error_2_2",
        "DRO_error_2_3",
        "DRO_error_2_4",
        "DRO_error_2_5",
        "DRO_error_2_6",
        "DRO_error_2_7",
        "DRO_error_3_0",
        "DRO_error_3_1",
        "DRO_error_3_2",
        "DRO_error_4_0",
        "DRO_error_4_1",
        "DRO_error_4_2",
        "DRO_error_5_0"
    ]

    for dro in available_dro:
        custom_kwargs = dict([])

        error_type = ValueError
        if dro == "DRO_error_2_7":
            error_type = NotImplementedError

        with pytest.raises(error_type) as error_info:
            image, mask = extract_images(
                image_export_format="native",
                image=os.path.join(CURRENT_DIR, "data", "ibsi_suv", "DRO", dro, "PT"),
                mask=os.path.join(CURRENT_DIR, "data", "ibsi_suv", "DRO", dro, "RS"),
                roi_name="DRO_mask",
                **custom_kwargs
            )

        print(f"{dro}: {str(error_info.value)}")
        # Check for ValueError that does not actually produce a meaningful error.
        assert "not enough values to unpack" not in str(error_info.value)
