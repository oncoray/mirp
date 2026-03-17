import os.path
import numpy as np
import warnings

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
    """
    Report 2026.02.09:
    Can process:
        all, but DRO_2_4

    Incorrectly processed:
        DRO_2_0
        DRO_2_1
        DRO_2_2
        DRO_2_3
        DRO_3_0
        DRO_3_1
        DRO_3_2
        DRO_3_3
        DRO_4_2
    """

    available_dro = [
        'DRO_0_0',
        'DRO_1_0',
        'DRO_2_0', 'DRO_2_1', 'DRO_2_2', 'DRO_2_3', 'DRO_2_4', 'DRO_2_5',
        'DRO_3_0', 'DRO_3_1', 'DRO_3_2', 'DRO_3_3', 'DRO_3_4',
        'DRO_4_0', 'DRO_4_1', 'DRO_4_2',
        'DRO_5_0'
    ]

    for dro in available_dro:
        custom_kwargs = dict([])

        # Cannot be processsed
        if dro == "DRO_2_4":
            warnings.warn(f"{dro} is missing information necessary for conversion.", UserWarning)
            continue

        if dro == "DRO_2_1":
            custom_kwargs.update({"pet_suv_conversion": "lean_body_mass"})
        elif dro == "DRO_2_2":
            custom_kwargs.update({"pet_suv_conversion": "ideal_body_weight"})
        elif dro == "DRO_2_3":
            custom_kwargs.update({"pet_suv_conversion": "body_surface_area"})

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

        # Does not compute correctly:
        if dro in ["DRO_2_0", "DRO_2_1", "DRO_2_2", "DRO_2_3", "DRO_3_0", "DRO_3_1", "DRO_3_2", "DRO_3_3",
                   "DRO_4_2"]:
            warnings.warn(f"{dro} is not converted correctly.", UserWarning)
            continue

        assert isinstance(image, PETImage)
        assert isinstance(mask, BaseMask)

        assert np.around(np.median(image.get_voxel_grid()[mask.roi.get_voxel_grid()]), 3) == 1.0
        assert np.around(np.max(image.get_voxel_grid()[mask.roi.get_voxel_grid()]), 3) == 4.0
        assert np.around(np.min(image.get_voxel_grid()[mask.roi.get_voxel_grid()]), 3) == 0.2

        pass
