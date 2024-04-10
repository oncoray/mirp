import os

import numpy as np

from mirp import extract_images
from mirp.settings.import_config_parameters import import_configuration_settings
from mirp._masks.base_mask import BaseMask

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_base_mask_compute_diagnostic_features():
    """Tests the compute_diagnostic_features method of BaseMask"""
    settings = import_configuration_settings(
        compute_features=False
    )[0]

    images, masks = extract_images(
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_digital_phantom", "nifti", "image", "phantom.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_digital_phantom", "nifti", "mask", "mask.nii.gz"),
        image_export_format="native",
        settings=settings
    )[0]

    data = masks[0].compute_diagnostic_features(
        image=images[0],
        settings=settings
    )

    assert data["diag_int_map_dim_x"].values == 5
    assert data["diag_int_map_dim_y"].values == 4
    assert data["diag_int_map_dim_z"].values == 4


def test_base_mask_compute_export_all(tmp_path):
    """Test the export and write methods of BaseMask"""
    settings = import_configuration_settings(
        compute_features=False
    )[0]

    images, masks = extract_images(
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_digital_phantom", "nifti", "image", "phantom.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_digital_phantom", "nifti", "mask", "mask.nii.gz"),
        image_export_format="native",
        settings=settings
    )[0]

    mask: BaseMask = masks[0]

    # test export method (dict)
    exported_mask = mask.export(export_format="dict")
    assert isinstance(exported_mask, dict)
    assert "mask" in exported_mask
    exported_mask = mask.export(export_format="dict", write_all=True)
    assert isinstance(exported_mask, dict)
    assert "intensity_mask" in exported_mask
    assert "morphology_mask" in exported_mask

    # test export method (numpy)
    exported_mask = mask.export(export_format="numpy")
    assert isinstance(exported_mask, np.ndarray)
    exported_mask = mask.export(export_format="numpy", write_all=True)
    assert isinstance(exported_mask, list)
    assert isinstance(exported_mask[0], np.ndarray)
    assert isinstance(exported_mask[1], np.ndarray)

    # test write method (nifti)
    mask.write(dir_path=tmp_path, write_all=True, file_format="nifti")
    files = os.listdir(tmp_path)
    files = [file for file in files if file.endswith("nii.gz")]
    assert len(files) == 2
    assert any(file.endswith("_int.nii.gz") for file in files)
    assert any(file.endswith("_morph.nii.gz") for file in files)

    # test write method (numpy)
    mask.write(dir_path=tmp_path, write_all=True, file_format="numpy")
    files = os.listdir(tmp_path)
    files = [file for file in files if file.endswith("npy")]
    assert len(files) == 2
    assert any(file.endswith("_int.npy") for file in files)
    assert any(file.endswith("_morph.npy") for file in files)
