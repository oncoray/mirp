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


def test_base_mask_interpolate():
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

    # Update settings
    settings.img_interpolate.interpolate = True
    settings.img_interpolate.new_spacing = [1.0, 1.0, 1.0]

    # Test by registering to image.
    test_mask = mask.copy()
    test_mask.interpolate(image=images[0], settings=settings)

    # Nothing should change compared to the original roi, as the image is still the same.
    assert test_mask.roi.image_spacing == mask.roi.image_spacing
    assert np.array_equal(test_mask.roi.get_voxel_grid(), mask.roi.get_voxel_grid())
    assert test_mask.roi_intensity.image_spacing == mask.roi_intensity.image_spacing
    assert np.array_equal(test_mask.roi_intensity.get_voxel_grid(), mask.roi_intensity.get_voxel_grid())
    assert test_mask.roi_morphology.image_spacing == mask.roi_morphology.image_spacing
    assert np.array_equal(test_mask.roi_morphology.get_voxel_grid(), mask.roi_morphology.get_voxel_grid())

    # Test stand-alone interpolation.
    test_mask = mask.copy()
    test_mask.interpolate(image=None, settings=settings)

    # The test mask should be interpolated.
    assert test_mask.roi.image_spacing == (1.0, 1.0, 1.0)
    assert test_mask.roi_intensity.image_spacing == (1.0, 1.0, 1.0)
    assert test_mask.roi_morphology.image_spacing == (1.0, 1.0, 1.0)


def test_base_mask_get_slices():
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

    # All slices from every type of roi in the mask.
    mask_slices = mask.get_slices()

    assert len(mask_slices) == 4
    assert mask_slices[0].roi is not None
    assert mask_slices[0].roi_intensity is not None
    assert mask_slices[0].roi_morphology is not None
    assert mask_slices[0].roi.image_origin == mask.roi.image_origin
    assert mask_slices[1].roi.image_origin == (2.0, 0.0, 0.0)

    # All slices from the original roi in the mask.
    mask_slices = mask.get_slices(primary_mask_only=True)

    assert len(mask_slices) == 4
    assert mask_slices[0].roi is not None
    assert mask_slices[0].roi_intensity is None
    assert mask_slices[0].roi_morphology is None
    assert mask_slices[0].roi.image_origin == mask.roi.image_origin
    assert mask_slices[1].roi.image_origin == (2.0, 0.0, 0.0)

    # Select single slice from every type of roi in the mask.
    mask_slice: BaseMask = mask.get_slices(slice_number=1)

    assert mask_slice.roi is not None
    assert mask_slice.roi_intensity is not None
    assert mask_slice.roi_morphology is not None
    assert mask_slice.roi.image_origin == (2.0, 0.0, 0.0)

    # Select single slice from the original roi in the mask.
    mask_slice: BaseMask = mask.get_slices(slice_number=1, primary_mask_only=True)

    assert mask_slice.roi is not None
    assert mask_slice.roi_intensity is None
    assert mask_slice.roi_morphology is None
    assert mask_slice.roi.image_origin == (2.0, 0.0, 0.0)
