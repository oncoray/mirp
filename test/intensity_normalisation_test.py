import os.path
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

GENERIC_KWARGS = dict([
    ("write_features", False),
    ("export_features", True),
    ("write_images", False),
    ("export_images", True),
    ("image_export_format", "native"),
    ("image", os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1", "image")),
    ("mask", os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1", "mask")),
    ("roi_name", "GTV_Mass"),
    ("base_feature_families","statistics")
])


def test_intensity_normalisation():
    from mirp.settings.image_processing_parameters import ImagePostProcessingClass
    from mirp import extract_features_and_images
    from mirp._masks.base_mask import BaseMask
    from mirp._images.generic_image import GenericImage
    from mirp._images.mr_image import MRImage

    for intensity_normalisation_method in ImagePostProcessingClass()._get_available_intensity_normalisation_methods():
        # Skip over select intensity normalisation methods that require a custom test.
        if intensity_normalisation_method == "custom_scale":
            continue

        # No tissue mask.
        data_no_mask = extract_features_and_images(
            tissue_mask_type="none",
            intensity_normalisation=intensity_normalisation_method,
            **GENERIC_KWARGS
        )

        # Tissue mask
        data_mask = extract_features_and_images(
            tissue_mask_type="relative_range",
            tissue_mask_range=[0.02, 1.00],
            intensity_normalisation=intensity_normalisation_method,
            **GENERIC_KWARGS
        )

        feature_data_nm = data_no_mask[0][0]
        image_nm = data_no_mask[0][1][0]
        mask_nm = data_no_mask[0][2][0]
        feature_data_m = data_mask[0][0]
        image_m = data_mask[0][1][0]
        mask_m = data_mask[0][2][0]

        assert len(feature_data_nm) == 1
        assert len(feature_data_m) == 1
        assert isinstance(mask_nm, BaseMask)
        assert isinstance(mask_m, BaseMask)
        assert isinstance(image_nm, MRImage)
        assert isinstance(image_m, MRImage)

        if intensity_normalisation_method == "none":
            assert np.min(image_nm.get_voxel_grid()) == 0.0
            assert np.max(image_nm.get_voxel_grid()) == 1807.0
            assert np.min(image_m.get_voxel_grid()) == 0.0
            assert np.max(image_m.get_voxel_grid()) == 1807.0
            assert 239.0 < np.std(image_nm.get_voxel_grid()) < 240.0
            assert 239.0 < np.std(image_nm.get_voxel_grid()) < 240.0

        elif intensity_normalisation_method == "range":
            assert np.min(image_nm.get_voxel_grid()) == 0.0
            assert np.max(image_nm.get_voxel_grid()) == 1.0
            assert -0.05 < np.min(image_m.get_voxel_grid()) < 0.0
            assert np.max(image_m.get_voxel_grid()) == 1.0
            assert 0.13 < np.std(image_nm.get_voxel_grid()) < 0.14
            assert 0.13 < np.std(image_m.get_voxel_grid()) < 0.14

        elif intensity_normalisation_method == "relative_range":
            assert np.min(image_nm.get_voxel_grid()) == 0.0
            assert np.max(image_nm.get_voxel_grid()) == 1.0
            assert -0.05 < np.min(image_m.get_voxel_grid()) < 0.0
            assert np.max(image_m.get_voxel_grid()) == 1.0
            assert 0.13 < np.std(image_nm.get_voxel_grid()) < 0.14
            assert 0.13 < np.std(image_m.get_voxel_grid()) < 0.14

        elif intensity_normalisation_method == "quantile_range":
            assert np.min(image_nm.get_voxel_grid()) < 0.0
            assert np.max(image_nm.get_voxel_grid()) > 2.0
            assert np.min(image_m.get_voxel_grid()) < 0.0
            assert np.max(image_m.get_voxel_grid()) > 2.0
            assert 0.29 < np.std(image_nm.get_voxel_grid()) < 0.30
            assert 0.29 < np.std(image_m.get_voxel_grid()) < 0.30

        elif intensity_normalisation_method == "standardisation":
            assert -1.1 < np.min(image_nm.get_voxel_grid()) < -1.0
            assert 6.4 < np.max(image_nm.get_voxel_grid()) < 6.5
            assert -1.4 < np.min(image_m.get_voxel_grid()) < -1.3
            assert 6.3 < np.max(image_m.get_voxel_grid()) < 6.4
            assert 0.99 < np.std(image_nm.get_voxel_grid()) < 1.00
            assert 1.00 < np.std(image_m.get_voxel_grid()) < 1.01

        elif intensity_normalisation_method == "histogram_equalisation":
            assert np.min(image_nm.get_voxel_grid()) == 0.0
            assert np.max(image_nm.get_voxel_grid()) == 1.0
            assert np.min(image_m.get_voxel_grid()) == 0.0
            assert np.max(image_m.get_voxel_grid()) == 1.0
            assert 0.27 < np.std(image_nm.get_voxel_grid()) < 0.28
            assert 0.32 < np.std(image_m.get_voxel_grid()) < 0.33

        elif intensity_normalisation_method == "adaptive_equalisation":
            assert np.min(image_nm.get_voxel_grid()) == 0.0
            assert np.max(image_nm.get_voxel_grid()) == 1.0
            assert np.min(image_m.get_voxel_grid()) == 0.0
            assert np.max(image_m.get_voxel_grid()) == 1.0
            assert 0.27 < np.std(image_nm.get_voxel_grid()) < 0.28
            assert 0.32 < np.std(image_m.get_voxel_grid()) < 0.33

        else:
            raise NotImplementedError(f"Test for intensity_normalisation_method {intensity_normalisation_method} not implemented")


def test_custom_scale():
    ...