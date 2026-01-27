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
    from mirp._images.mr_image import MRImage

    for intensity_normalisation_method in ImagePostProcessingClass()._get_available_intensity_normalisation_methods():
        # Skip over select intensity normalisation methods that require a custom test.
        if intensity_normalisation_method in ["custom_scale", "match_reference", "match_reference_normalised"]:
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
            assert 0.19 < np.std(image_nm.get_voxel_grid()) < 0.20
            assert 0.19 < np.std(image_m.get_voxel_grid()) < 0.20

        elif intensity_normalisation_method == "match_uniform":
            assert 0.00 < np.min(image_nm.get_voxel_grid()) < 0.01
            assert 0.99 < np.max(image_nm.get_voxel_grid()) < 1.00
            assert np.min(image_m.get_voxel_grid()) == 0.0
            assert 0.99 < np.max(image_m.get_voxel_grid()) < 1.00
            assert 0.28 < np.std(image_nm.get_voxel_grid()) < 0.29
            assert 0.32 < np.std(image_m.get_voxel_grid()) < 0.33

        elif intensity_normalisation_method == "match_sigmoid":
            assert -3.10 < np.min(image_nm.get_voxel_grid()) < -3.09
            assert 5.33 < np.max(image_nm.get_voxel_grid()) < 5.34
            assert -2.84 < np.min(image_m.get_voxel_grid()) < -2.83
            assert 5.29 < np.max(image_m.get_voxel_grid()) < 5.30
            assert 0.99 < np.std(image_nm.get_voxel_grid()) < 1.00
            assert 1.39 < np.std(image_m.get_voxel_grid()) < 1.40

        else:
            raise NotImplementedError(f"Test for intensity_normalisation_method {intensity_normalisation_method} not implemented")


def test_custom_scale():
    from mirp import extract_features_and_images
    from mirp._images.mr_image import MRImage
    from mirp._masks.base_mask import BaseMask

    # No tissue mask.
    data_no_mask = extract_features_and_images(
        tissue_mask_type="none",
        intensity_normalisation="custom_scale",
        intensity_normalisation_standardisation_shift=500.0,
        intensity_normalisation_standardisation_scale=100.0,
        **GENERIC_KWARGS
    )

    # Tissue mask
    data_mask = extract_features_and_images(
        tissue_mask_type="relative_range",
        tissue_mask_range=[0.02, 1.00],
        intensity_normalisation="custom_scale",
        intensity_normalisation_standardisation_shift=500.0,
        intensity_normalisation_standardisation_scale=100.0,
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

    # Mask should not have an effect.
    assert np.min(image_nm.get_voxel_grid()) == -5.0
    assert 13.0 < np.max(image_nm.get_voxel_grid()) < 13.1
    assert np.min(image_m.get_voxel_grid()) == -5.0
    assert 13.0 < np.max(image_m.get_voxel_grid()) < 13.1
    assert 2.39 < np.std(image_nm.get_voxel_grid()) < 2.40
    assert 2.39 < np.std(image_m.get_voxel_grid()) < 2.40


def test_match_reference():
    from mirp import extract_features_and_images, extract_images
    from mirp._images.mr_image import MRImage
    from mirp._images.pet_image import PETImage
    from mirp._masks.base_mask import BaseMask

    # Reference image (FDG-PET)
    data_ref = extract_images(
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "FDG_PET", "mask"),
        roi_name="GTV_Mass"
    )

    image_ref = data_ref[0][0][0]
    assert isinstance(image_ref, PETImage)
    image_ref = image_ref.get_voxel_grid()

    # No tissue mask.
    data_no_mask = extract_features_and_images(
        tissue_mask_type="none",
        intensity_normalisation="match_reference",
        intensity_normalisation_reference=image_ref,
        **GENERIC_KWARGS
    )

    # Tissue mask
    data_mask = extract_features_and_images(
        tissue_mask_type="relative_range",
        tissue_mask_range=[0.02, 1.00],
        intensity_normalisation="match_reference",
        intensity_normalisation_reference=image_ref,
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

    assert np.min(image_nm.get_voxel_grid()) == np.min(image_ref)
    assert np.max(image_ref) - 0.01 < np.max(image_nm.get_voxel_grid()) <= np.max(image_ref)
    assert np.min(image_m.get_voxel_grid()) == np.min(image_ref)
    assert np.max(image_ref) - 0.01 < np.max(image_m.get_voxel_grid()) <= np.max(image_ref)
    assert 0.657 < np.std(image_nm.get_voxel_grid()) < 0.658
    assert 0.589 < np.std(image_m.get_voxel_grid()) < 0.590

    # The no mask image should have comparable variance and quantiles.
    q_ref = np.quantile(image_ref, [0.1, 0.25, 0.75, 0.9])
    q_match = np.quantile(image_nm.get_voxel_grid(), [0.1, 0.25, 0.75, 0.9])

    assert np.std(image_ref) - 0.01 < np.std(image_nm.get_voxel_grid()) < np.std(image_ref) + 0.01
    for ii in np.arange(len(q_match)):
        assert q_ref[ii] - 0.01 < q_match[ii] < q_ref[ii] + 0.01

    # Test with normalisation
    # No tissue mask.
    data_no_mask = extract_features_and_images(
        tissue_mask_type="none",
        intensity_normalisation="match_reference_normalised",
        intensity_normalisation_reference=image_ref,
        **GENERIC_KWARGS
    )

    # Tissue mask
    data_mask = extract_features_and_images(
        tissue_mask_type="relative_range",
        tissue_mask_range=[0.02, 1.00],
        intensity_normalisation="match_reference_normalised",
        intensity_normalisation_reference=image_ref,
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

    assert np.min(image_nm.get_voxel_grid()) == 0.0
    assert np.max(image_nm.get_voxel_grid()) == 1.0
    assert np.min(image_m.get_voxel_grid()) == 0.0
    assert np.max(image_m.get_voxel_grid()) == 1.0
    assert 0.009 < np.std(image_nm.get_voxel_grid()) < 0.010
    assert 0.008 < np.std(image_m.get_voxel_grid()) < 0.009