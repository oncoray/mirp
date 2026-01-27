import os.path

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
    from mirp._images.ct_image import CTImage

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

        if intensity_normalisation_method == "none":
            assert isinstance(image_nm, CTImage)
            assert isinstance(image_m, CTImage)
        else:
            assert isinstance(image_nm, GenericImage)
            assert isinstance(image_m, GenericImage)

        if intensity_normalisation_method == "none":
            assert feature_data_nm["stat_max"].to_numpy()[0] == 350.0
            assert feature_data_m["stat_max"].to_numpy()[0] == 350.0
            assert feature_data_nm["stat_min"].to_numpy()[0] == 0.0
            assert feature_data_m["stat_min"].to_numpy()[0] == 0.0

        else:
            raise NotImplementedError(f"Test for intensity_normalisation_method {intensity_normalisation_method} not implemented")


def test_custom_scale():
    ...