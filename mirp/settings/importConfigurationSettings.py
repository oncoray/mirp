import copy
import warnings
from typing import Union, List
from xml.etree.ElementTree import Element
from xml.etree import ElementTree as ElemTree

from mirp.settings.settingsGeneric import SettingsClass
from mirp.settings.settingsImageTransformation import ImageTransformationSettingsClass
from mirp.settings.settingsFeatureExtraction import FeatureExtractionSettingsClass
from mirp.settings.settingsMaskResegmentation import ResegmentationSettingsClass
from mirp.settings.settingsPerturbation import ImagePerturbationSettingsClass
from mirp.settings.settingsImageProcessing import ImagePostProcessingClass, get_post_processing_settings
from mirp.settings.settingsInterpolation import (ImageInterpolationSettingsClass, MaskInterpolationSettingsClass,
                                                 get_image_interpolation_settings, get_mask_interpolation_settings)
from mirp.settings.settingsGeneral import GeneralSettingsClass, get_general_settings
from mirp.settings.utilities import str2list, str2type, read_node, update_settings_from_branch


def import_configuration_generator(
    xml_tree: None | Element = None,
    **kwargs
):
    kwargs = copy.deepcopy(kwargs)

    if isinstance(xml_tree, Element):
        # General settings
        update_settings_from_branch(
            kwargs=kwargs,
            branch=xml_tree.find("general"),
            settings=get_general_settings()
        )

        # Post-processing settings
        update_settings_from_branch(
            kwargs=kwargs,
            branch=xml_tree.find("post_processing"),
            settings=get_post_processing_settings()
        )

        # Image interpolation settings
        update_settings_from_branch(
            kwargs=kwargs,
            branch=xml_tree.find("img_interpolate"),
            settings=get_image_interpolation_settings()
        )

        if xml_tree.find("img_interpolate") is not None and xml_tree.find("img_interpolate").find("new_non_iso_spacing") is not None:
            warnings.warn(
                f"The new_non_iso_spacing tag has been deprecated. Use the new_spacing tag instead.",
                DeprecationWarning
            )

        # Mask interpolation settings
        update_settings_from_branch(
            kwargs=kwargs,
            branch=xml_tree.find("roi_interpolate"),
            settings=get_mask_interpolation_settings()
        )

    # Create settings class.
    settings = SettingsClass(**kwargs)

    yield settings


def import_configuration_settings(
        compute_features: bool,
        path: Union[None, str] = None,
        **kwargs
) -> List[SettingsClass]:
    import os.path

    # Make a copy of the kwargs argument to avoid updating by reference.
    kwargs = copy.deepcopy(kwargs)

    # Check if a configuration file is used.
    if path is None:

        # Prevent checking of feature parameters if features are not computed.
        if not compute_features:
            kwargs.update({
                "base_feature_families": "none",
                "response_map_feature_families": "none"})
        else:
            if "base_feature_families" not in kwargs:
                kwargs.update({"base_feature_families": "all"})
            if "response_map_feature_families" not in kwargs:
                kwargs.update({"response_map_feature_families": "statistics"})

        # Set general settings.
        general_settings = GeneralSettingsClass(**kwargs)

        # Remove by_slice from the keyword arguments to avoid double passing.
        kwargs.pop("by_slice", None)
        kwargs.pop("no_approximation", None)

        # Set image interpolation settings
        image_interpolation_settings = ImageInterpolationSettingsClass(
            by_slice=general_settings.by_slice,
            **kwargs)

        # Set ROI interpolation settings
        roi_interpolation_settings = MaskInterpolationSettingsClass(**kwargs)

        # Set post-processing settings
        post_processing_settings = ImagePostProcessingClass(**kwargs)

        # Set perturbation settings
        perturbation_settings = ImagePerturbationSettingsClass(**kwargs)

        # Set resegmentation settings
        resegmentation_settings = ResegmentationSettingsClass(**kwargs)

        # Set feature extraction settings
        feature_extraction_settings = FeatureExtractionSettingsClass(
            by_slice=general_settings.by_slice,
            no_approximation=general_settings.no_approximation,
            **kwargs)

        # Set image transformation settings.
        image_transformation_settings = ImageTransformationSettingsClass(
            by_slice=general_settings.by_slice,
            response_map_feature_settings=feature_extraction_settings,
            **kwargs)

        return [SettingsClass(
            general_settings=general_settings,
            img_interpolate_settings=image_interpolation_settings,
            roi_interpolate_settings=roi_interpolation_settings,
            post_process_settings=post_processing_settings,
            perturbation_settings=perturbation_settings,
            roi_resegment_settings=resegmentation_settings,
            feature_extr_settings=feature_extraction_settings,
            img_transform_settings=image_transformation_settings)
        ]

    elif not os.path.exists(path):
        raise FileNotFoundError(f"The settings file could not be found at {path}.")

    # Load xml
    tree = ElemTree.parse(path)
    root = tree.getroot()

    # Empty list for settings
    settings_list = []

    # Set default values for feature families.
    base_feature_families = "all"
    response_map_feature_families = "statistical"

    # Prevent checking of feature parameters if features are not computed.
    if not compute_features:
        base_feature_families = None
        response_map_feature_families = None

    for branch in root.findall("config"):

        # Process general settings
        general_branch = branch.find("general")

        if general_branch is not None:
            general_settings = GeneralSettingsClass(
                by_slice=str2type(general_branch.find("by_slice"), "bool", False),
                mask_merge=str2type(general_branch.find("mask_merge"), "bool", False),
                mask_split=str2type(general_branch.find("mask_split"), "bool", False),
                mask_select_largest_region=str2type(general_branch.find("mask_select_largest_region"), "bool", False),
                mask_select_largest_slice=str2type(general_branch.find("mask_select_largest_slice"), "bool", False),
                config_str=str2type(general_branch.find("config_str"), "str", ""),
                no_approximation=str2type(general_branch.find("no_approximation"), "bool", False)
            )

        else:
            general_settings = GeneralSettingsClass()

        # Process image interpolation settings
        img_interp_branch = branch.find("img_interpolate")

        if img_interp_branch is not None:

            if img_interp_branch.find("new_non_iso_spacing") is not None:
                warnings.warn(f"The new_non_iso_spacing tag has been deprecated. Use the new_spacing tag instead.",
                              DeprecationWarning)

            img_interp_settings = ImageInterpolationSettingsClass(
                by_slice=general_settings.by_slice,
                spline_order=str2type(img_interp_branch.find("spline_order"), "int", 3),
                new_spacing=str2list(img_interp_branch.find("new_spacing"), "float", None),
                anti_aliasing=str2type(img_interp_branch.find("anti_aliasing"), "bool", True),
                smoothing_beta=str2type(img_interp_branch.find("smoothing_beta"), "float", 0.98)
            )

        else:
            img_interp_settings = ImageInterpolationSettingsClass(by_slice=general_settings.by_slice)

        # Process roi interpolation settings
        roi_interp_branch = branch.find("roi_interpolate")

        if roi_interp_branch is not None:
            roi_interp_settings = MaskInterpolationSettingsClass(
                roi_spline_order=str2type(roi_interp_branch.find("spline_order"), "int", 1),
                roi_interpolation_mask_inclusion_threshold=str2type(roi_interp_branch.find("incl_threshold"), "float", 0.5))

        else:
            roi_interp_settings = MaskInterpolationSettingsClass()

        # Image post-acquisition processing settings
        post_process_branch = branch.find("post_processing")

        if post_process_branch is not None:
            post_process_settings = ImagePostProcessingClass(
                bias_field_correction=str2type(post_process_branch.find("bias_field_correction"), "bool", False),
                bias_field_correction_n_fitting_levels=str2type(post_process_branch.find("n_fitting_levels"), "int", 3),
                bias_field_correction_n_max_iterations=str2list(post_process_branch.find("n_max_iterations"), "int", 100),
                bias_field_convergence_threshold=str2type(post_process_branch.find("convergence_threshold"), "float", 0.001),
                intensity_normalisation=str2type(post_process_branch.find("intensity_normalisation"), "str", "none"),
                intensity_normalisation_range=str2list(post_process_branch.find("intensity_normalisation_range"), "float", None),
                intensity_normalisation_saturation=str2list(post_process_branch.find("intensity_normalisation_saturation"), "float", None),
                tissue_mask_type=str2type(post_process_branch.find("tissue_mask_type"), "str", "relative_range"),
                tissue_mask_range=str2list(post_process_branch.find("tissue_mask_range"), "float", None))

        else:
            post_process_settings = ImagePostProcessingClass()

        # Image and roi volume adaptation settings
        perturbation_branch = branch.find("vol_adapt")

        if perturbation_branch is not None:
            perturbation_settings = ImagePerturbationSettingsClass(
                crop_around_roi=str2type(read_node(
                    perturbation_branch, ["crop_around_roi", "resect"]), "bool", False),
                crop_distance=str2type(perturbation_branch.find("crop_distance"), "float", 150.0),
                perturbation_noise_repetitions=str2type(perturbation_branch.find("noise_repetitions"), "int", 0),
                perturbation_noise_level=str2type(perturbation_branch.find("noise_level"), "float"),
                perturbation_rotation_angles=str2list(read_node(
                    perturbation_branch, ["rotation_angles", "rot_angles"]), "float", 0.0),
                perturbation_translation_fraction=str2list(read_node(
                    perturbation_branch, ["translation_fraction", "translate_frac"]), "float", 0.0),
                perturbation_roi_adapt_type=str2type(perturbation_branch.find("roi_adapt_type"), "str", "distance"),
                perturbation_roi_adapt_size=str2list(perturbation_branch.find("roi_adapt_size"), "float", 0.0),
                perturbation_roi_adapt_max_erosion=str2type(read_node(
                    perturbation_branch, ["roi_adapt_max_erosion", "eroded_vol_fract"]), "float", 0.8),
                perturbation_randomise_roi_repetitions=str2type(perturbation_branch.find("roi_randomise_repetitions"), "int", 0),
                roi_split_boundary_size=str2list(perturbation_branch.find("roi_boundary_size"), "float", 0.0),
                roi_split_max_erosion=str2type(read_node(
                    perturbation_branch, ["roi_split_max_erosion", "bulk_min_vol_fract"]), "float", 0.6))

        else:
            perturbation_settings = ImagePerturbationSettingsClass()

        # Process roi segmentation settings
        roi_resegment_branch = branch.find("roi_resegment")

        if roi_resegment_branch is not None:
            roi_resegment_settings = ResegmentationSettingsClass(
                resegmentation_intensity_range=str2list(read_node(
                    roi_resegment_branch, ["intensity_range", "g_thresh"]), "float", None),
                resegmentation_sigma=str2type(roi_resegment_branch.find("sigma"), "float", 3.0))

        else:
            roi_resegment_settings = ResegmentationSettingsClass()

        # Process feature extraction settings
        feature_extr_branch = branch.find("feature_extr")

        if feature_extr_branch is not None:
            if feature_extr_branch.find("glcm_merge_method") is not None:
                warnings.warn(
                    "The glcm_merge_method tag has been deprecated. Use the glcm_spatial_method tag instead. This takes"
                    " the following values: `2d_average`, `2d_slice_merge`, '2.5d_direction_merge', '2.5d_volume_merge',"
                    " '3d_average', and `3d_volume_merge`",
                    DeprecationWarning
                )

            if feature_extr_branch.find("glrlm_merge_method") is not None:
                warnings.warn(
                    "The glrlm_merge_method tag has been deprecated. Use the glrlm_spatial_method tag instead. This "
                    "takes the following values: `2d_average`, `2d_slice_merge`, '2.5d_direction_merge', "
                    "'2.5d_volume_merge', '3d_average', and `3d_volume_merge`",
                    DeprecationWarning
                )

            feature_extr_settings = FeatureExtractionSettingsClass(
                by_slice=general_settings.by_slice,
                no_approximation=general_settings.no_approximation,
                base_feature_families=str2list(read_node(
                    feature_extr_branch, ["feature_families", "families"]), "str", base_feature_families),
                base_discretisation_method=str2list(read_node(
                    feature_extr_branch, ["discretisation_method", "discr_method"]), "str", None),
                base_discretisation_n_bins=str2list(read_node(
                    feature_extr_branch, ["discretisation_n_bins", "discr_n_bins"]), "int", None),
                base_discretisation_bin_width=str2list(read_node(
                    feature_extr_branch, ["discretisation_bin_width", "discr_bin_width"]), "float", None),
                ivh_discretisation_method=str2type(read_node(
                    feature_extr_branch, ["ivh_discretisation_method", "ivh_discr_method"]), "str", "none"),
                ivh_discretisation_n_bins=str2type(read_node(
                    feature_extr_branch, ["ivh_discretisation_n_bins", "ivh_discr_n_bins"]), "int", 1000),
                ivh_discretisation_bin_width=str2type(read_node(
                    feature_extr_branch, ["ivh_discretisation_bin_width", "ivh_discr_bin_width"]), "float", None),
                glcm_distance=str2list(read_node(
                    feature_extr_branch, ["glcm_distance", "glcm_dist"]), "float", 1.0),
                glcm_spatial_method=str2list(feature_extr_branch.find("glcm_spatial_method"), "str", None),
                glrlm_spatial_method=str2list(feature_extr_branch.find("glrlm_spatial_method"), "str", None),
                glszm_spatial_method=str2list(feature_extr_branch.find("glszm_spatial_method"), "str", None),
                gldzm_spatial_method=str2list(feature_extr_branch.find("gldzm_spatial_method"), "str", None),
                ngtdm_spatial_method=str2list(feature_extr_branch.find("ngtdm_spatial_method"), "str", None),
                ngldm_distance=str2list(read_node(
                    feature_extr_branch, ["ngldm_distance", "ngldm_dist"]), "float", 1.0),
                ngldm_difference_level=str2list(read_node(
                    feature_extr_branch, ["ngldm_difference_level", "ngldm_diff_lvl"]), "float", 0.0),
                ngldm_spatial_method=str2list(feature_extr_branch.find("ngldm_spatial_method"), "str", None)
            )

        else:
            # If the section is absent, assume that no features are extracted.
            feature_extr_settings = FeatureExtractionSettingsClass(
                by_slice=general_settings.by_slice,
                no_approximation=general_settings.no_approximation,
                base_feature_families=None)

        # Process image transformation settings
        img_transform_branch = branch.find("img_transform")

        if img_transform_branch is not None:
            if img_transform_branch.find("log_average") is not None:
                warnings.warn(
                    "The log_average tag has been deprecated. Use the laplacian_of_gaussian_pooling_method tag "
                    "instead with the value `mean` to emulate log_average=True.",
                    DeprecationWarning
                )

            if img_transform_branch.find("riesz_steered") is not None:
                warnings.warn(
                    "The riesz_steered tag has been deprecated. Steerable Riesz filter are now identified by the name "
                    "of the filter kernel (filter_kernels parameter).",
                    DeprecationWarning
                )

            img_transform_settings = ImageTransformationSettingsClass(
                by_slice=general_settings.by_slice,
                response_map_feature_settings=feature_extr_settings,
                response_map_feature_families=str2list(
                    img_transform_branch.find("feature_families"), "str", response_map_feature_families),
                response_map_discretisation_method=str2list(
                    img_transform_branch.find("discretisation_method"), "str", "fixed_bin_number"),
                response_map_discretisation_bin_width=str2list(
                    img_transform_branch.find("discretisation_bin_width"), "float", None),
                response_map_discretisation_n_bins=str2list(
                    img_transform_branch.find("discretisation_n_bins"), "int", 16),
                filter_kernels=str2list(read_node(
                    img_transform_branch, ["filter_kernels", "spatial_filters"]), "str", None),
                boundary_condition=str2type(img_transform_branch.find("boundary_condition"), "str", "mirror"),
                separable_wavelet_families=str2list(read_node(
                    img_transform_branch, "separable_wavelet_families", "wavelet_fam"), "str", None),
                separable_wavelet_set=str2list(read_node(
                    img_transform_branch, "separable_wavelet_set", "wavelet_filter_set"), "str", "all"),
                separable_wavelet_stationary=True,
                separable_wavelet_decomposition_level=str2list(read_node(
                    img_transform_branch, "separable_wavelet_decomposition_level", "wavelet_decomposition_level"),
                    "int", 1),
                separable_wavelet_rotation_invariance=str2type(read_node(
                    img_transform_branch, "separable_wavelet_rotation_invariance", "wavelet_rot_invar"), "bool", True),
                separable_wavelet_pooling_method=str2type(read_node(
                    img_transform_branch, "separable_wavelet_pooling_method", "wavelet_pooling_method"), "str", "max"),
                separable_wavelet_boundary_condition=str2type(
                    img_transform_branch.find("separable_wavelet_boundary_condition"), "str", None),
                nonseparable_wavelet_families=str2list(read_node(
                    img_transform_branch, "nonseparable_wavelet_families"), "str", None),
                nonseparable_wavelet_decomposition_level=str2list(read_node(
                    img_transform_branch, "nonseparable_wavelet_decomposition_level", "wavelet_decomposition_level"),
                    "int", 1),
                nonseparable_wavelet_response=str2type(
                    img_transform_branch.find("nonseparable_wavelet_response"), "str", "real"),
                nonseparable_wavelet_boundary_condition=str2type(
                    img_transform_branch.find("nonseparable_wavelet_boundary_condition"), "str", None),
                gaussian_sigma=str2list(read_node(
                    img_transform_branch, ["gaussian_sigma", "gauss_sigma"]), "float", None),
                gaussian_kernel_truncate=str2type(read_node(
                    img_transform_branch, ["gaussian_kernel_truncate", "gaussian_sigma_truncate"]), "float", 4.0),
                gaussian_kernel_boundary_condition=str2type(
                    img_transform_branch.find("gaussian_kernel_boundary_condition"), "str", None),
                laplacian_of_gaussian_sigma=str2list(read_node(
                    img_transform_branch, ["laplacian_of_gaussian_sigma", "log_sigma"]), "float", None),
                laplacian_of_gaussian_kernel_truncate=str2type(read_node(
                    img_transform_branch, ["laplacian_of_gaussian_kernel_truncate", "log_sigma_truncate"]),
                    "float", 4.0),
                laplacian_of_gaussian_pooling_method=str2type(
                    img_transform_branch.find("laplacian_of_gaussian_pooling_method"), "str", "none"),
                laplacian_of_gaussian_boundary_condition=str2type(
                    img_transform_branch.find("laplacian_of_gaussian_boundary_condition"), "str", None),
                laws_kernel=str2list(img_transform_branch.find("laws_kernel"), "str", None),
                laws_compute_energy=str2type(
                    img_transform_branch.find("laws_calculate_energy"), "bool", True),
                laws_delta=str2list(img_transform_branch.find("laws_delta"), "int", 7),
                laws_rotation_invariance=str2type(read_node(
                    img_transform_branch, ["laws_rotation_invariance", "laws_rot_invar"]), "bool", True),
                laws_pooling_method=str2type(img_transform_branch.find("laws_pooling_method"), "str", "max"),
                laws_boundary_condition=str2type(img_transform_branch.find('laws_boundary_condition'), "str", None),
                gabor_sigma=str2list(img_transform_branch.find("gabor_sigma"), "float", None),
                gabor_lambda=str2list(img_transform_branch.find("gabor_lambda"), "float", None),
                gabor_kernel_truncate=str2type(read_node(
                    img_transform_branch, ["gabor_kernel_truncate", "gabor_sigma_truncate"]), "float", 10.0),
                gabor_gamma=str2list(img_transform_branch.find("gabor_gamma"), "float", 1.0),
                gabor_theta=str2list(read_node(
                    img_transform_branch, "gabor_theta", "gabor_theta_initial"), "float", 0.0),
                gabor_theta_step=str2type(img_transform_branch.find("gabor_theta_step"), "float", None),
                gabor_response=str2type(img_transform_branch.find("gabor_response"), "str", "modulus"),
                gabor_rotation_invariance=str2type(read_node(
                    img_transform_branch, ["gabor_rotation_invariance", "gabor_rot_invar"]), "bool", True),
                gabor_pooling_method=str2type(img_transform_branch.find("gabor_pooling_method"), "str", "max"),
                gabor_boundary_condition=str2type(img_transform_branch.find('gabor_boundary_condition'), "str", None),
                mean_filter_kernel_size=str2list(read_node(
                    img_transform_branch, ["mean_filter_kernel_size", "mean_filter_size"]), "int", None),
                mean_filter_boundary_condition=str2type(
                    img_transform_branch.find('mean_filter_boundary_condition'), "str", None),
                riesz_filter_order=str2list(read_node(
                    img_transform_branch, ["riesz_filter_order", "riesz_order"]), "int", None),
                riesz_filter_tensor_sigma=str2list(img_transform_branch.find("riesz_filter_tensor_sigma"), "float", None)
            )

        else:
            img_transform_settings = ImageTransformationSettingsClass(
                by_slice=general_settings.by_slice,
                response_map_feature_settings=feature_extr_settings,
                response_map_feature_families=None)

        # Deep learning branch
        deep_learning_branch = branch.find("deep_learning")

        if deep_learning_branch is not None:
            warnings.warn(
                "deep_learning parameter branch has been deprecated. Parameters for image  "
                "processing for deep learning can now be set directly using the  "
                "extract_images_for_deep_learning function.",
                DeprecationWarning
            )

        # Parse to settings
        settings_list += [SettingsClass(
            general_settings=general_settings,
            img_interpolate_settings=img_interp_settings,
            roi_interpolate_settings=roi_interp_settings,
            post_process_settings=post_process_settings,
            perturbation_settings=perturbation_settings,
            roi_resegment_settings=roi_resegment_settings,
            feature_extr_settings=feature_extr_settings,
            img_transform_settings=img_transform_settings)]

    return settings_list
