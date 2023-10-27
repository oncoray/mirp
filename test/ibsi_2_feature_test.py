import os

from mirp.extractFeaturesAndImages import extract_features
from mirp.settings.settingsGeneric import SettingsClass
from mirp.settings.settingsImageTransformation import ImageTransformationSettingsClass
from mirp.settings.settingsFeatureExtraction import FeatureExtractionSettingsClass
from mirp.settings.settingsMaskResegmentation import ResegmentationSettingsClass
from mirp.settings.settingsPerturbation import ImagePerturbationSettingsClass
from mirp.settings.settingsImageProcessing import ImagePostProcessingClass
from mirp.settings.settingsInterpolation import ImageInterpolationSettingsClass, MaskInterpolationSettingsClass
from mirp.settings.settingsGeneral import GeneralSettingsClass

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PERTURB_IMAGES = False
WRITE_TEMP_FILES = True


def within_tolerance(ref, tol, x):
    # Read from pandas Series
    x = x.values[0]

    if abs(x - ref) <= tol:
        return True
    else:
        return False


def _get_default_settings(
        by_slice: bool,
        configuration_id: str,
        base_feature_families="none"):
    """Set default settings for response map tests."""

    general_settings = GeneralSettingsClass(
        by_slice=by_slice,
        config_str=configuration_id
    )

    # 2D-analysis does not use interpolation, whereas the volumetric (3D) analysis does.
    if by_slice:
        image_interpolation_settings = ImageInterpolationSettingsClass(
            by_slice=by_slice,
            anti_aliasing=False
        )

    else:
        image_interpolation_settings = ImageInterpolationSettingsClass(
            by_slice=by_slice,
            spline_order=3,
            new_spacing=1.0,
            anti_aliasing=False
        )

    resegmentation_settings = ResegmentationSettingsClass(
        resegmentation_intensity_range=[-1000.0, 400.0]
    )

    if PERTURB_IMAGES:
        perturbation_settings = ImagePerturbationSettingsClass(
            crop_around_roi=False,
            perturbation_rotation_angles=[-15.0, -10.0, 5.0, 0.0, 5.0, 10.0, 15.0],
            perturbation_translation_fraction=[0.00, 0.25, 0.50, 0.75],
            perturbation_roi_adapt_size=[-2.0, 0.0, 2.0],
            perturbation_roi_adapt_type="distance"
        )

    else:
        perturbation_settings = ImagePerturbationSettingsClass(
            crop_around_roi=False
        )

    feature_computation_parameters = FeatureExtractionSettingsClass(
        by_slice=by_slice,
        no_approximation=True,
        base_feature_families=base_feature_families,
    )

    return general_settings, image_interpolation_settings, feature_computation_parameters, resegmentation_settings, perturbation_settings


def _process_experiment(
        configuration_id: str,
        by_slice: bool,
        image_transformation_settings: ImageTransformationSettingsClass,
        base_feature_families: str = "none"):

    # Set testing directory
    test_dir = os.path.join(CURRENT_DIR, "data", "temp")
    if not os.path.isdir(test_dir) and WRITE_TEMP_FILES:
        os.makedirs(test_dir)

    # Get default settings.
    general_settings, image_interpolation_settings, feature_computation_parameters, resegmentation_settings, \
    perturbation_settings = _get_default_settings(
        by_slice=by_slice,
        configuration_id=configuration_id,
        base_feature_families=base_feature_families)

    settings = SettingsClass(
        general_settings=general_settings,
        post_process_settings=ImagePostProcessingClass(),
        img_interpolate_settings=image_interpolation_settings,
        roi_interpolate_settings=MaskInterpolationSettingsClass(),
        roi_resegment_settings=resegmentation_settings,
        perturbation_settings=perturbation_settings,
        img_transform_settings=image_transformation_settings,
        feature_extr_settings=feature_computation_parameters
    )

    data = extract_features(
        write_features=False,
        export_features=True,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        settings=settings
    )

    data = data[0]

    if WRITE_TEMP_FILES:
        file_name = [configuration_id, "perturb", "features.csv"] if PERTURB_IMAGES else [configuration_id, "features.csv"]

        data.to_csv(
            os.path.join(test_dir, "_".join(file_name)),
            sep=";",
            decimal=".",
            index=False
        )

    return data


def test_ibsi_2_config_none():
    """
    Compare computed feature values with reference values for configurations 1A and 1B of IBSI 2 phase 2.
    """

    # Configuration 1.A ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels=None
    )

    data = _process_experiment(
        configuration_id="1.A",
        by_slice=True,
        image_transformation_settings=image_transformation_settings,
        base_feature_families="statistics"
    )

    assert (within_tolerance(3.65, 0.34, data["stat_kurt"]))
    assert (within_tolerance(-434, 21, data["stat_p10"]))
    assert (within_tolerance(93, 0.1, data["stat_p90"]))
    assert (within_tolerance(-4.92, 0.42, data["stat_cov"]))
    assert (within_tolerance(6.96e+09, 4.5e+08, data["stat_energy"]))
    assert (within_tolerance(69, 6.4, data["stat_iqr"]))
    assert (within_tolerance(377, 10, data["stat_max"]))
    assert (within_tolerance(-47, 4.6, data["stat_mean"]))
    assert (within_tolerance(160, 5, data["stat_mad"]))
    assert (within_tolerance(41, 0.6, data["stat_median"]))
    assert (within_tolerance(122, 4, data["stat_medad"]))
    assert (within_tolerance(-1000, 10, data["stat_min"]))
    assert (within_tolerance(1, 0.85, data["stat_qcod"]))
    assert (within_tolerance(1380, 10, data["stat_range"]))
    assert (within_tolerance(64.4, 5.8, data["stat_rmad"]))
    assert (within_tolerance(236, 5, data["stat_rms"]))
    assert (within_tolerance(-2.17, 0.07, data["stat_skew"]))
    assert (within_tolerance(53300, 2000, data["stat_var"]))

    # Configuration 1.B ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels=None
    )

    data = _process_experiment(
        configuration_id="1.B",
        by_slice=False,
        image_transformation_settings=image_transformation_settings,
        base_feature_families="statistics"
    )

    assert (within_tolerance(3.71, 0.47, data["stat_kurt"]))
    assert (within_tolerance(-427, 29, data["stat_p10"]))
    assert (within_tolerance(92, 0.1, data["stat_p90"]))
    assert (within_tolerance(-4.94, 0.64, data["stat_cov"]))
    assert (within_tolerance(1.96e+10, 1.9e+09, data["stat_energy"]))
    assert (within_tolerance(67, 9.1, data["stat_iqr"]))
    assert (within_tolerance(377, 15, data["stat_max"]))
    assert (within_tolerance(-46.4, 5.9, data["stat_mean"]))
    assert (within_tolerance(159, 7, data["stat_mad"]))
    assert (within_tolerance(41, 0.7, data["stat_median"]))
    assert (within_tolerance(121, 6, data["stat_medad"]))
    assert (within_tolerance(-997, 3, data["stat_min"]))
    assert (within_tolerance(0.944, 0.925, data["stat_qcod"]))
    assert (within_tolerance(1370, 20, data["stat_range"]))
    assert (within_tolerance(63.6, 7.3, data["stat_rmad"]))
    assert (within_tolerance(234, 7, data["stat_rms"]))
    assert (within_tolerance(-2.18, 0.09, data["stat_skew"]))
    assert (within_tolerance(52600, 2800, data["stat_var"]))


def test_ibsi_2_config_mean_filter():
    """
    Compare computed feature values with reference values for configurations 2A and 2B of IBSI 2 phase 2.
    """

    # Configuration 2.A ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="mean",
        mean_filter_kernel_size=5
    )

    data = _process_experiment(
        configuration_id="2.A",
        by_slice=True,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("mean_d_5_", "") for column_name in data.columns]

    assert (within_tolerance(3.53, 0.33, data["stat_kurt"]))
    assert (within_tolerance(-402, 18, data["stat_p10"]))
    assert (within_tolerance(79.8, 0.1, data["stat_p90"]))
    assert (within_tolerance(-4.33, 0.34, data["stat_cov"]))
    assert (within_tolerance(6.15e+09, 3.9e+08, data["stat_energy"]))
    assert (within_tolerance(82, 10.5, data["stat_iqr"]))
    assert (within_tolerance(334, 7, data["stat_max"]))
    assert (within_tolerance(-49.9, 4.4, data["stat_mean"]))
    assert (within_tolerance(153, 5, data["stat_mad"]))
    assert (within_tolerance(38.1, 0.7, data["stat_median"]))
    assert (within_tolerance(116, 4, data["stat_medad"]))
    assert (within_tolerance(-889, 3, data["stat_min"]))
    assert (within_tolerance(1.85, 0.66, data["stat_qcod"]))
    assert (within_tolerance(1220, 10, data["stat_range"]))
    assert (within_tolerance(67.7, 5.6, data["stat_rmad"]))
    assert (within_tolerance(222, 5, data["stat_rms"]))
    assert (within_tolerance(-2.13, 0.07, data["stat_skew"]))
    assert (within_tolerance(46600, 1600, data["stat_var"]))

    # Configuration 2.B ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="mean",
        mean_filter_kernel_size=5
    )

    data = _process_experiment(
        configuration_id="2.B",
        by_slice=False,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("mean_d_5_", "") for column_name in data.columns]

    assert (within_tolerance(3.59, 0.46, data["stat_kurt"]))
    assert (within_tolerance(-389, 25, data["stat_p10"]))
    assert (within_tolerance(77.2, 0.1, data["stat_p90"]))
    assert (within_tolerance(-4.22, 0.47, data["stat_cov"]))
    assert (within_tolerance(1.68e+10, 1.6e+09, data["stat_energy"]))
    assert (within_tolerance(92.6, 13.5, data["stat_iqr"]))
    assert (within_tolerance(316, 7, data["stat_max"]))
    assert (within_tolerance(-49.9, 5.7, data["stat_mean"]))
    assert (within_tolerance(149, 6, data["stat_mad"]))
    assert (within_tolerance(37.3, 0.6, data["stat_median"]))
    assert (within_tolerance(114, 5, data["stat_medad"]))
    assert (within_tolerance(-906, 5, data["stat_min"]))
    assert (within_tolerance(2.97, 0.58, data["stat_qcod"]))
    assert (within_tolerance(1220, 10, data["stat_range"]))
    assert (within_tolerance(68.1, 6.9, data["stat_rmad"]))
    assert (within_tolerance(217, 7, data["stat_rms"]))
    assert (within_tolerance(-2.13, 0.09, data["stat_skew"]))
    assert (within_tolerance(44400, 2300, data["stat_var"]))


def test_ibsi_2_config_laplacian_of_gaussian_filter():
    """
    Compare computed feature values with reference values for configurations 3A and 3B of IBSI 2 phase 2.
    """

    # Configuration 3.A ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="laplacian_of_gaussian",
        laplacian_of_gaussian_sigma=1.5,
        laplacian_of_gaussian_kernel_truncate=4.0
    )

    data = _process_experiment(
        configuration_id="3.A",
        by_slice=True,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("log_s_1.5_", "") for column_name in data.columns]

    assert (within_tolerance(7.02, 0.3, data["stat_kurt"]))
    assert (within_tolerance(-27.1, 0.5, data["stat_p10"]))
    assert (within_tolerance(13.2, 1.3, data["stat_p90"]))
    assert (within_tolerance(-9.66, 1.21, data["stat_cov"]))
    assert (within_tolerance(70300000, 4200000, data["stat_energy"]))
    assert (within_tolerance(8.74, 0.19, data["stat_iqr"]))
    assert (within_tolerance(191, 1, data["stat_max"]))
    assert (within_tolerance(-2.44, 0.12, data["stat_mean"]))
    assert (within_tolerance(13.2, 0.3, data["stat_mad"]))
    assert (within_tolerance(-0.585, 0.012, data["stat_median"]))
    assert (within_tolerance(12.9, 0.4, data["stat_medad"]))
    assert (within_tolerance(-178, 1, data["stat_min"]))
    assert (within_tolerance(-2.93, 0.06, data["stat_qcod"]))
    assert (within_tolerance(369, 2, data["stat_range"]))
    assert (within_tolerance(4.98, 0.16, data["stat_rmad"]))
    assert (within_tolerance(23.7, 0.5, data["stat_rms"]))
    assert (within_tolerance(0.295, 0.027, data["stat_skew"]))
    assert (within_tolerance(556, 24, data["stat_var"]))

    # Configuration 3.B ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="laplacian_of_gaussian",
        laplacian_of_gaussian_sigma=1.5,
        laplacian_of_gaussian_kernel_truncate=4.0
    )

    data = _process_experiment(
        configuration_id="3.B",
        by_slice=False,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("log_s_1.5_", "") for column_name in data.columns]

    assert (within_tolerance(6.13, 0.27, data["stat_kurt"]))
    assert (within_tolerance(-32.2, 0.5, data["stat_p10"]))
    assert (within_tolerance(17.4, 1.9, data["stat_p90"]))
    assert (within_tolerance(-9.12, 1.63, data["stat_cov"]))
    assert (within_tolerance(2.61e+08, 1.9e+07, data["stat_energy"]))
    assert (within_tolerance(11.4, 0.3, data["stat_iqr"]))
    assert (within_tolerance(204, 1, data["stat_max"]))
    assert (within_tolerance(-2.94, 0.2, data["stat_mean"]))
    assert (within_tolerance(15.5, 0.4, data["stat_mad"]))
    assert (within_tolerance(-0.919, 0.024, data["stat_median"]))
    assert (within_tolerance(15.3, 0.4, data["stat_medad"]))
    assert (within_tolerance(-173, 5, data["stat_min"]))
    assert (within_tolerance(-2.34, 0.07, data["stat_qcod"]))
    assert (within_tolerance(377, 5, data["stat_range"]))
    assert (within_tolerance(6.37, 0.19, data["stat_rmad"]))
    assert (within_tolerance(27, 0.6, data["stat_rms"]))
    assert (within_tolerance(0.428, 0.009, data["stat_skew"]))
    assert (within_tolerance(720, 33, data["stat_var"]))


def test_ibsi_2_config_laws_filter():
    """
    Compare computed feature values with reference values for configurations 4A and 4B of IBSI 2 phase 2.
    """

    # Configuration 4.A ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="laws",
        laws_kernel="L5E5",
        laws_compute_energy=True,
        laws_rotation_invariance=True,
        laws_delta=7,
        laws_pooling_method="max"
    )

    data = _process_experiment(
        configuration_id="4.A",
        by_slice=True,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("laws_l5e5_energy_delta_7_invar_", "") for column_name in data.columns]

    assert (within_tolerance(-0.279, 0.039, data["stat_kurt"]))
    assert (within_tolerance(26.9, 0.1, data["stat_p10"]))
    assert (within_tolerance(347, 4, data["stat_p90"]))
    assert (within_tolerance(0.904, 0.007, data["stat_cov"]))
    assert (within_tolerance(5e+09, 2.7e+08, data["stat_energy"]))
    assert (within_tolerance(219, 5, data["stat_iqr"]))
    assert (within_tolerance(672, 4, data["stat_max"]))
    assert (within_tolerance(148, 3, data["stat_mean"]))
    assert (within_tolerance(116, 2, data["stat_mad"]))
    assert (within_tolerance(88.7, 5, data["stat_median"]))
    assert (within_tolerance(111, 2, data["stat_medad"]))
    assert (within_tolerance(15.5, 0.1, data["stat_min"]))
    assert (within_tolerance(0.773, 0.003, data["stat_qcod"]))
    assert (within_tolerance(657, 4, data["stat_range"]))
    assert (within_tolerance(92.9, 1.8, data["stat_rmad"]))
    assert (within_tolerance(200, 4, data["stat_rms"]))
    assert (within_tolerance(0.877, 0.024, data["stat_skew"]))
    assert (within_tolerance(18000, 500, data["stat_var"]))

    # Configuration 4.B ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="laws",
        laws_kernel="L5E5E5",
        laws_compute_energy=True,
        laws_rotation_invariance=True,
        laws_delta=7,
        laws_pooling_method="max"
    )

    data = _process_experiment(
        configuration_id="4.B",
        by_slice=False,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("laws_l5e5e5_energy_delta_7_invar_", "") for column_name in data.columns]

    assert (within_tolerance(-0.711, 0.044, data["stat_kurt"]))
    assert (within_tolerance(35.6, 0.1, data["stat_p10"]))
    assert (within_tolerance(293, 4, data["stat_p90"]))
    assert (within_tolerance(0.743, 0.005, data["stat_cov"]))
    assert (within_tolerance(1.12e+10, 7e+08, data["stat_energy"]))
    assert (within_tolerance(188, 4, data["stat_iqr"]))
    assert (within_tolerance(525, 1, data["stat_max"]))
    assert (within_tolerance(142, 3, data["stat_mean"]))
    assert (within_tolerance(92.4, 1.4, data["stat_mad"]))
    assert (within_tolerance(113, 4, data["stat_median"]))
    assert (within_tolerance(90.8, 1.6, data["stat_medad"]))
    assert (within_tolerance(28.5, 0.1, data["stat_min"]))
    assert (within_tolerance(0.699, 0.003, data["stat_qcod"]))
    assert (within_tolerance(496, 1, data["stat_range"]))
    assert (within_tolerance(75.9, 1.4, data["stat_rmad"]))
    assert (within_tolerance(177, 3, data["stat_rms"]))
    assert (within_tolerance(0.645, 0.028, data["stat_skew"]))
    assert (within_tolerance(11100, 300, data["stat_var"]))


def test_ibsi_2_config_gabor_filter():
    """
    Compare computed feature values with reference values for configurations 5A and 5B of IBSI 2 phase 2.
    """

    # Configuration 5.A ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="gabor",
        gabor_sigma=5.0,
        gabor_gamma=1.5,
        gabor_lambda=2.0,
        gabor_theta=0.0,
        gabor_theta_step=22.5,
        gabor_response="modulus",
        gabor_rotation_invariance=True,
        gabor_pooling_method="mean"
    )

    data = _process_experiment(
        configuration_id="5.A",
        by_slice=True,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("gabor_s_5.0_g_1.5_l_2.0_2D_", "") for column_name in data.columns]

    assert (within_tolerance(2.84, 0.14, data["stat_kurt"]))
    assert (within_tolerance(68.8, 0.9, data["stat_p10"]))
    assert (within_tolerance(145, 2, data["stat_p90"]))
    assert (within_tolerance(0.313, 0.003, data["stat_cov"]))
    assert (within_tolerance(1.46e+09, 2e+07, data["stat_energy"]))
    assert (within_tolerance(37.2, 0.6, data["stat_iqr"]))
    assert (within_tolerance(354, 5, data["stat_max"]))
    assert (within_tolerance(103, 1, data["stat_mean"]))
    assert (within_tolerance(24.2, 0.4, data["stat_mad"]))
    assert (within_tolerance(97, 1.3, data["stat_median"]))
    assert (within_tolerance(23.7, 0.4, data["stat_medad"]))
    assert (within_tolerance(31.4, 0.5, data["stat_min"]))
    assert (within_tolerance(0.187, 0.001, data["stat_qcod"]))
    assert (within_tolerance(322, 5, data["stat_range"]))
    assert (within_tolerance(15.8, 0.2, data["stat_rmad"]))
    assert (within_tolerance(108, 2, data["stat_rms"]))
    assert (within_tolerance(1.31, 0.03, data["stat_skew"]))
    assert (within_tolerance(1040, 10, data["stat_var"]))

    # Configuration 5.B ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="gabor",
        gabor_sigma=5.0,
        gabor_gamma=1.5,
        gabor_lambda=2.0,
        gabor_theta=0.0,
        gabor_theta_step=22.5,
        gabor_response="modulus",
        gabor_rotation_invariance=True,
        gabor_pooling_method="mean"
    )

    data = _process_experiment(
        configuration_id="5.B",
        by_slice=False,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("gabor_s_5.0_g_1.5_l_2.0_3D_invar_", "") for column_name in data.columns]

    assert (within_tolerance(4.34, 0.2, data["stat_kurt"]))
    assert (within_tolerance(24.6, 0.1, data["stat_p10"]))
    assert (within_tolerance(59.3, 0.3, data["stat_p90"]))
    assert (within_tolerance(0.377, 0.004, data["stat_cov"]))
    assert (within_tolerance(6.62e+08, 9e+06, data["stat_energy"]))
    assert (within_tolerance(17.4, 0.1, data["stat_iqr"]))
    assert (within_tolerance(175, 3, data["stat_max"]))
    assert (within_tolerance(40.2, 0.2, data["stat_mean"]))
    assert (within_tolerance(11.3, 0.1, data["stat_mad"]))
    assert (within_tolerance(37.2, 0.1, data["stat_median"]))
    assert (within_tolerance(11, 0.1, data["stat_medad"]))
    assert (within_tolerance(9.53, 0.11, data["stat_min"]))
    assert (within_tolerance(0.226, 0.002, data["stat_qcod"]))
    assert (within_tolerance(165, 3, data["stat_range"]))
    assert (within_tolerance(7.31, 0.06, data["stat_rmad"]))
    assert (within_tolerance(43, 0.2, data["stat_rms"]))
    assert (within_tolerance(1.57, 0.03, data["stat_skew"]))
    assert (within_tolerance(231, 2, data["stat_var"]))


def test_ibsi_2_config_daubechies_filter():
    """
    Compare computed feature values with reference values for configurations 6A, 6B, 7A and 7B of IBSI 2 phase 2.
    """

    # Configuration 6.A ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="separable_wavelet",
        separable_wavelet_families="db3",
        separable_wavelet_set="LH",
        separable_wavelet_rotation_invariance=True,
        separable_wavelet_decomposition_level=1,
        separable_wavelet_pooling_method="mean"
    )

    data = _process_experiment(
        configuration_id="6.A",
        by_slice=True,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("wavelet_db3_lh_level_1_invar_", "") for column_name in data.columns]

    assert (within_tolerance(7.72, 0.24, data["stat_kurt"]))
    assert (within_tolerance(-18.8, 0.4, data["stat_p10"]))
    assert (within_tolerance(17.7, 0.4, data["stat_p90"]))
    assert (within_tolerance(-112, 37, data["stat_cov"]))
    assert (within_tolerance(53500000, 2e+06, data["stat_energy"]))
    assert (within_tolerance(15.8, 0.1, data["stat_iqr"]))
    assert (within_tolerance(202, 2, data["stat_max"]))
    assert (within_tolerance(-0.185, 0.025, data["stat_mean"]))
    assert (within_tolerance(13, 0.2, data["stat_mad"]))
    assert (within_tolerance(0.0456, 0.0041, data["stat_median"]))
    assert (within_tolerance(13, 0.2, data["stat_medad"]))
    assert (within_tolerance(-245, 2, data["stat_min"]))
    assert (within_tolerance(491, 10, data["stat_qcod"]))
    assert (within_tolerance(447, 3, data["stat_range"]))
    assert (within_tolerance(6.82, 0.08, data["stat_rmad"]))
    assert (within_tolerance(20.7, 0.3, data["stat_rms"]))
    assert (within_tolerance(0.0837, 0.0188, data["stat_skew"]))
    assert (within_tolerance(427, 11, data["stat_var"]))

    # Configuration 6.B ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="separable_wavelet",
        separable_wavelet_families="db3",
        separable_wavelet_set="LLH",
        separable_wavelet_rotation_invariance=True,
        separable_wavelet_decomposition_level=1,
        separable_wavelet_pooling_method="mean"
    )

    data = _process_experiment(
        configuration_id="6.B",
        by_slice=False,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("wavelet_db3_llh_level_1_invar_", "") for column_name in data.columns]

    assert (within_tolerance(8.98, 0.35, data["stat_kurt"]))
    assert (within_tolerance(-13.8, 0.5, data["stat_p10"]))
    assert (within_tolerance(12.1, 0.4, data["stat_p90"]))
    assert (within_tolerance(-86.9, 32.6, data["stat_cov"]))
    assert (within_tolerance(89600000, 5300000, data["stat_energy"]))
    assert (within_tolerance(9.35, 0.15, data["stat_iqr"]))
    assert (within_tolerance(155, 1, data["stat_max"]))
    assert (within_tolerance(-0.182, 0.024, data["stat_mean"]))
    assert (within_tolerance(9.26, 0.22, data["stat_mad"]))
    assert (within_tolerance(0.0575, 0.0046, data["stat_median"]))
    assert (within_tolerance(9.25, 0.22, data["stat_medad"]))
    assert (within_tolerance(-148, 1, data["stat_min"]))
    assert (within_tolerance(-162, 27, data["stat_qcod"]))
    assert (within_tolerance(303, 2, data["stat_range"]))
    assert (within_tolerance(4.21, 0.09, data["stat_rmad"]))
    assert (within_tolerance(15.8, 0.3, data["stat_rms"]))
    assert (within_tolerance(0.157, 0.018, data["stat_skew"]))
    assert (within_tolerance(250, 9, data["stat_var"]))

    # Configuration 7.A ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="separable_wavelet",
        separable_wavelet_families="db3",
        separable_wavelet_set="HH",
        separable_wavelet_rotation_invariance=True,
        separable_wavelet_decomposition_level=2,
        separable_wavelet_pooling_method="mean"
    )

    data = _process_experiment(
        configuration_id="7.A",
        by_slice=True,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("wavelet_db3_hh_level_2_invar_", "") for column_name in data.columns]

    assert (within_tolerance(6.22, 0.16, data["stat_kurt"]))
    assert (within_tolerance(-39.6, 1, data["stat_p10"]))
    assert (within_tolerance(40.5, 0.9, data["stat_p90"]))
    assert (within_tolerance(178, 77, data["stat_cov"]))
    assert (within_tolerance(2.39e+08, 9e+06, data["stat_energy"]))
    assert (within_tolerance(28.8, 0.4, data["stat_iqr"]))
    assert (within_tolerance(352, 4, data["stat_max"]))
    assert (within_tolerance(0.245, 0.03, data["stat_mean"]))
    assert (within_tolerance(26.9, 0.4, data["stat_mad"]))
    assert (within_tolerance(0.0675, 0.0061, data["stat_median"]))
    assert (within_tolerance(26.9, 0.4, data["stat_medad"]))
    assert (within_tolerance(-349, 7, data["stat_min"]))
    assert (within_tolerance(104, 66, data["stat_qcod"]))
    assert (within_tolerance(701, 9, data["stat_range"]))
    assert (within_tolerance(13, 0.2, data["stat_rmad"]))
    assert (within_tolerance(43.7, 0.6, data["stat_rms"]))
    assert (within_tolerance(0.0469, 0.0072, data["stat_skew"]))
    assert (within_tolerance(1910, 40, data["stat_var"]))

    # Configuration 7.B ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="separable_wavelet",
        separable_wavelet_families="db3",
        separable_wavelet_set="HHH",
        separable_wavelet_rotation_invariance=True,
        separable_wavelet_decomposition_level=2,
        separable_wavelet_pooling_method="mean"
    )

    data = _process_experiment(
        configuration_id="7.B",
        by_slice=False,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("wavelet_db3_hhh_level_2_invar_", "") for column_name in data.columns]

    assert (within_tolerance(5.45, 0.09, data["stat_kurt"]))
    assert (within_tolerance(-20.6, 0.4, data["stat_p10"]))
    assert (within_tolerance(20.4, 0.4, data["stat_p90"]))
    assert (within_tolerance(-506, 149, data["stat_cov"]))
    assert (within_tolerance(1.51e+08, 7e+06, data["stat_energy"]))
    assert (within_tolerance(16.3, 0.2, data["stat_iqr"]))
    assert (within_tolerance(201, 4, data["stat_max"]))
    assert (within_tolerance(-0.0406, 0.0051, data["stat_mean"]))
    assert (within_tolerance(13.4, 0.2, data["stat_mad"]))
    assert (within_tolerance(-0.0164, 0.0013, data["stat_median"]))
    assert (within_tolerance(13.4, 0.2, data["stat_medad"]))
    assert (within_tolerance(-203, 3, data["stat_min"]))
    assert (within_tolerance(-684, 130, data["stat_qcod"]))
    assert (within_tolerance(404, 7, data["stat_range"]))
    assert (within_tolerance(7.2, 0.1, data["stat_rmad"]))
    assert (within_tolerance(20.6, 0.3, data["stat_rms"]))
    assert (within_tolerance(-0.0112, 0.0027, data["stat_skew"]))
    assert (within_tolerance(422, 11, data["stat_var"]))


def test_ibsi_2_config_simoncelli_filter():
    """
    Compare computed feature values with reference values for configurations 8A, 8B, 9A and 9B of IBSI 2 phase 2.
    """

    # Configuration 8.A ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="nonseparable_wavelet",
        nonseparable_wavelet_families="simoncelli",
        nonseparable_wavelet_decomposition_level=1,
    )

    data = _process_experiment(
        configuration_id="8.A",
        by_slice=True,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("wavelet_simoncelli_level_1_", "") for column_name in data.columns]

    assert (within_tolerance(8.19, 0.3, data["stat_kurt"]))
    assert (within_tolerance(-33.5, 0.9, data["stat_p10"]))
    assert (within_tolerance(34.4, 1, data["stat_p90"]))
    assert (within_tolerance(159, 30, data["stat_cov"]))
    assert (within_tolerance(1.94e+08, 9e+06, data["stat_energy"]))
    assert (within_tolerance(24.8, 0.3, data["stat_iqr"]))
    assert (within_tolerance(408, 3, data["stat_max"]))
    assert (within_tolerance(0.248, 0.047, data["stat_mean"]))
    assert (within_tolerance(23.6, 0.5, data["stat_mad"]))
    assert (within_tolerance(-0.0323, 0.0073, data["stat_median"]))
    assert (within_tolerance(23.6, 0.5, data["stat_medad"]))
    assert (within_tolerance(-395, 3, data["stat_min"]))
    assert (within_tolerance(-441, 29, data["stat_qcod"]))
    assert (within_tolerance(803, 5, data["stat_range"]))
    assert (within_tolerance(11.1, 0.2, data["stat_rmad"]))
    assert (within_tolerance(39.3, 0.7, data["stat_rms"]))
    assert (within_tolerance(-0.0473, 0.0145, data["stat_skew"]))
    assert (within_tolerance(1550, 50, data["stat_var"]))

    # Configuration 8.B ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="nonseparable_wavelet",
        nonseparable_wavelet_families="simoncelli",
        nonseparable_wavelet_decomposition_level=1,
    )

    data = _process_experiment(
        configuration_id="8.B",
        by_slice=False,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("wavelet_simoncelli_level_1_", "") for column_name in data.columns]

    assert (within_tolerance(7.64, 0.33, data["stat_kurt"]))
    assert (within_tolerance(-36.5, 1.3, data["stat_p10"]))
    assert (within_tolerance(38.1, 1.3, data["stat_p90"]))
    assert (within_tolerance(134, 27, data["stat_cov"]))
    assert (within_tolerance(6.48e+08, 3.9e+07, data["stat_energy"]))
    assert (within_tolerance(25.5, 0.4, data["stat_iqr"]))
    assert (within_tolerance(374, 3, data["stat_max"]))
    assert (within_tolerance(0.32, 0.059, data["stat_mean"]))
    assert (within_tolerance(25.3, 0.6, data["stat_mad"]))
    assert (within_tolerance(-0.00947, 0.0107, data["stat_median"]))
    assert (within_tolerance(25.3, 0.6, data["stat_medad"]))
    assert (within_tolerance(-411, 5, data["stat_min"]))
    assert (within_tolerance(785, 6, data["stat_range"]))
    assert (within_tolerance(11.7, 0.3, data["stat_rmad"]))
    assert (within_tolerance(42.5, 0.9, data["stat_rms"]))
    assert (within_tolerance(-0.0719, 0.0163, data["stat_skew"]))
    assert (within_tolerance(1810, 70, data["stat_var"]))

    # Configuration 9.A ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="nonseparable_wavelet",
        nonseparable_wavelet_families="simoncelli",
        nonseparable_wavelet_decomposition_level=2,
    )

    data = _process_experiment(
        configuration_id="9.A",
        by_slice=True,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("wavelet_simoncelli_level_2_", "") for column_name in data.columns]

    assert (within_tolerance(6.11, 0.21, data["stat_kurt"]))
    assert (within_tolerance(-59.2, 1.8, data["stat_p10"]))
    assert (within_tolerance(71.5, 1.9, data["stat_p90"]))
    assert (within_tolerance(32.6, 24.4, data["stat_cov"]))
    assert (within_tolerance(5.74e+08, 3.1e+07, data["stat_energy"]))
    assert (within_tolerance(34.2, 0.8, data["stat_iqr"]))
    assert (within_tolerance(470, 6, data["stat_max"]))
    assert (within_tolerance(2.08, 0.17, data["stat_mean"]))
    assert (within_tolerance(40.1, 0.9, data["stat_mad"]))
    assert (within_tolerance(0.14, 0.028, data["stat_median"]))
    assert (within_tolerance(40, 0.9, data["stat_medad"]))
    assert (within_tolerance(-535, 1, data["stat_min"]))
    assert (within_tolerance(47, 15.2, data["stat_qcod"]))
    assert (within_tolerance(1000, 10, data["stat_range"]))
    assert (within_tolerance(17.6, 0.5, data["stat_rmad"]))
    assert (within_tolerance(67.7, 1.3, data["stat_rms"]))
    assert (within_tolerance(-0.0596, 0.0145, data["stat_skew"]))
    assert (within_tolerance(4580, 170, data["stat_var"]))

    # Configuration 9.B ------------------------------------------------------------------------------------------------
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="nonseparable_wavelet",
        nonseparable_wavelet_families="simoncelli",
        nonseparable_wavelet_decomposition_level=2,
    )

    data = _process_experiment(
        configuration_id="9.B",
        by_slice=False,
        image_transformation_settings=image_transformation_settings
    )

    data.columns = [column_name.replace("wavelet_simoncelli_level_2_", "") for column_name in data.columns]

    assert (within_tolerance(5.58, 0.18, data["stat_kurt"]))
    assert (within_tolerance(-65.9, 2.2, data["stat_p10"]))
    assert (within_tolerance(82.8, 1.8, data["stat_p90"]))
    assert (within_tolerance(27.7, 20.4, data["stat_cov"]))
    assert (within_tolerance(1.97e+09, 1.4e+08, data["stat_energy"]))
    assert (within_tolerance(41, 1, data["stat_iqr"]))
    assert (within_tolerance(471, 13, data["stat_max"]))
    assert (within_tolerance(2.68, 0.22, data["stat_mean"]))
    assert (within_tolerance(45.1, 1.1, data["stat_mad"]))
    assert (within_tolerance(0.233, 0.046, data["stat_median"]))
    assert (within_tolerance(45, 1.1, data["stat_medad"]))
    assert (within_tolerance(-605, 2, data["stat_min"]))
    assert (within_tolerance(47.4, 20.7, data["stat_qcod"]))
    assert (within_tolerance(1080, 20, data["stat_range"]))
    assert (within_tolerance(21, 0.5, data["stat_rmad"]))
    assert (within_tolerance(74.1, 1.6, data["stat_rms"]))
    assert (within_tolerance(-0.0858, 0.0107, data["stat_skew"]))
    assert (within_tolerance(5490, 220, data["stat_var"]))
