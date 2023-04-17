import os

from mirp.experimentClass import ExperimentClass
from mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass, \
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass, \
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

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


def _get_default_settings(by_slice: bool,
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
            interpolate=False,
            anti_aliasing=False
        )

    else:
        image_interpolation_settings = ImageInterpolationSettingsClass(
            by_slice=by_slice,
            interpolate=True,
            spline_order=3,
            new_spacing=1.0,
            anti_aliasing=False
        )

    resegmentation_settings = ResegmentationSettingsClass(
        resegmentation_method="threshold",
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
        roi_interpolate_settings=RoiInterpolationSettingsClass(),
        roi_resegment_settings=resegmentation_settings,
        perturbation_settings=perturbation_settings,
        img_transform_settings=image_transformation_settings,
        feature_extr_settings=feature_computation_parameters
    )

    main_experiment = ExperimentClass(
        modality="CT",
        subject="phantom",
        cohort=None,
        write_path=None,
        image_folder=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        roi_folder=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_reg_img_folder=None,
        image_file_name_pattern=None,
        registration_image_file_name_pattern=None,
        roi_names=["GTV-1"],
        data_str=[configuration_id],
        provide_diagnostics=True,
        settings=settings,
        compute_features=True,
        extract_images=False,
        plot_images=False,
        keep_images_in_memory=False
    )

    data = main_experiment.process()

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

    data.columns = [column_name.replace("mean_d_5_", "") for column_name in data.columns.values]

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

    data.columns = [column_name.replace("mean_d_5_", "") for column_name in data.columns.values]

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

    # TODO assert stuff one IBSI 2 is done.

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

    # TODO assert stuff one IBSI 2 is done.


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

    # TODO assert stuff one IBSI 2 is done.

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

    # TODO assert stuff one IBSI 2 is done.


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

    # TODO assert stuff one IBSI 2 is done.

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

    # TODO assert stuff one IBSI 2 is done.


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

    # TODO assert stuff one IBSI 2 is done.

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

    # TODO assert stuff one IBSI 2 is done.

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

    # TODO assert stuff one IBSI 2 is done.

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

    # TODO assert stuff one IBSI 2 is done.


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

    # TODO assert stuff one IBSI 2 is done.

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

    # TODO assert stuff one IBSI 2 is done.

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

    # TODO assert stuff one IBSI 2 is done.

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

    # TODO assert stuff one IBSI 2 is done.