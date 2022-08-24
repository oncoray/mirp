import warnings

import itk
import numpy as np
import os
from shutil import rmtree

from mirp.experimentClass import ExperimentClass
from mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass,\
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass,\
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REMOVE_TEMP_RESPONSE_MAPS = False


def _get_default_settings(by_slice: bool = False):
    """Set default settings for response map tests."""

    general_settings = GeneralSettingsClass(
        by_slice=by_slice
    )

    image_interpolation_settings = ImageInterpolationSettingsClass(
        by_slice=by_slice,
        interpolate=False,
        anti_aliasing=False
    )

    feature_computation_parameters = FeatureExtractionSettingsClass(
        by_slice=by_slice,
        no_approximation=True,
        base_feature_families="none"
    )

    return general_settings, image_interpolation_settings, feature_computation_parameters


def _setup_experiment(configuration_id: str,
                      by_slice: bool,
                      phantom: str,
                      test_dir: str,
                      image_transformation_settings: ImageTransformationSettingsClass):

    # Check if the temporary directory still exists.
    if os.path.isdir(test_dir) and REMOVE_TEMP_RESPONSE_MAPS:
        rmtree(test_dir)

    # Get default settings.
    general_settings, image_interpolation_settings, feature_computation_parameters = _get_default_settings(by_slice=by_slice)

    settings = SettingsClass(
        general_settings=general_settings,
        post_process_settings=ImagePostProcessingClass(),
        img_interpolate_settings=image_interpolation_settings,
        roi_interpolate_settings=RoiInterpolationSettingsClass(),
        roi_resegment_settings=ResegmentationSettingsClass(),
        perturbation_settings=ImagePerturbationSettingsClass(),
        img_transform_settings=image_transformation_settings,
        feature_extr_settings=feature_computation_parameters
    )

    main_experiment = ExperimentClass(
        modality="CT",
        subject=phantom,
        cohort=None,
        write_path=test_dir,
        image_folder=os.path.join(CURRENT_DIR, "data", "ibsi_2_digital_phantom", phantom, "image"),
        roi_folder=os.path.join(CURRENT_DIR, "data", "ibsi_2_digital_phantom", phantom, "mask"),
        roi_reg_img_folder=None,
        image_file_name_pattern=None,
        registration_image_file_name_pattern=None,
        roi_names=["mask"],
        data_str=[configuration_id],
        provide_diagnostics=False,
        settings=settings,
        compute_features=False,
        extract_images=True,
        plot_images=False,
        keep_images_in_memory=False)

    return main_experiment


def _test_filter_configuration(configuration_id: str,
                               phantom: str,
                               image_transformation_settings: ImageTransformationSettingsClass,
                               filter_kernel=None):

    test_dir = os.path.join(CURRENT_DIR, "data", "temp")
    reference_dir = os.path.join(CURRENT_DIR, "data", "ibsi_2_reference_response_maps")

    # Retrieve filter kernel.
    if filter_kernel is None:
        filter_kernel = image_transformation_settings.spatial_filters[0]

    experiment = _setup_experiment(configuration_id=configuration_id,
                                   by_slice=image_transformation_settings.by_slice,
                                   phantom=phantom,
                                   test_dir=test_dir,
                                   image_transformation_settings=image_transformation_settings)

    # Generate data
    _ = experiment.process()

    # List files in temp_dir
    dir_files = os.listdir(test_dir)
    dir_files = [file for file in dir_files if configuration_id in file]
    dir_files = [file for file in dir_files if filter_kernel in file]
    if len(dir_files) > 1:
        raise ValueError("More than one viable test response map was found.")

    # Read generated test file
    test_response_map = itk.imread(filename=os.path.join(test_dir, dir_files[0]))
    test_response_map_voxels = itk.GetArrayFromImage(test_response_map)

    # Read reference file
    dir_files = os.listdir(reference_dir)
    dir_files = [file for file in dir_files if configuration_id in file]
    dir_files = [file for file in dir_files if filter_kernel in file]
    if len(dir_files) > 1:
        warnings.warn("More than one viable test response map was found.", UserWarning)
        return None

    elif len(dir_files) == 0:
        warnings.warn("No test response map was found.", UserWarning)
        return None

    reference_response_map = itk.imread(filename=os.path.join(reference_dir, dir_files[0]))
    reference_response_map_voxels = itk.GetArrayFromImage(reference_response_map)

    assert(np.allclose(test_response_map_voxels,
                       reference_response_map_voxels,
                       atol=0.001))


def test_ibsi_2_mean_filter():
    """
    Configuration 1: Response maps for the mean filter.
    """

    # Set configuration identifiers.
    configuration_ids = ["1A1", "1A2", "1A3", "1A4"]
    filter_kernel = "mean"

    # Test 1A1
    image_transformation_settings_1a1 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        mean_filter_kernel_size=15,
        mean_filter_boundary_condition="constant"
    )

    # Test 1A2
    image_transformation_settings_1a2 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        mean_filter_kernel_size=15,
        mean_filter_boundary_condition="nearest"
    )

    # Test 1A3
    image_transformation_settings_1a3 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        mean_filter_kernel_size=15,
        mean_filter_boundary_condition="wrap"
    )

    # Test 1A4
    image_transformation_settings_1a4 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        mean_filter_kernel_size=15,
        mean_filter_boundary_condition="reflect"
    )

    # Add to settings.
    settings_list = [image_transformation_settings_1a1, image_transformation_settings_1a2,
                     image_transformation_settings_1a3, image_transformation_settings_1a4]

    # Iterate over configurations.
    for ii, image_transformation_settings in enumerate(settings_list):
        _test_filter_configuration(configuration_id=configuration_ids[ii],
                                   phantom="checkerboard",
                                   image_transformation_settings=image_transformation_settings)

    # Test 1B1
    image_transformation_settings_1b1 = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        mean_filter_kernel_size=15,
        mean_filter_boundary_condition="constant"
    )

    _test_filter_configuration(configuration_id="1B1",
                               phantom="impulse",
                               image_transformation_settings=image_transformation_settings_1b1)


def test_ibsi_2_log_filter():
    """
    Configuration 2: Response maps for the Laplacian-of-Gaussian filter.
    """

    # Set configuration identifiers.
    filter_kernel = "laplacian_of_gaussian"

    # Test 2A
    image_transformation_settings_2a = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        laplacian_of_gaussian_sigma=3.0,
        laplacian_of_gaussian_kernel_truncate=4.0,
        laplacian_of_gaussian_boundary_condition="constant"
    )

    _test_filter_configuration(configuration_id="2A",
                               phantom="impulse",
                               image_transformation_settings=image_transformation_settings_2a,
                               filter_kernel="log")

    # Test 2B
    image_transformation_settings_2b = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        laplacian_of_gaussian_sigma=5.0,
        laplacian_of_gaussian_kernel_truncate=4.0,
        laplacian_of_gaussian_boundary_condition="reflect"
    )

    _test_filter_configuration(configuration_id="2B",
                               phantom="checkerboard",
                               image_transformation_settings=image_transformation_settings_2b,
                               filter_kernel="log")

    # Test 2C
    image_transformation_settings_2c = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        laplacian_of_gaussian_sigma=5.0,
        laplacian_of_gaussian_kernel_truncate=4.0,
        laplacian_of_gaussian_boundary_condition="reflect"
    )

    _test_filter_configuration(configuration_id="2C",
                               phantom="checkerboard",
                               image_transformation_settings=image_transformation_settings_2c,
                               filter_kernel="log")
