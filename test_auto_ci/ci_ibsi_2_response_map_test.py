import warnings

import itk
import numpy as np
import os

from mirp.extract_features_and_images import extract_images
from mirp.settings.generic import SettingsClass
from mirp.settings.transformation_parameters import ImageTransformationSettingsClass
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
from mirp.settings.resegmentation_parameters import ResegmentationSettingsClass
from mirp.settings.perturbation_parameters import ImagePerturbationSettingsClass
from mirp.settings.image_processing_parameters import ImagePostProcessingClass
from mirp.settings.interpolation_parameters import ImageInterpolationSettingsClass, MaskInterpolationSettingsClass
from mirp.settings.general_parameters import GeneralSettingsClass

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test")


def _get_default_settings(by_slice: bool = False):
    """Set default settings for response map tests."""

    general_settings = GeneralSettingsClass(
        by_slice=by_slice
    )

    image_interpolation_settings = ImageInterpolationSettingsClass(
        by_slice=by_slice,
        anti_aliasing=False
    )

    feature_computation_parameters = FeatureExtractionSettingsClass(
        by_slice=by_slice,
        no_approximation=True,
        base_feature_families="none"
    )

    return general_settings, image_interpolation_settings, feature_computation_parameters


def _setup_experiment(
        by_slice: bool,
        phantom: str,
        image_transformation_settings: ImageTransformationSettingsClass):

    # Get default settings.
    general_settings, image_interpolation_settings, feature_computation_parameters = _get_default_settings(by_slice=by_slice)

    settings = SettingsClass(
        general_settings=general_settings,
        post_process_settings=ImagePostProcessingClass(),
        img_interpolate_settings=image_interpolation_settings,
        roi_interpolate_settings=MaskInterpolationSettingsClass(),
        roi_resegment_settings=ResegmentationSettingsClass(),
        perturbation_settings=ImagePerturbationSettingsClass(),
        img_transform_settings=image_transformation_settings,
        feature_extr_settings=feature_computation_parameters
    )

    images = extract_images(
        write_images=False,
        export_images=True,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_2_digital_phantom", phantom, "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_2_digital_phantom", phantom, "mask"),
        settings=settings
    )

    # Return the transformed image.
    images, mask = images[0]

    return images[1]


def _test_filter_configuration(
        configuration_id: str,
        phantom: str,
        image_transformation_settings: ImageTransformationSettingsClass,
        filter_kernel=None):

    reference_dir = os.path.join(CURRENT_DIR, "data", "ibsi_2_reference_response_maps")

    # Retrieve filter kernel.
    if filter_kernel is None:
        filter_kernel = image_transformation_settings.spatial_filters[0]

    image = _setup_experiment(
        by_slice=image_transformation_settings.by_slice,
        phantom=phantom,
        image_transformation_settings=image_transformation_settings)

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

    assert(np.allclose(image["image"], reference_response_map_voxels, atol=0.001))


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
    settings_list = [
        image_transformation_settings_1a1,
        image_transformation_settings_1a2,
        image_transformation_settings_1a3,
        image_transformation_settings_1a4]

    # Iterate over configurations.
    for ii, image_transformation_settings in enumerate(settings_list):
        _test_filter_configuration(
            configuration_id=configuration_ids[ii],
            phantom="checkerboard",
            image_transformation_settings=image_transformation_settings
        )

    # Test 1B1
    image_transformation_settings_1b1 = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        mean_filter_kernel_size=15,
        mean_filter_boundary_condition="constant"
    )

    _test_filter_configuration(
        configuration_id="1B1",
        phantom="impulse",
        image_transformation_settings=image_transformation_settings_1b1
    )


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

    _test_filter_configuration(
        configuration_id="2A",
        phantom="impulse",
        image_transformation_settings=image_transformation_settings_2a,
        filter_kernel="log"
    )

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

    _test_filter_configuration(
        configuration_id="2B",
        phantom="checkerboard",
        image_transformation_settings=image_transformation_settings_2b,
        filter_kernel="log"
    )

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

    _test_filter_configuration(
        configuration_id="2C",
        phantom="checkerboard",
        image_transformation_settings=image_transformation_settings_2c,
        filter_kernel="log"
    )


def test_ibsi_2_laws_filter():
    """
    Configuration 3: Response maps for the Laws filter.
    """

    # Set configuration identifiers.
    filter_kernel = "laws"

    # Test 3A1
    image_transformation_settings_3a1 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        laws_kernel="E5L5S5",
        laws_compute_energy=False,
        laws_rotation_invariance=False,
        laws_boundary_condition="constant"
    )

    # Test 3A2
    image_transformation_settings_3a2 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        laws_kernel="E5L5S5",
        laws_compute_energy=False,
        laws_rotation_invariance=True,
        laws_pooling_method="max",
        laws_boundary_condition="constant"
    )

    # Test 3A3
    image_transformation_settings_3a3 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        laws_kernel="E5L5S5",
        laws_compute_energy=True,
        laws_rotation_invariance=True,
        laws_delta=7,
        laws_pooling_method="max",
        laws_boundary_condition="constant"
    )

    # Add to settings.
    settings_list = [image_transformation_settings_3a1, image_transformation_settings_3a2,
                     image_transformation_settings_3a3]

    # Set configuration identifiers.
    configuration_ids = ["3A1", "3A2", "3A3"]

    # Iterate over configurations.
    for ii, image_transformation_settings in enumerate(settings_list):
        _test_filter_configuration(
            configuration_id=configuration_ids[ii],
            phantom="impulse",
            image_transformation_settings=image_transformation_settings
        )

    # Test 3B1
    image_transformation_settings_3b1 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        laws_kernel="E3W5R5",
        laws_compute_energy=False,
        laws_rotation_invariance=False,
        laws_boundary_condition="reflect"
    )

    # Test 3B2
    image_transformation_settings_3b2 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        laws_kernel="E3W5R5",
        laws_compute_energy=False,
        laws_rotation_invariance=True,
        laws_pooling_method="max",
        laws_boundary_condition="reflect"
    )

    # Test 3B3
    image_transformation_settings_3b3 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        laws_kernel="E3W5R5",
        laws_compute_energy=True,
        laws_rotation_invariance=True,
        laws_delta=7,
        laws_pooling_method="max",
        laws_boundary_condition="reflect"
    )

    # Test 3C1
    image_transformation_settings_3c1 = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        laws_kernel="L5S5",
        laws_compute_energy=False,
        laws_rotation_invariance=False,
        laws_boundary_condition="reflect"
    )

    # Test 3C2
    image_transformation_settings_3c2 = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        laws_kernel="L5S5",
        laws_compute_energy=False,
        laws_rotation_invariance=True,
        laws_pooling_method="max",
        laws_boundary_condition="reflect"
    )

    # Test 3C3
    image_transformation_settings_3c3 = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        laws_kernel="L5S5",
        laws_compute_energy=True,
        laws_rotation_invariance=True,
        laws_delta=7,
        laws_pooling_method="max",
        laws_boundary_condition="reflect"
    )

    # Add to settings.
    settings_list = [
        image_transformation_settings_3b1, image_transformation_settings_3b2,
        image_transformation_settings_3b3, image_transformation_settings_3c1,
        image_transformation_settings_3c2, image_transformation_settings_3c3
    ]

    # Set configuration identifiers.
    configuration_ids = ["3B1", "3B2", "3B3", "3C1", "3C2", "3C3"]

    # Iterate over configurations.
    for ii, image_transformation_settings in enumerate(settings_list):
        _test_filter_configuration(
            configuration_id=configuration_ids[ii],
            phantom="checkerboard",
            image_transformation_settings=image_transformation_settings
        )


def test_ibsi_2_gabor_filter():
    """
    Configuration 4: Response maps for the Gabor filter.
    """

    # Set configuration identifiers.
    filter_kernel = "gabor"

    # Test 4A1
    image_transformation_settings_4a1 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        gabor_sigma=10.0,
        gabor_gamma=0.5,
        gabor_lambda=4.0,
        gabor_theta=60.0,
        gabor_response="modulus",
        gabor_rotation_invariance=False,
        gabor_boundary_condition="constant"
    )

    # Test 4A2
    image_transformation_settings_4a2 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        gabor_sigma=10.0,
        gabor_gamma=0.5,
        gabor_lambda=4.0,
        gabor_theta=0.0,
        gabor_theta_step=45.0,
        gabor_response="modulus",
        gabor_rotation_invariance=True,
        gabor_pooling_method="mean",
        gabor_boundary_condition="constant"
    )

    # Add to settings.
    settings_list = [image_transformation_settings_4a1, image_transformation_settings_4a2]

    # Set configuration identifiers.
    configuration_ids = ["4A1", "4A2"]

    # Iterate over configurations.
    for ii, image_transformation_settings in enumerate(settings_list):
        _test_filter_configuration(
            configuration_id=configuration_ids[ii],
            phantom="impulse",
            image_transformation_settings=image_transformation_settings
        )

    # Test 4B1
    image_transformation_settings_4b1 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        gabor_sigma=20.0,
        gabor_gamma=2.5,
        gabor_lambda=8.0,
        gabor_theta=225.0,
        gabor_response="modulus",
        gabor_rotation_invariance=False,
        gabor_boundary_condition="reflect"
    )

    # Test 4B2
    image_transformation_settings_4b2 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        gabor_sigma=20.0,
        gabor_gamma=2.5,
        gabor_lambda=8.0,
        gabor_theta=0.0,
        gabor_theta_step=22.5,
        gabor_response="modulus",
        gabor_rotation_invariance=True,
        gabor_pooling_method="mean",
        gabor_boundary_condition="reflect"
    )

    # Add to settings.
    settings_list = [image_transformation_settings_4b1, image_transformation_settings_4b2]

    # Set configuration identifiers.
    configuration_ids = ["4B1", "4B2"]

    # Iterate over configurations.
    for ii, image_transformation_settings in enumerate(settings_list):
        _test_filter_configuration(
            configuration_id=configuration_ids[ii],
            phantom="sphere",
            image_transformation_settings=image_transformation_settings
        )


def test_ibsi_2_daubechies_filter():
    """
    Configuration 5: Response maps for the Daubechies 2 separable wavelet filter.
    """

    # Set configuration identifiers.
    filter_kernel = "separable_wavelet"

    # Test 5A1
    image_transformation_settings_5a1 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        separable_wavelet_families="db2",
        separable_wavelet_set="LHL",
        separable_wavelet_rotation_invariance=False,
        separable_wavelet_boundary_condition="constant"
    )

    # Test 5A2
    image_transformation_settings_5a2 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        separable_wavelet_families="db2",
        separable_wavelet_set="LHL",
        separable_wavelet_rotation_invariance=True,
        separable_wavelet_pooling_method="mean",
        separable_wavelet_boundary_condition="constant"
    )

    # Add to settings.
    settings_list = [image_transformation_settings_5a1, image_transformation_settings_5a2]

    # Set configuration identifiers.
    configuration_ids = ["5A1", "5A2"]

    # Iterate over configurations.
    for ii, image_transformation_settings in enumerate(settings_list):
        _test_filter_configuration(
            configuration_id=configuration_ids[ii],
            phantom="impulse",
            image_transformation_settings=image_transformation_settings,
            filter_kernel="wavelet_db2"
        )


def test_ibsi_2_coifflet_filter():
    """
    Configuration 6: Response maps for the Coifflet 1 separable wavelet filter.
    """

    # Set configuration identifiers.
    filter_kernel = "separable_wavelet"

    # Test 6A1
    image_transformation_settings_6a1 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        separable_wavelet_families="coif1",
        separable_wavelet_set="HHL",
        separable_wavelet_rotation_invariance=False,
        separable_wavelet_boundary_condition="wrap"
    )

    # Test 6A2
    image_transformation_settings_6a2 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        separable_wavelet_families="coif1",
        separable_wavelet_set="HHL",
        separable_wavelet_rotation_invariance=True,
        separable_wavelet_pooling_method="mean",
        separable_wavelet_boundary_condition="wrap"
    )

    # Add to settings.
    settings_list = [image_transformation_settings_6a1, image_transformation_settings_6a2]

    # Set configuration identifiers.
    configuration_ids = ["6A1", "6A2"]

    # Iterate over configurations.
    for ii, image_transformation_settings in enumerate(settings_list):
        _test_filter_configuration(
            configuration_id=configuration_ids[ii],
            phantom="sphere",
            image_transformation_settings=image_transformation_settings,
            filter_kernel="wavelet_coif1"
        )


def test_ibsi_2_haar_filter():
    """
    Configuration 7: Response maps for the Haar separable wavelet filter.
    """

    # Set configuration identifiers.
    filter_kernel = "separable_wavelet"

    # Test 7A1
    image_transformation_settings_7a1 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        separable_wavelet_families="haar",
        separable_wavelet_set="LLL",
        separable_wavelet_rotation_invariance=True,
        separable_wavelet_pooling_method="mean",
        separable_wavelet_decomposition_level=2,
        separable_wavelet_boundary_condition="reflect"
    )

    # Test 7A2
    image_transformation_settings_7a2 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        separable_wavelet_families="haar",
        separable_wavelet_set="HHH",
        separable_wavelet_rotation_invariance=True,
        separable_wavelet_pooling_method="mean",
        separable_wavelet_decomposition_level=2,
        separable_wavelet_boundary_condition="reflect"
    )

    # Add to settings.
    settings_list = [image_transformation_settings_7a1, image_transformation_settings_7a2]

    # Set configuration identifiers.
    configuration_ids = ["7A1", "7A2"]

    # Iterate over configurations.
    for ii, image_transformation_settings in enumerate(settings_list):
        _test_filter_configuration(
            configuration_id=configuration_ids[ii],
            phantom="checkerboard",
            image_transformation_settings=image_transformation_settings,
            filter_kernel="wavelet_haar"
        )


def test_ibsi_2_simoncelli_filter():
    """
    Configuration 8: Response maps for the non-separable Simoncelli wavelet filter.
    """

    # Set configuration identifiers.
    filter_kernel = "nonseparable_wavelet"

    # Test 8A1
    image_transformation_settings_8a1 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        nonseparable_wavelet_families="simoncelli",
        nonseparable_wavelet_decomposition_level=1,
        nonseparable_wavelet_boundary_condition="wrap"
    )

    # Test 8A2
    image_transformation_settings_8a2 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        nonseparable_wavelet_families="simoncelli",
        nonseparable_wavelet_decomposition_level=2,
        nonseparable_wavelet_boundary_condition="wrap"
    )

    # Test 8A3
    image_transformation_settings_8a3 = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None,
        response_map_feature_families="none",
        filter_kernels=filter_kernel,
        nonseparable_wavelet_families="simoncelli",
        nonseparable_wavelet_decomposition_level=3,
        nonseparable_wavelet_boundary_condition="wrap"
    )

    # Add to settings.
    settings_list = [
        image_transformation_settings_8a1,
        image_transformation_settings_8a2,
        image_transformation_settings_8a3
    ]

    # Set configuration identifiers.
    configuration_ids = ["8A1", "8A2", "8A3"]

    # Iterate over configurations.
    for ii, image_transformation_settings in enumerate(settings_list):
        _test_filter_configuration(
            configuration_id=configuration_ids[ii],
            phantom="checkerboard",
            image_transformation_settings=image_transformation_settings,
            filter_kernel="wavelet_simoncelli"
        )
