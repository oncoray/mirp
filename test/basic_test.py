# The aim of this test-set is to test basic functionality in the presence of edge-cases.

import os
import pytest

import numpy as np
import pandas as pd

from mirp.experimentClass import ExperimentClass
from mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass,\
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass,\
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REMOVE_TEMP_RESPONSE_MAPS = False


def test_orientation():
    """
    Test internal representation of image objects using the orientation phantom.
    """

    from mirp.imageRead import load_image
    image_object, roi_list = load_image(
        image_folder=os.path.join(CURRENT_DIR, "data", "misc_images", "orientation", "image"),
        modality="CT",
        roi_folder=os.path.join(CURRENT_DIR, "data", "misc_images", "orientation", "mask"),
        roi_names="mask")

    # Assert minimum and maximum values in the voxel grid.
    assert np.min(image_object.get_voxel_grid()) == 0.0
    assert np.max(image_object.get_voxel_grid()) == 141.0

    # Check dimensions. MIRP expects a (z, y, x) orientation.
    assert np.array_equal(image_object.size, np.array([64, 48, 32]))

    # Check orientation. The minimum value should be in the origin, and the maximum value in the most distal voxel.
    assert image_object.get_voxel_grid()[0, 0, 0] == 0.0
    assert image_object.get_voxel_grid()[-1, -1, -1] == 141.0

    # Check if origin and spacing match initial values.
    assert np.array_equal(image_object.origin, np.array([0.0, 1.0, 2.0]))
    assert np.array_equal(image_object.spacing, np.array([0.5, 1.0, 1.5]))


def _setup_experiment(image, roi):
    by_slice = False

    # Configure settings.
    general_settings = GeneralSettingsClass(
        by_slice=by_slice
    )

    image_interpolation_settings = ImageInterpolationSettingsClass(
        by_slice=by_slice,
        new_spacing=1.0
    )

    # Test all the things!
    feature_computation_parameters = FeatureExtractionSettingsClass(
        by_slice=general_settings.by_slice,
        no_approximation=True,
        base_discretisation_method=["fixed_bin_number", "fixed_bin_size"],
        base_discretisation_n_bins=12,
        base_discretisation_bin_width=25.0,
        ivh_discretisation_method="fixed_bin_number",
        glcm_distance=[1.0],
        glcm_spatial_method=["2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge",
                             "3d_average", "3d_volume_merge"],
        glrlm_spatial_method=["2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge",
                              "3d_average", "3d_volume_merge"],
        glszm_spatial_method=["2d", "2.5d", "3d"],
        gldzm_spatial_method=["2d", "2.5d", "3d"],
        ngtdm_spatial_method=["2d", "2.5d", "3d"],
        ngldm_distance=[1.0],
        ngldm_spatial_method=["2d", "2.5d", "3d"],
        ngldm_difference_level=[0.0]
    )

    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=general_settings.by_slice,
        response_map_feature_settings=None
    )

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

    experiment = ExperimentClass(
        modality="CT",
        subject="_".join([image, roi]),
        cohort=None,
        write_path=None,
        image_folder=os.path.join(CURRENT_DIR, "data", "misc_images", image, "image"),
        roi_folder=os.path.join(CURRENT_DIR, "data", "misc_images", image, "mask"),
        roi_reg_img_folder=None,
        image_file_name_pattern=None,
        registration_image_file_name_pattern=None,
        roi_names=[roi],
        data_str=None,
        provide_diagnostics=False,
        settings=settings,
        compute_features=True,
        extract_images=False,
        plot_images=False,
        keep_images_in_memory=False
    )

    return experiment


def test_edge_cases_basic_pipeline():
    """
    Test feature extraction using the basic pipeline. The following cases are tested using an uninformative phantom
    that has the value 1 everywhere:
    -   using a normal mask that completely covers the image. This is to test how MIRP responds to uninformative images.
    -   using a mask that only contains a single voxel. This is to test how MIRP responds to masks with a single voxel.
    -   using a mask that has disconnected voxels.
    -   using an empty mask.
    -   using a mask that becomes empty after resegmentation.

    Both 3D and 2D (slice) phantoms are used.

    So in short, this pipeline tests the worst possible images and masks to figure out what happens.
    """

    images = ["uninformative", "uninformative_slice"]
    rois = ["full_mask", "one_voxel_mask", "disconnected_mask", "empty_mask"]

    for image in images:
        for roi in rois:
            # Setup experiment.
            experiment = _setup_experiment(image=image, roi=roi)

            # Test
            if roi == "empty_mask":
                with pytest.raises(ValueError) as excinfo:
                    experiment.process()

                assert "is not a mask consisting of 0s and 1s" in str(excinfo.value)

            else:
                data = experiment.process()
                assert isinstance(data, pd.DataFrame)
