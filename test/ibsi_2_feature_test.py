import os

from mirp.experimentClass import ExperimentClass
from mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass, \
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass, \
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PERTURB_IMAGES = False
WRITE_TEMP_FILES = True


def _get_default_settings(by_slice: bool = False,
                          base_feature_families="none"):
    """Set default settings for response map tests."""

    general_settings = GeneralSettingsClass(
        by_slice=by_slice
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
        # TODO: add in perturbation settings from IBSI 1.
        perturbation_settings = ImagePerturbationSettingsClass(
            crop_around_roi=False
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


def _process_experiment(configuration_id: str,
                        by_slice: bool,
                        image_transformation_settings: ImageTransformationSettingsClass,
                        base_feature_families: str = "none"):

    # Set testing directory
    test_dir = os.path.join(CURRENT_DIR, "data", "temp")
    if not os.path.isdir(test_dir) and WRITE_TEMP_FILES:
        os.makedirs(test_dir)

    # Get default settings.
    general_settings, image_interpolation_settings, feature_computation_parameters, resegmentation_settings, \
        perturbation_settings = _get_default_settings(by_slice=by_slice,
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

    # Configure image transformation settings.
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels=None
    )

    data = _process_experiment(
        configuration_id="1A",
        by_slice=True,
        image_transformation_settings=image_transformation_settings,
        base_feature_families="statistics"
    )

    # TODO assert stuff one IBSI 2 is done.

    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels=None
    )

    data = _process_experiment(
        configuration_id="1B",
        by_slice=False,
        image_transformation_settings=image_transformation_settings,
        base_feature_families="statistics"
    )

    # TODO assert stuff one IBSI 2 is done.


def test_ibsi_2_config_mean():
    """
    Compare computed feature values with reference values for configurations 1A and 1B of IBSI 2 phase 2.
    """

    # Configure image transformation settings.
    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=True,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="mean",
        mean_filter_kernel_size=5
    )

    data = _process_experiment(
        configuration_id="2A",
        by_slice=True,
        image_transformation_settings=image_transformation_settings
    )

    # TODO assert stuff one IBSI 2 is done.

    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_families="statistics",
        response_map_feature_settings=None,
        boundary_condition="reflect",
        filter_kernels="mean",
        mean_filter_kernel_size=5
    )

    data = _process_experiment(
        configuration_id="2B",
        by_slice=False,
        image_transformation_settings=image_transformation_settings
    )

    # TODO assert stuff one IBSI 2 is done.