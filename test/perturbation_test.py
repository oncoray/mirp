import os

import numpy as np

from mirp.experimentClass import ExperimentClass
from mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass, \
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass, \
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

WRITE_TEMP_FILES = False


def test_noise_perturbation():

    perturbation_settings = ImagePerturbationSettingsClass(
        crop_around_roi=False,
        perturbation_noise_repetitions=1
    )

    # Set up experiment.
    experiment = generate_experiments(perturbation_settings=perturbation_settings)

    # Run computations.
    feature_table, img_obj, roi_list = experiment.process()

    # Assert that object location and origin have not changed.
    assert np.allclose(img_obj.origin, [-100.400, -79.626, -174.395])
    assert np.allclose(img_obj.orientation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(roi_list[0].roi.origin, [-100.400, -79.626, -174.395])
    assert np.allclose(roi_list[0].roi.orientation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    assert np.isclose(feature_table["morph_volume"][0], 358417.64)
    assert 40.0 < feature_table["stat_mean"][0] < 50.0
    assert ~np.isclose(feature_table["stat_mean"][0], 43.829533)


def test_translation_perturbation():
    perturbation_settings = ImagePerturbationSettingsClass(
        crop_around_roi=False,
        perturbation_translation_fraction=0.1
    )

    # Set up experiment.
    experiment = generate_experiments(perturbation_settings=perturbation_settings)

    # Run computations.
    feature_table, img_obj, roi_list = experiment.process()

    # Origin has changed.
    assert ~np.allclose(img_obj.origin, [-100.400, -79.626, -174.395])
    assert ~np.allclose(roi_list[0].roi.origin, [-100.400, -79.626, -174.395])
    assert np.allclose(img_obj.origin, [-100.100, -79.528, -174.297])
    assert np.allclose(roi_list[0].roi.origin, [-100.100, -79.528, -174.297])

    # Assert that object orientation did not change.
    assert np.allclose(img_obj.orientation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(roi_list[0].roi.orientation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Volume should not change that much.
    assert 358400.0 < feature_table["morph_volume"][0] < 358600.0

    # Mean value should change slightly.
    assert 40.0 < feature_table["stat_mean"][0] < 50.0
    assert ~np.isclose(feature_table["stat_mean"][0], 43.829533)


def test_rotation_perturbation():
    perturbation_settings = ImagePerturbationSettingsClass(
        crop_around_roi=False,
        perturbation_rotation_angles=45.0,
        # perturbation_roi_adapt_type="fraction",
        # perturbation_roi_adapt_size=0.2,
        # perturbation_randomise_roi_repetitions=1
    )

    # Set up experiment.
    experiment = generate_experiments(perturbation_settings=perturbation_settings)

    # Run computations.
    feature_table, img_obj, roi_list = experiment.process()

    # Origin has changed in x-y plane.
    assert ~np.allclose(img_obj.origin, [-100.400, -79.626, -174.395])
    assert ~np.allclose(roi_list[0].roi.origin, [-100.400, -79.626, -174.395])
    assert np.allclose(img_obj.origin, [-100.400, -121.130, -76.265])
    assert np.allclose(roi_list[0].roi.origin, [-100.400, -121.130, -76.265])

    # Orientation has changed.
    assert np.allclose(img_obj.orientation,
                       [[1.0, 0.0, 0.0],
                        [0.0, 1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)],
                        [0.0, -1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]])
    assert np.allclose(roi_list[0].roi.orientation,
                       [[1.0, 0.0, 0.0],
                        [0.0, 1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)],
                        [0.0, -1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]])

    # Volume should not change that much.
    assert 358300.0 < feature_table["morph_volume"][0] < 358600.0

    # Mean value should change slightly.
    assert 40.0 < feature_table["stat_mean"][0] < 50.0
    assert ~np.isclose(feature_table["stat_mean"][0], 43.829533)


def generate_experiments(perturbation_settings):

    modality = "CT"

    # Get settings.
    settings = create_settings(
        modality=modality,
        perturbation_settings=perturbation_settings)

    # Set testing directory
    if WRITE_TEMP_FILES:
        write_path = os.path.join(CURRENT_DIR, "data", "temp")

        # Create directory, if necessary.
        if not os.path.isdir(write_path):
            os.makedirs(write_path)
    else:
        write_path = None

    experiment = ExperimentClass(
        modality=modality,
        subject="test_subject",
        cohort=None,
        write_path=write_path,
        image_folder=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        roi_folder=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_reg_img_folder=None,
        image_file_name_pattern=None,
        registration_image_file_name_pattern=None,
        roi_names=["GTV-1"],
        data_str=[modality],
        provide_diagnostics=False,
        settings=settings,
        compute_features=True,
        extract_images=True,
        plot_images=False,
        keep_images_in_memory=False
    )

    return experiment


def create_settings(
        modality: str,
        perturbation_settings: ImagePerturbationSettingsClass):
    """Set default settings for generating response maps and computing feature values."""

    general_settings = GeneralSettingsClass(
        by_slice=False
    )

    new_spacing = 1.0

    image_interpolation_settings = ImageInterpolationSettingsClass(
        by_slice=False,
        interpolate=False,
        spline_order=3,
        new_spacing=new_spacing,
        anti_aliasing=False
    )

    resegmentation_intensity_range = [0.0, np.nan]
    if modality == "CT":
        resegmentation_intensity_range = [-200.0, 200.0]

    resegmentation_settings = ResegmentationSettingsClass(
        resegmentation_method="threshold",
        resegmentation_intensity_range=resegmentation_intensity_range
    )

    feature_computation_settings = FeatureExtractionSettingsClass(
        by_slice=False,
        no_approximation=False,
        base_feature_families=["morphological", "statistics"])

    image_transformation_settings = ImageTransformationSettingsClass(
        by_slice=False,
        response_map_feature_settings=None
    )

    settings = SettingsClass(
        general_settings=general_settings,
        post_process_settings=ImagePostProcessingClass(),
        img_interpolate_settings=image_interpolation_settings,
        roi_interpolate_settings=RoiInterpolationSettingsClass(),
        roi_resegment_settings=resegmentation_settings,
        perturbation_settings=perturbation_settings,
        img_transform_settings=image_transformation_settings,
        feature_extr_settings=feature_computation_settings
    )

    return settings

