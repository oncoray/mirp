import os

import numpy as np
import pandas as pd

from mirp.experimentClass import ExperimentClass
from mirp.settings.settingsClass import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass, \
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass, \
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

WRITE_TEMP_FILES = False


def test_noise_perturbation():

    perturbation_settings = ImagePerturbationSettingsClass(
        crop_around_roi=False,
        perturbation_noise_repetitions=2
    )

    # Set up experiment.
    data = run_experiment(perturbation_settings=perturbation_settings)
    feature_table = pd.concat([x[0] for x in data])
    image = [x[1][0] for x in data]
    mask = [x[2][0] for x in data]

    # Assert that object location and origin have not changed.
    assert np.allclose(image[0]["image_origin"], [-101.4000, -79.9255, -174.7290])
    assert np.allclose(image[0]["image_orientation"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(mask[0]["image_origin"], [-101.4000, -79.9255, -174.7290])
    assert np.allclose(mask[0]["image_orientation"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Check noise levels and ids.
    assert image[0]["noise_level"] > 0.0
    assert image[0]["noise_level"] == image[1]["noise_level"]
    assert image[0]["noise_id"] == 0
    assert image[1]["noise_id"] == 1

    # Check feature table.
    assert np.allclose(feature_table["morph_volume"], 357750.3)
    assert np.all(40.0 < feature_table["stat_mean"].values[0] < 50.0)
    assert np.all(40.0 < feature_table["stat_mean"].values[1] < 50.0)
    assert feature_table["stat_mean"].values[0] != feature_table["stat_mean"].values[1]
    assert not np.allclose(feature_table["stat_mean"], 43.085083)
    assert feature_table["image_noise_iteration_id"].values[0] == 0
    assert feature_table["image_noise_iteration_id"].values[1] == 1


def test_translation_perturbation():
    perturbation_settings = ImagePerturbationSettingsClass(
        crop_around_roi=False,
        perturbation_translation_fraction=0.1
    )

    # Run experiment.
    data = run_experiment(perturbation_settings=perturbation_settings)
    feature_table = pd.concat([x[0] for x in data])
    image = [x[1][0] for x in data]
    mask = [x[2][0] for x in data]

    # Origin has changed.
    assert not np.allclose(image[0]["image_origin"], [-101.4000, -79.9255, -174.7290])
    assert not np.allclose(mask[0]["image_origin"], [-101.4000, -79.9255, -174.7290])
    assert np.allclose(image[0]["image_origin"], [-101.300, -79.826, -174.629])
    assert np.allclose(mask[0]["image_origin"], [-101.300, -79.826, -174.629])

    # Assert that object orientation did not change.
    assert np.allclose(image[0]["image_orientation"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(mask[0]["image_orientation"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Volume should not change that much.
    assert 357500.0 < feature_table["morph_volume"][0] < 358500.0

    # Mean value should change slightly.
    assert 40.0 < feature_table["stat_mean"][0] < 50.0
    assert not np.isclose(feature_table["stat_mean"][0], 43.085083)


def test_translation_perturbation_multiple():
    perturbation_settings = ImagePerturbationSettingsClass(
        crop_around_roi=False,
        perturbation_translation_fraction=[0.0, 0.5]
    )

    # Run 3D experiment.
    data = run_experiment(perturbation_settings=perturbation_settings)
    feature_table = pd.concat([x[0] for x in data])
    image = [x[1][0] for x in data]

    # Check translation
    assert np.array_equal(image[0]["translation"], np.array([0.0, 0.0, 0.0]))
    assert np.array_equal(image[1]["translation"], np.array([0.5, 0.0, 0.0]))
    assert np.array_equal(image[2]["translation"], np.array([0.0, 0.5, 0.0]))
    assert np.array_equal(image[3]["translation"], np.array([0.5, 0.5, 0.0]))
    assert np.array_equal(image[4]["translation"], np.array([0.0, 0.0, 0.5]))
    assert np.array_equal(image[5]["translation"], np.array([0.5, 0.0, 0.5]))
    assert np.array_equal(image[6]["translation"], np.array([0.0, 0.5, 0.5]))
    assert np.array_equal(image[7]["translation"], np.array([0.5, 0.5, 0.5]))

    assert np.array_equal(
        feature_table["image_translation_z"].values,
        np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
    )
    assert np.array_equal(
        feature_table["image_translation_y"].values,
        np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5])
    )
    assert np.array_equal(
        feature_table["image_translation_x"].values,
        np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    )

    # Run 2D experiment.
    data = run_experiment(perturbation_settings=perturbation_settings, by_slice=True)
    feature_table = pd.concat([x[0] for x in data])
    image = [x[1][0] for x in data]

    # Check translation
    assert np.array_equal(image[0]["translation"], np.array([0.0, 0.0, 0.0]))
    assert np.array_equal(image[1]["translation"], np.array([0.0, 0.5, 0.0]))
    assert np.array_equal(image[2]["translation"], np.array([0.0, 0.0, 0.5]))
    assert np.array_equal(image[3]["translation"], np.array([0.0, 0.5, 0.5]))

    assert np.array_equal(
        feature_table["image_translation_z"].values,
        np.array([0.0, 0.0, 0.0, 0.0])
    )
    assert np.array_equal(
        feature_table["image_translation_y"].values,
        np.array([0.0, 0.5, 0.0, 0.5])
    )
    assert np.array_equal(
        feature_table["image_translation_x"].values,
        np.array([0.0, 0.0, 0.5, 0.5])
    )


def test_rotation_perturbation():
    perturbation_settings = ImagePerturbationSettingsClass(
        crop_around_roi=False,
        perturbation_rotation_angles=45.0
    )

    # Run experiment.
    data = run_experiment(perturbation_settings=perturbation_settings)
    feature_table = pd.concat([x[0] for x in data])
    image = [x[1][0] for x in data]
    mask = [x[2][0] for x in data]

    # Origin has changed in x-y plane.
    assert not np.allclose(image[0]["image_origin"], [-101.4000, -79.9255, -174.7290])
    assert not np.allclose(mask[0]["image_origin"], [-101.4000, -79.9255, -174.7290])
    assert np.allclose(image[0]["image_origin"], [-101.400, -121.579, -76.290])
    assert np.allclose(mask[0]["image_origin"], [-101.400, -121.579, -76.290])

    # Orientation has changed.
    assert np.allclose(
        image[0]["image_orientation"],
        [[1.0, 0.0, 0.0], [0.0, 1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)], [0.0, -1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]]
    )
    assert np.allclose(
        mask[0]["image_orientation"],
        [[1.0, 0.0, 0.0], [0.0, 1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)], [0.0, -1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]]
    )

    # Volume should not change that much.
    assert 357500.0 < feature_table["morph_volume"][0] < 358500.0

    # Mean value should change slightly.
    assert 40.0 < feature_table["stat_mean"][0] < 50.0
    assert ~np.isclose(feature_table["stat_mean"][0], 43.085083)


def test_rotation_perturbation_multiple():
    perturbation_settings = ImagePerturbationSettingsClass(
        crop_around_roi=False,
        perturbation_rotation_angles=[0.0, 45.0, 90.0]
    )

    # Run experiment.
    data = run_experiment(perturbation_settings=perturbation_settings)
    feature_table = pd.concat([x[0] for x in data])
    image = [x[1][0] for x in data]

    assert image[0]["rotation"] == 0.0
    assert image[1]["rotation"] == 45.0
    assert image[2]["rotation"] == 90.0

    assert feature_table["image_rotation_angle"].values[0] == 0.0
    assert feature_table["image_rotation_angle"].values[1] == 45.0
    assert feature_table["image_rotation_angle"].values[2] == 90.0


def test_perturbation_fraction_growth():
    perturbation_settings = ImagePerturbationSettingsClass(
        crop_around_roi=False,
        perturbation_roi_adapt_type="fraction",
        perturbation_roi_adapt_size=0.2
    )

    # Set up experiment.
    experiment = run_experiment(perturbation_settings=perturbation_settings)

    # Run computations.
    feature_table, img_obj, roi_list = experiment.process()

    # Origin remains the same.
    assert np.allclose(img_obj.origin, [-101.4000, -79.9255, -174.7290])
    assert np.allclose(roi_list[0].roi.origin, [-101.4000, -79.9255, -174.7290])

    # Assert that object orientation did not change.
    assert np.allclose(img_obj.orientation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(roi_list[0].roi.orientation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Volume should grow by 20%.
    assert 357500.0 * 1.20 < feature_table["morph_volume"][0] < 358500.0 * 1.20

    # Mean value should change slightly.
    assert 40.0 < feature_table["stat_mean"][0] < 50.0
    assert ~np.isclose(feature_table["stat_mean"][0], 43.085083)


def test_perturbation_fraction_shrink():
    perturbation_settings = ImagePerturbationSettingsClass(
        crop_around_roi=False,
        perturbation_roi_adapt_type="fraction",
        perturbation_roi_adapt_size=-0.2
    )

    # Set up experiment.
    experiment = run_experiment(perturbation_settings=perturbation_settings)

    # Run computations.
    feature_table, img_obj, roi_list = experiment.process()

    # Origin remains the same.
    assert np.allclose(img_obj.origin, [-101.4000, -79.9255, -174.7290])
    assert np.allclose(roi_list[0].roi.origin, [-101.4000, -79.9255, -174.7290])

    # Assert that object orientation did not change.
    assert np.allclose(img_obj.orientation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(roi_list[0].roi.orientation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Volume should shrink by 20%.
    assert 357500.0 * 0.80 < feature_table["morph_volume"][0] < 358500.0 * 0.80

    # Mean value should change slightly.
    assert 40.0 < feature_table["stat_mean"][0] < 50.0
    assert ~np.isclose(feature_table["stat_mean"][0], 43.085083)


def test_perturbation_distance_grow():
    perturbation_settings = ImagePerturbationSettingsClass(
        crop_around_roi=False,
        perturbation_roi_adapt_type="distance",
        perturbation_roi_adapt_size=2.0
    )

    # Set up experiment.
    experiment = run_experiment(perturbation_settings=perturbation_settings)

    # Run computations.
    feature_table, img_obj, roi_list = experiment.process()

    # Origin remains the same.
    assert np.allclose(img_obj.origin, [-101.4000, -79.9255, -174.7290])
    assert np.allclose(roi_list[0].roi.origin, [-101.4000, -79.9255, -174.7290])

    # Assert that object orientation did not change.
    assert np.allclose(img_obj.orientation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(roi_list[0].roi.orientation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Volume should grow.
    assert 420000.0 < feature_table["morph_volume"][0] < 421000.0

    # Mean value should change slightly.
    assert 40.0 < feature_table["stat_mean"][0] < 50.0
    assert ~np.isclose(feature_table["stat_mean"][0], 43.085083)


def test_perturbation_distance_shrink():
    perturbation_settings = ImagePerturbationSettingsClass(
        crop_around_roi=False,
        perturbation_roi_adapt_type="distance",
        perturbation_roi_adapt_size=-2.0
    )

    # Set up experiment.
    experiment = run_experiment(perturbation_settings=perturbation_settings)

    # Run computations.
    feature_table, img_obj, roi_list = experiment.process()

    # Origin remains the same.
    assert np.allclose(img_obj.origin, [-101.4000, -79.9255, -174.7290])
    assert np.allclose(roi_list[0].roi.origin, [-101.4000, -79.9255, -174.7290])

    # Assert that object orientation did not change.
    assert np.allclose(img_obj.orientation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(roi_list[0].roi.orientation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Volume should grow.
    assert 300000.0 < feature_table["morph_volume"][0] < 301000.0

    # Mean value should change slightly.
    assert 40.0 < feature_table["stat_mean"][0] < 50.0
    assert ~np.isclose(feature_table["stat_mean"][0], 43.085083)


def test_perturbation_roi_randomisation():
    perturbation_settings = ImagePerturbationSettingsClass(
        crop_around_roi=False,
        perturbation_randomise_roi_repetitions=1
    )

    # Set up experiment.
    experiment = run_experiment(perturbation_settings=perturbation_settings)

    # Run computations.
    feature_table, img_obj, roi_list = experiment.process()

    # Origin remains the same.
    assert np.allclose(img_obj.origin, [-101.4000, -79.9255, -174.7290])
    assert np.allclose(roi_list[0].roi.origin, [-101.4000, -79.9255, -174.7290])

    # Assert that object orientation did not change.
    assert np.allclose(img_obj.orientation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(roi_list[0].roi.orientation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Volume may change slightly.
    assert 350000.0 < feature_table["morph_volume"][0] < 370000.0

    # Mean value should change slightly.
    assert 40.0 < feature_table["stat_mean"][0] < 50.0
    assert ~np.isclose(feature_table["stat_mean"][0], 43.085083)


def test_perturbation_roi_randomisation_rotation():
    perturbation_settings = ImagePerturbationSettingsClass(
        crop_around_roi=False,
        perturbation_randomise_roi_repetitions=1,
        perturbation_rotation_angles=45.0
    )

    # Set up experiment.
    experiment = run_experiment(perturbation_settings=perturbation_settings)

    # Run computations.
    feature_table, img_obj, roi_list = experiment.process()

    # Origin has changed in x-y plane.
    assert ~np.allclose(img_obj.origin, [-101.4000, -79.9255, -174.7290])
    assert ~np.allclose(roi_list[0].roi.origin, [-101.4000, -79.9255, -174.7290])
    assert np.allclose(img_obj.origin, [-101.400, -121.579, -76.290])
    assert np.allclose(roi_list[0].roi.origin, [-101.400, -121.579, -76.290])

    # Orientation has changed.
    assert np.allclose(img_obj.orientation,
                       [[1.0, 0.0, 0.0],
                        [0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
                        [0.0, -1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]])
    assert np.allclose(roi_list[0].roi.orientation,
                       [[1.0, 0.0, 0.0],
                        [0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
                        [0.0, -1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]])

    # Volume may change slightly.
    assert 350000.0 < feature_table["morph_volume"][0] < 370000.0

    # Mean value should change slightly.
    assert 40.0 < feature_table["stat_mean"][0] < 50.0
    assert ~np.isclose(feature_table["stat_mean"][0], 43.085083)


def run_experiment(perturbation_settings, by_slice=False):
    from mirp.extractFeaturesAndImages import extract_features_and_images
    modality = "CT"

    # Get settings.
    settings = create_settings(
        modality=modality,
        by_slice=by_slice,
        perturbation_settings=perturbation_settings
    )

    # Set testing directory
    if WRITE_TEMP_FILES:
        write_path = os.path.join(CURRENT_DIR, "data", "temp")

        # Create directory, if necessary.
        if not os.path.isdir(write_path):
            os.makedirs(write_path)
    else:
        write_path = None

    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        settings=settings
    )

    return data


def create_settings(
        modality: str,
        by_slice: bool,
        perturbation_settings: ImagePerturbationSettingsClass
):
    """Set default settings for generating response maps and computing feature values."""

    general_settings = GeneralSettingsClass(
        by_slice=by_slice
    )

    new_spacing = 1.0

    image_interpolation_settings = ImageInterpolationSettingsClass(
        by_slice=False,
        interpolate=True,
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
