import os

import numpy as np
import pandas as pd

from mirp.settings.generic import SettingsClass
from mirp.settings.transformation_parameters import ImageTransformationSettingsClass
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
from mirp.settings.resegmentation_parameters import ResegmentationSettingsClass
from mirp.settings.perturbation_parameters import ImagePerturbationSettingsClass
from mirp.settings.image_processing_parameters import ImagePostProcessingClass
from mirp.settings.interpolation_parameters import ImageInterpolationSettingsClass, MaskInterpolationSettingsClass
from mirp.settings.general_parameters import GeneralSettingsClass

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REMOVE_TEMP_RESPONSE_MAPS = False


def test_orientation():
    """
    Test internal representation of image objects using the orientation phantom.
    """
    from mirp._data_import.read_data import read_image
    from mirp.data_import.import_image import import_image

    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "misc_images", "orientation", "image", "orientation.nii.gz")
    )

    image = read_image(image=image_list[0])

    # Assert minimum and maximum values in the voxel grid.
    assert np.min(image.get_voxel_grid()) == 0.0
    assert np.max(image.get_voxel_grid()) == 141.0

    # Check dimensions. MIRP expects a (z, y, x) orientation.
    assert np.array_equal(image.image_dimension, (64, 48, 32))

    # Check orientation. The minimum value should be in the origin, and the maximum value in the most distal voxel.
    assert image.get_voxel_grid()[0, 0, 0] == 0.0
    assert image.get_voxel_grid()[-1, -1, -1] == 141.0

    # Check if origin and spacing match initial values.
    assert np.array_equal(image.image_origin, (0.0, 1.0, 2.0))
    assert np.array_equal(image.image_spacing, (0.5, 1.0, 1.5))

    # Check if the affine matrix is correct.
    assert np.array_equal(
        image.image_orientation, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )


def run_experiment(image, roi, **kwargs):
    from mirp.extract_features_and_images import extract_features

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
        base_feature_families="all",
        base_discretisation_method=["fixed_bin_number", "fixed_bin_size"],
        base_discretisation_n_bins=12,
        base_discretisation_bin_width=25.0,
        ivh_discretisation_method="fixed_bin_number",
        glcm_distance=[1.0],
        glcm_spatial_method=[
            "2d_average", "2d_slice_merge",
            "2.5d_direction_merge", "2.5d_volume_merge",
            "3d_average", "3d_volume_merge"
        ],
        glrlm_spatial_method=[
            "2d_average", "2d_slice_merge",
            "2.5d_direction_merge", "2.5d_volume_merge",
            "3d_average", "3d_volume_merge"
        ],
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
        roi_interpolate_settings=MaskInterpolationSettingsClass(),
        roi_resegment_settings=ResegmentationSettingsClass(**kwargs),
        perturbation_settings=ImagePerturbationSettingsClass(),
        img_transform_settings=image_transformation_settings,
        feature_extr_settings=feature_computation_parameters
    )

    data = extract_features(
        write_features=False,
        export_features=True,
        image=os.path.join(CURRENT_DIR, "data", "misc_images", image, "image"),
        image_modality="CT",
        mask=os.path.join(CURRENT_DIR, "data", "misc_images", image, "mask"),
        mask_name=roi,
        settings=settings
    )

    data = data[0]
    return data


def test_xml_configurations(tmp_path):
    # Read the data settings xml file, and update path to image and mask.
    from xml.etree import ElementTree as ElemTree
    from mirp.extract_features_and_images import extract_features

    # Load xml.
    tree = ElemTree.parse(os.path.join(CURRENT_DIR, "data", "configuration_files", "test_config_data.xml"))
    paths_branch = tree.getroot()

    # Update paths in xml file.
    for image in paths_branch.iter("image"):
        image.text = str(os.path.join(CURRENT_DIR, "data", "sts_images"))
    for mask in paths_branch.iter("mask"):
        mask.text = str(os.path.join(CURRENT_DIR, "data", "sts_images"))

    # Save as temporary xml file.
    tree.write(tmp_path / "temp_test_config_data.xml")

    data = extract_features(
        write_features=False,
        export_features=True,
        image=str(tmp_path / "temp_test_config_data.xml"),
        settings=os.path.join(CURRENT_DIR, "data", "configuration_files", "test_config_settings.xml")
    )

    data = pd.concat(data)
    assert len(data) == 2
    assert all(data["sample_name"].values == ["STS_002", "STS_003"])
    assert all(data["image_modality"].values == "pet")
    assert all(data["image_mask_name"].values == "GTV_Mass_PET")
    assert all(data["image_voxel_size_x"].values == 3.0)
    assert all(data["image_voxel_size_y"].values == 3.0)
    assert all(data["image_voxel_size_z"].values == 3.0)


def test_edge_cases_basic_pipeline():
    """
    Test feature extraction using the basic pipeline. The following cases are tested using an uninformative phantom
    that has the value 1 everywhere:
    -   using a normal mask that completely covers the image. This is to test how MIRP responds to uninformative _images.
    -   using a mask that only contains a single voxel. This is to test how MIRP responds to _masks with a single voxel.
    -   using a mask that has disconnected voxels.
    -   using an empty mask.
    -   using a mask that becomes empty after resegmentation.

    Both 3D and 2D (slice) phantoms are used.

    So in short, this pipeline tests the worst possible _images and _masks to figure out what happens.
    """

    images = ["uninformative", "uninformative_slice"]
    rois = ["full_mask", "one_voxel_mask", "disconnected_mask", "empty_mask"]

    for image in images:
        for roi in rois:
            # Setup experiment.
            data = run_experiment(image=image, roi=roi)

            # Test
            if roi == "empty_mask":
                assert data is None

            else:
                assert isinstance(data, pd.DataFrame)

    # Test ROI that becomes empty after re-segmentation
    for image in images:
        # Resegmentation
        data = run_experiment(
            image=image,
            roi="full_mask",
            resegmentation_intensity_range=[100.0, 200.0]
        )

        # Setup experiment.
        assert isinstance(data, pd.DataFrame)
