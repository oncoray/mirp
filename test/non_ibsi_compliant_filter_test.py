import os
import numpy as np
from mirp import extract_features_and_images

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_square_transformation_filter():
    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        ibsi_compliant=False,
        base_feature_families="statistics",
        filter_kernels="pyradiomics_square"
    )

    feature_data = data[0][0]
    assert len(feature_data) == 1
    assert feature_data["stat_min"].values[0] == -1000.0
    assert feature_data["square_stat_min"].values[0] == 0.0
    assert np.max(data[0][1][0].get_voxel_grid()) == np.max(data[0][1][1].get_voxel_grid())
    assert not np.array_equal(data[0][1][0].get_voxel_grid(), data[0][1][1].get_voxel_grid())


def test_square_root_transformation_filter():
    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        ibsi_compliant=False,
        base_feature_families="statistics",
        filter_kernels="pyradiomics_square_root"
    )

    feature_data = data[0][0]
    assert len(feature_data) == 1
    assert feature_data["stat_min"].values[0] == -1000.0
    assert feature_data["sqrt_stat_min"].values[0] < 0.0
    assert np.max(data[0][1][0].get_voxel_grid()) == np.max(data[0][1][1].get_voxel_grid())
    assert not np.array_equal(data[0][1][0].get_voxel_grid(), data[0][1][1].get_voxel_grid())


def test_logarithm_transformation_filter():
    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        ibsi_compliant=False,
        base_feature_families="statistics",
        filter_kernels="pyradiomics_logarithm"
    )

    feature_data = data[0][0]
    assert len(feature_data) == 1
    assert feature_data["stat_min"].values[0] == -1000.0
    assert feature_data["lgrthm_stat_min"].values[0] < 0.0
    assert np.max(data[0][1][0].get_voxel_grid()) == np.max(data[0][1][1].get_voxel_grid())
    assert not np.array_equal(data[0][1][0].get_voxel_grid(), data[0][1][1].get_voxel_grid())


def test_exponential_transformation_filter():
    data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=True,
        image_export_format="native",
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        ibsi_compliant=False,
        base_feature_families="statistics",
        filter_kernels="pyradiomics_exponential"
    )

    feature_data = data[0][0]
    assert len(feature_data) == 1
    assert feature_data["stat_min"].values[0] == -1000.0
    assert feature_data["exp_stat_min"].values[0] > 0.0
    assert np.isclose(np.max(data[0][1][0].get_voxel_grid()), np.max(data[0][1][1].get_voxel_grid()))
    assert not np.array_equal(data[0][1][0].get_voxel_grid(), data[0][1][1].get_voxel_grid())
