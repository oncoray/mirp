import os.path
import numpy as np
import pytest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

GENERIC_KWARGS = dict([
    ("write_features", False),
    ("export_features", True),
    ("image", os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1", "image")),
    ("mask", os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1", "mask")),
    ("roi_name", "GTV_Mass")
])


def test_morphological_features():
    # These are just functional tests: there is no reference standard.
    from mirp import extract_features

    data_compliant = extract_features(
        base_feature_families="morphology",
        ibsi_compliant=True,
        **GENERIC_KWARGS
    )[0]

    data_non_compliant = extract_features(
        base_feature_families="morphology",
        ibsi_compliant=False,
        **GENERIC_KWARGS
    )[0]

    non_compliant_features = ["morph_max_2d_diam_z", "morph_max_2d_diam_y", "morph_max_2d_diam_x"]
    assert all(x not in data_compliant.columns for x in non_compliant_features)
    assert all(x in data_non_compliant.columns for x in non_compliant_features)


def test_statistics_features():
    # Theses are just functional tests: there is no reference standard.
    from mirp import extract_features

    data_compliant = extract_features(
        base_feature_families="statistics",
        ibsi_compliant=True,
        **GENERIC_KWARGS
    )[0]

    data_non_compliant = extract_features(
        base_feature_families="statistics",
        ibsi_compliant=False,
        **GENERIC_KWARGS
    )[0]

    data_non_compliant_offset = extract_features(
        base_feature_families="statistics",
        stat_value_shift=100.0,
        ibsi_compliant=False,
        **GENERIC_KWARGS
    )[0]

    non_compliant_features = ["stat_energy_offset", "stat_rms_offset", "stat_total_energy", "stat_total_energy_offset"]
    assert all(x not in data_compliant.columns for x in non_compliant_features)
    assert all(x in data_non_compliant.columns for x in non_compliant_features)
    assert all(x in data_non_compliant_offset.columns for x in non_compliant_features)

    assert data_non_compliant.stat_energy_offset.to_numpy() < data_non_compliant_offset.stat_energy_offset.to_numpy()
    assert data_non_compliant.stat_rms_offset.to_numpy() < data_non_compliant_offset.stat_rms_offset.to_numpy()
    assert data_non_compliant.stat_total_energy_offset.to_numpy() < data_non_compliant_offset.stat_total_energy_offset.to_numpy()

    # Test other percentiles.
    data_pct = extract_features(
        base_feature_families="statistics",
        ibsi_compliant=True,
        stat_percentile=[10.0, 20.0, 80.0, 90.0],
        **GENERIC_KWARGS
    )[0]

    assert (data_pct.stat_p10.to_numpy() < data_pct.stat_p20.to_numpy() < data_pct.stat_p80.to_numpy() <
            data_pct.stat_p90.to_numpy())


def test_pooled_texture_features():
    from mirp import extract_features

    # It should not be possible to use pooling methods other than average for IBSI-compliant workflows.
    with pytest.raises(ValueError, match="set ibsi_compliant=False"):
        _ = extract_features(
            base_feature_families=["cm"],
            ibsi_compliant=True,
            base_discretisation_method="fixed_bin_number",
            base_discretisation_n_bins=16,
            texture_feature_pooling_method=["average", "min", "max"],
            glcm_spatial_method=["3d_average", "3d_volume_merge"],
            **GENERIC_KWARGS
        )

    # Test that all pooling methods function.
    data = extract_features(
            base_feature_families=["cm"],
            ibsi_compliant=False,
            base_discretisation_method="fixed_bin_number",
            base_discretisation_n_bins=16,
            texture_feature_pooling_method=["average", "min", "max", "range", "std", "var"],
            glcm_spatial_method="3d_average",
            **GENERIC_KWARGS
        )[0]

    assert data.columns.str.startswith("cm_").sum() == 156
    assert data.cm_joint_max_d1_3d_range_fbn_n16.to_numpy()[0] == data.cm_joint_max_d1_3d_max_fbn_n16.to_numpy()[0] - \
           data.cm_joint_max_d1_3d_min_fbn_n16.to_numpy()[0]
    assert data.cm_joint_max_d1_3d_min_fbn_n16.to_numpy()[0] < data.cm_joint_max_d1_3d_avg_fbn_n16.to_numpy()[0] < \
           data.cm_joint_max_d1_3d_max_fbn_n16.to_numpy()[0]
    assert data.cm_joint_max_d1_3d_std_fbn_n16.to_numpy()[0] == np.sqrt(data.cm_joint_max_d1_3d_var_fbn_n16.to_numpy()[0])

    # Test that pooling methods revert to average for methods that merge all matrices instead.
    data = extract_features(
        base_feature_families=["cm"],
        ibsi_compliant=False,
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=16,
        texture_feature_pooling_method=["average", "min", "max", "range", "std", "var"],
        glcm_spatial_method="3d_volume_merge",
        **GENERIC_KWARGS
    )[0]

    assert data.columns.str.startswith("cm_").sum() == 26
    assert "cm_joint_max_d1_3d_v_mrg_fbn_n16" in data.columns
    assert "cm_joint_max_d1_3d_v_mrg_std_fbn_n16" not in data.columns

    # Test that pooling methods work with all directional spatial methods.
    data = extract_features(
        base_feature_families=["cm"],
        ibsi_compliant=False,
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=16,
        texture_feature_pooling_method=["min"],
        glcm_spatial_method=[
            "2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge", "3d_average", "3d_volume_merge"
        ],
        **GENERIC_KWARGS
    )[0]

    assert data.columns.str.startswith("cm_").sum() == 156
    assert "cm_joint_max_d1_2d_min_fbn_n16" in data.columns
    assert "cm_joint_max_d1_2d_s_mrg_min_fbn_n16" in data.columns
    assert "cm_joint_max_d1_2.5d_d_mrg_min_fbn_n16" in data.columns
    assert "cm_joint_max_d1_2.5d_v_mrg_fbn_n16" in data.columns  # Matrix merging method
    assert "cm_joint_max_d1_3d_min_fbn_n16" in data.columns
    assert "cm_joint_max_d1_3d_v_mrg_fbn_n16" in data.columns  # Matrix merging method

    # Test that pooling methods work with all non-directional spatial methods.
    data = extract_features(
        base_feature_families=["szm"],
        ibsi_compliant=False,
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=16,
        texture_feature_pooling_method=["min"],
        glszm_spatial_method=["2d", "2.5d", "3d"],
        **GENERIC_KWARGS
    )[0]

    assert data.columns.str.startswith("szm_").sum() == 48
    assert "szm_sze_2d_min_fbn_n16" in data.columns
    assert "szm_sze_2.5d_fbn_n16" in data.columns  # Matrix merging method
    assert "szm_sze_3d_fbn_n16" in data.columns  # Matrix merging method

    # Check for all types of texture features - RLM (GLCM already tested)
    data = extract_features(
        base_feature_families="rlm",
        ibsi_compliant=False,
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=16,
        texture_feature_pooling_method="min",
        glrlm_spatial_method="3d_average",
        **GENERIC_KWARGS
    )[0]

    assert data.columns.str.startswith("rlm_").sum() == 16
    assert "rlm_sre_3d_min_fbn_n16" in data.columns

    # Check for all types of texture features - DZM (SZM already tested).
    data = extract_features(
        base_feature_families="dzm",
        ibsi_compliant=False,
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=16,
        texture_feature_pooling_method="min",
        gldzm_spatial_method="2d",
        **GENERIC_KWARGS
    )[0]

    assert data.columns.str.startswith("dzm_").sum() == 16
    assert "dzm_sde_2d_min_fbn_n16" in data.columns

    # Check for all types of texture features - NGTDM
    data = extract_features(
        base_feature_families="ngtdm",
        ibsi_compliant=False,
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=16,
        texture_feature_pooling_method="min",
        ngtdm_spatial_method="2d",
        **GENERIC_KWARGS
    )[0]

    assert data.columns.str.startswith("ngt_").sum() == 5
    assert "ngt_coarseness_2d_min_fbn_n16" in data.columns

    # Check for all types of texture features - NGLDM
    data = extract_features(
        base_feature_families="ngldm",
        ibsi_compliant=False,
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=16,
        texture_feature_pooling_method="min",
        ngldm_spatial_method="2d",
        **GENERIC_KWARGS
    )[0]

    assert data.columns.str.startswith("ngl_").sum() == 17
    assert "ngl_lde_d1_a0.0_2d_min_fbn_n16" in data.columns