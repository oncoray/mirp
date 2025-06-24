import os
import sys
import pytest
from math import log10, floor

from mirp.extract_features_and_images import extract_features
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


def within_tolerance(ref, tol, x):
    # Read from pandas Series
    x = x.values[0]

    # Prevent issues due to rounding errors.
    x = float(x)
    tol = float(tol)

    # Round correctly
    if x != 0.0:
        x = round(x, 3 - int(floor(log10(abs(x)))) - 1)

    if abs(x - ref) - tol <= sys.float_info.epsilon:
        return True
    else:
        return False


@pytest.mark.ci
def test_ibsi_1_digital_phantom():
    """
    Compare computed feature values with reference values for the digital phantom.
    """

    # Configure settings used for the digital phantom.
    general_settings = GeneralSettingsClass(by_slice=False)

    image_interpolation_settings = ImageInterpolationSettingsClass(
        by_slice=general_settings.by_slice,
        anti_aliasing=False
    )

    feature_computation_parameters = FeatureExtractionSettingsClass(
        by_slice=general_settings.by_slice,
        ibsi_compliant=general_settings.ibsi_compliant,
        no_approximation=True,
        base_feature_families="all",
        base_discretisation_method="none",
        ivh_discretisation_method="none",
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
        roi_resegment_settings=ResegmentationSettingsClass(),
        perturbation_settings=ImagePerturbationSettingsClass(),
        img_transform_settings=image_transformation_settings,
        feature_extr_settings=feature_computation_parameters
    )

    data = extract_features(
        write_features=False,
        export_features=True,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_digital_phantom", "nifti", "image", "phantom.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_digital_phantom", "nifti", "mask", "mask.nii.gz"),
        image_modality="ct",
        settings=settings
    )

    data = data[0]

    assert within_tolerance(556, 4, data["morph_volume"])
    assert within_tolerance(592, 4, data["morph_vol_approx"])
    assert within_tolerance(388, 3, data["morph_area_mesh"])
    assert within_tolerance(0.698, 0.004, data["morph_av"])
    assert within_tolerance(0.0411, 3e-04, data["morph_comp_1"])
    assert within_tolerance(0.599, 0.004, data["morph_comp_2"])
    assert within_tolerance(1.19, 0.01, data["morph_sph_dispr"])
    assert within_tolerance(0.843, 0.005, data["morph_sphericity"])
    assert within_tolerance(0.186, 0.001, data["morph_asphericity"])
    assert within_tolerance(0.672, 0.004, data["morph_com"])
    assert within_tolerance(13.1, 0.1, data["morph_diam"])
    assert within_tolerance(11.4, 0.1, data["morph_pca_maj_axis"])
    assert within_tolerance(9.31, 0.06, data["morph_pca_min_axis"])
    assert within_tolerance(8.54, 0.05, data["morph_pca_least_axis"])
    assert within_tolerance(0.816, 0.005, data["morph_pca_elongation"])
    assert within_tolerance(0.749, 0.005, data["morph_pca_flatness"])
    assert within_tolerance(0.869, 0.005, data["morph_vol_dens_aabb"])
    assert within_tolerance(0.866, 0.005, data["morph_area_dens_aabb"])
    assert within_tolerance(1.17, 0.01, data["morph_vol_dens_aee"])
    assert within_tolerance(1.36, 0.01, data["morph_area_dens_aee"])
    assert within_tolerance(0.961, 0.006, data["morph_vol_dens_conv_hull"])
    assert within_tolerance(1.03, 0.01, data["morph_area_dens_conv_hull"])
    assert within_tolerance(1200, 10, data["morph_integ_int"])
    assert within_tolerance(0.0397, 3e-04, data["morph_moran_i"])
    assert within_tolerance(0.974, 0.006, data["morph_geary_c"])
    assert within_tolerance(2.6, 0, data["loc_peak_loc"])
    assert within_tolerance(3.1, 0, data["loc_peak_glob"])
    assert within_tolerance(2.15, 0, data["stat_mean"])
    assert within_tolerance(3.05, 0, data["stat_var"])
    assert within_tolerance(1.08, 0, data["stat_skew"])
    assert within_tolerance(-0.355, 0, data["stat_kurt"])
    assert within_tolerance(1, 0, data["stat_median"])
    assert within_tolerance(1, 0, data["stat_min"])
    assert within_tolerance(1, 0, data["stat_p10"])
    assert within_tolerance(4, 0, data["stat_p90"])
    assert within_tolerance(6, 0, data["stat_max"])
    assert within_tolerance(3, 0, data["stat_iqr"])
    assert within_tolerance(5, 0, data["stat_range"])
    assert within_tolerance(1.55, 0, data["stat_mad"])
    assert within_tolerance(1.11, 0, data["stat_rmad"])
    assert within_tolerance(1.15, 0, data["stat_medad"])
    assert within_tolerance(0.812, 0, data["stat_cov"])
    assert within_tolerance(0.6, 0, data["stat_qcod"])
    assert within_tolerance(567, 0, data["stat_energy"])
    assert within_tolerance(2.77, 0, data["stat_rms"])
    assert within_tolerance(2.15, 0, data["ih_mean"])
    assert within_tolerance(3.05, 0, data["ih_var"])
    assert within_tolerance(1.08, 0, data["ih_skew"])
    assert within_tolerance(-0.355, 0, data["ih_kurt"])
    assert within_tolerance(1, 0, data["ih_median"])
    assert within_tolerance(1, 0, data["ih_min"])
    assert within_tolerance(1, 0, data["ih_p10"])
    assert within_tolerance(4, 0, data["ih_p90"])
    assert within_tolerance(6, 0, data["ih_max"])
    assert within_tolerance(1, 0, data["ih_mode"])
    assert within_tolerance(3, 0, data["ih_iqr"])
    assert within_tolerance(5, 0, data["ih_range"])
    assert within_tolerance(1.55, 0, data["ih_mad"])
    assert within_tolerance(1.11, 0, data["ih_rmad"])
    assert within_tolerance(1.15, 0, data["ih_medad"])
    assert within_tolerance(0.812, 0, data["ih_cov"])
    assert within_tolerance(0.6, 0, data["ih_qcod"])
    assert within_tolerance(1.27, 0, data["ih_entropy"])
    assert within_tolerance(0.512, 0, data["ih_uniformity"])
    assert within_tolerance(8, 0, data["ih_max_grad"])
    assert within_tolerance(3, 0, data["ih_max_grad_g"])
    assert within_tolerance(-50, 0, data["ih_min_grad"])
    assert within_tolerance(1, 0, data["ih_min_grad_g"])
    assert within_tolerance(0.324, 0, data["ivh_v10"])
    assert within_tolerance(0.0946, 0, data["ivh_v90"])
    assert within_tolerance(5, 0, data["ivh_i10"])
    assert within_tolerance(2, 0, data["ivh_i90"])
    assert within_tolerance(0.23, 0, data["ivh_diff_v10_v90"])
    assert within_tolerance(3, 0, data["ivh_diff_i10_i90"])
    assert within_tolerance(0.519, 0, data["cm_joint_max_d1_2d_avg"])
    assert within_tolerance(2.14, 0, data["cm_joint_avg_d1_2d_avg"])
    assert within_tolerance(2.69, 0, data["cm_joint_var_d1_2d_avg"])
    assert within_tolerance(2.05, 0, data["cm_joint_entr_d1_2d_avg"])
    assert within_tolerance(1.42, 0, data["cm_diff_avg_d1_2d_avg"])
    assert within_tolerance(2.9, 0, data["cm_diff_var_d1_2d_avg"])
    assert within_tolerance(1.4, 0, data["cm_diff_entr_d1_2d_avg"])
    assert within_tolerance(4.28, 0, data["cm_sum_avg_d1_2d_avg"])
    assert within_tolerance(5.47, 0, data["cm_sum_var_d1_2d_avg"])
    assert within_tolerance(1.6, 0, data["cm_sum_entr_d1_2d_avg"])
    assert within_tolerance(0.368, 0, data["cm_energy_d1_2d_avg"])
    assert within_tolerance(5.28, 0, data["cm_contrast_d1_2d_avg"])
    assert within_tolerance(1.42, 0, data["cm_dissimilarity_d1_2d_avg"])
    assert within_tolerance(0.678, 0, data["cm_inv_diff_d1_2d_avg"])
    assert within_tolerance(0.851, 0, data["cm_inv_diff_norm_d1_2d_avg"])
    assert within_tolerance(0.619, 0, data["cm_inv_diff_mom_d1_2d_avg"])
    assert within_tolerance(0.899, 0, data["cm_inv_diff_mom_norm_d1_2d_avg"])
    assert within_tolerance(0.0567, 0, data["cm_inv_var_d1_2d_avg"])
    assert within_tolerance(-0.0121, 0, data["cm_corr_d1_2d_avg"])
    assert within_tolerance(5.09, 0, data["cm_auto_corr_d1_2d_avg"])
    assert within_tolerance(5.47, 0, data["cm_clust_tend_d1_2d_avg"])
    assert within_tolerance(7, 0, data["cm_clust_shade_d1_2d_avg"])
    assert within_tolerance(79.1, 0, data["cm_clust_prom_d1_2d_avg"])
    assert within_tolerance(-0.155, 0, data["cm_info_corr1_d1_2d_avg"])
    assert within_tolerance(0.487, 0, data["cm_info_corr2_d1_2d_avg"])
    assert within_tolerance(0.512, 0, data["cm_joint_max_d1_2d_s_mrg"])
    assert within_tolerance(2.14, 0, data["cm_joint_avg_d1_2d_s_mrg"])
    assert within_tolerance(2.71, 0, data["cm_joint_var_d1_2d_s_mrg"])
    assert within_tolerance(2.24, 0, data["cm_joint_entr_d1_2d_s_mrg"])
    assert within_tolerance(1.4, 0, data["cm_diff_avg_d1_2d_s_mrg"])
    assert within_tolerance(3.06, 0, data["cm_diff_var_d1_2d_s_mrg"])
    assert within_tolerance(1.49, 0, data["cm_diff_entr_d1_2d_s_mrg"])
    assert within_tolerance(4.29, 0, data["cm_sum_avg_d1_2d_s_mrg"])
    assert within_tolerance(5.66, 0, data["cm_sum_var_d1_2d_s_mrg"])
    assert within_tolerance(1.79, 0, data["cm_sum_entr_d1_2d_s_mrg"])
    assert within_tolerance(0.352, 0, data["cm_energy_d1_2d_s_mrg"])
    assert within_tolerance(5.19, 0, data["cm_contrast_d1_2d_s_mrg"])
    assert within_tolerance(1.4, 0, data["cm_dissimilarity_d1_2d_s_mrg"])
    assert within_tolerance(0.683, 0, data["cm_inv_diff_d1_2d_s_mrg"])
    assert within_tolerance(0.854, 0, data["cm_inv_diff_norm_d1_2d_s_mrg"])
    assert within_tolerance(0.625, 0, data["cm_inv_diff_mom_d1_2d_s_mrg"])
    assert within_tolerance(0.901, 0, data["cm_inv_diff_mom_norm_d1_2d_s_mrg"])
    assert within_tolerance(0.0553, 0, data["cm_inv_var_d1_2d_s_mrg"])
    assert within_tolerance(0.0173, 0, data["cm_corr_d1_2d_s_mrg"])
    assert within_tolerance(5.14, 0, data["cm_auto_corr_d1_2d_s_mrg"])
    assert within_tolerance(5.66, 0, data["cm_clust_tend_d1_2d_s_mrg"])
    assert within_tolerance(6.98, 0, data["cm_clust_shade_d1_2d_s_mrg"])
    assert within_tolerance(80.4, 0, data["cm_clust_prom_d1_2d_s_mrg"])
    assert within_tolerance(-0.0341, 0, data["cm_info_corr1_d1_2d_s_mrg"])
    assert within_tolerance(0.263, 0, data["cm_info_corr2_d1_2d_s_mrg"])
    assert within_tolerance(0.489, 0, data["cm_joint_max_d1_2.5d_d_mrg"])
    assert within_tolerance(2.2, 0, data["cm_joint_avg_d1_2.5d_d_mrg"])
    assert within_tolerance(3.22, 0, data["cm_joint_var_d1_2.5d_d_mrg"])
    assert within_tolerance(2.48, 0, data["cm_joint_entr_d1_2.5d_d_mrg"])
    assert within_tolerance(1.46, 0, data["cm_diff_avg_d1_2.5d_d_mrg"])
    assert within_tolerance(3.11, 0, data["cm_diff_var_d1_2.5d_d_mrg"])
    assert within_tolerance(1.61, 0, data["cm_diff_entr_d1_2.5d_d_mrg"])
    assert within_tolerance(4.41, 0, data["cm_sum_avg_d1_2.5d_d_mrg"])
    assert within_tolerance(7.48, 0, data["cm_sum_var_d1_2.5d_d_mrg"])
    assert within_tolerance(2.01, 0, data["cm_sum_entr_d1_2.5d_d_mrg"])
    assert within_tolerance(0.286, 0, data["cm_energy_d1_2.5d_d_mrg"])
    assert within_tolerance(5.39, 0, data["cm_contrast_d1_2.5d_d_mrg"])
    assert within_tolerance(1.46, 0, data["cm_dissimilarity_d1_2.5d_d_mrg"])
    assert within_tolerance(0.668, 0, data["cm_inv_diff_d1_2.5d_d_mrg"])
    assert within_tolerance(0.847, 0, data["cm_inv_diff_norm_d1_2.5d_d_mrg"])
    assert within_tolerance(0.606, 0, data["cm_inv_diff_mom_d1_2.5d_d_mrg"])
    assert within_tolerance(0.897, 0, data["cm_inv_diff_mom_norm_d1_2.5d_d_mrg"])
    assert within_tolerance(0.0597, 0, data["cm_inv_var_d1_2.5d_d_mrg"])
    assert within_tolerance(0.178, 0, data["cm_corr_d1_2.5d_d_mrg"])
    assert within_tolerance(5.4, 0, data["cm_auto_corr_d1_2.5d_d_mrg"])
    assert within_tolerance(7.48, 0, data["cm_clust_tend_d1_2.5d_d_mrg"])
    assert within_tolerance(16.6, 0, data["cm_clust_shade_d1_2.5d_d_mrg"])
    assert within_tolerance(147, 0, data["cm_clust_prom_d1_2.5d_d_mrg"])
    assert within_tolerance(-0.124, 0, data["cm_info_corr1_d1_2.5d_d_mrg"])
    assert within_tolerance(0.487, 0, data["cm_info_corr2_d1_2.5d_d_mrg"])
    assert within_tolerance(0.492, 0, data["cm_joint_max_d1_2.5d_v_mrg"])
    assert within_tolerance(2.2, 0, data["cm_joint_avg_d1_2.5d_v_mrg"])
    assert within_tolerance(3.24, 0, data["cm_joint_var_d1_2.5d_v_mrg"])
    assert within_tolerance(2.61, 0, data["cm_joint_entr_d1_2.5d_v_mrg"])
    assert within_tolerance(1.44, 0, data["cm_diff_avg_d1_2.5d_v_mrg"])
    assert within_tolerance(3.23, 0, data["cm_diff_var_d1_2.5d_v_mrg"])
    assert within_tolerance(1.67, 0, data["cm_diff_entr_d1_2.5d_v_mrg"])
    assert within_tolerance(4.41, 0, data["cm_sum_avg_d1_2.5d_v_mrg"])
    assert within_tolerance(7.65, 0, data["cm_sum_var_d1_2.5d_v_mrg"])
    assert within_tolerance(2.14, 0, data["cm_sum_entr_d1_2.5d_v_mrg"])
    assert within_tolerance(0.277, 0, data["cm_energy_d1_2.5d_v_mrg"])
    assert within_tolerance(5.29, 0, data["cm_contrast_d1_2.5d_v_mrg"])
    assert within_tolerance(1.44, 0, data["cm_dissimilarity_d1_2.5d_v_mrg"])
    assert within_tolerance(0.673, 0, data["cm_inv_diff_d1_2.5d_v_mrg"])
    assert within_tolerance(0.85, 0, data["cm_inv_diff_norm_d1_2.5d_v_mrg"])
    assert within_tolerance(0.613, 0, data["cm_inv_diff_mom_d1_2.5d_v_mrg"])
    assert within_tolerance(0.899, 0, data["cm_inv_diff_mom_norm_d1_2.5d_v_mrg"])
    assert within_tolerance(0.0582, 0, data["cm_inv_var_d1_2.5d_v_mrg"])
    assert within_tolerance(0.182, 0, data["cm_corr_d1_2.5d_v_mrg"])
    assert within_tolerance(5.45, 0, data["cm_auto_corr_d1_2.5d_v_mrg"])
    assert within_tolerance(7.65, 0, data["cm_clust_tend_d1_2.5d_v_mrg"])
    assert within_tolerance(16.4, 0, data["cm_clust_shade_d1_2.5d_v_mrg"])
    assert within_tolerance(142, 0, data["cm_clust_prom_d1_2.5d_v_mrg"])
    assert within_tolerance(-0.0334, 0, data["cm_info_corr1_d1_2.5d_v_mrg"])
    assert within_tolerance(0.291, 0, data["cm_info_corr2_d1_2.5d_v_mrg"])
    assert within_tolerance(0.503, 0, data["cm_joint_max_d1_3d_avg"])
    assert within_tolerance(2.14, 0, data["cm_joint_avg_d1_3d_avg"])
    assert within_tolerance(3.1, 0, data["cm_joint_var_d1_3d_avg"])
    assert within_tolerance(2.4, 0, data["cm_joint_entr_d1_3d_avg"])
    assert within_tolerance(1.43, 0, data["cm_diff_avg_d1_3d_avg"])
    assert within_tolerance(3.06, 0, data["cm_diff_var_d1_3d_avg"])
    assert within_tolerance(1.56, 0, data["cm_diff_entr_d1_3d_avg"])
    assert within_tolerance(4.29, 0, data["cm_sum_avg_d1_3d_avg"])
    assert within_tolerance(7.07, 0, data["cm_sum_var_d1_3d_avg"])
    assert within_tolerance(1.92, 0, data["cm_sum_entr_d1_3d_avg"])
    assert within_tolerance(0.303, 0, data["cm_energy_d1_3d_avg"])
    assert within_tolerance(5.32, 0, data["cm_contrast_d1_3d_avg"])
    assert within_tolerance(1.43, 0, data["cm_dissimilarity_d1_3d_avg"])
    assert within_tolerance(0.677, 0, data["cm_inv_diff_d1_3d_avg"])
    assert within_tolerance(0.851, 0, data["cm_inv_diff_norm_d1_3d_avg"])
    assert within_tolerance(0.618, 0, data["cm_inv_diff_mom_d1_3d_avg"])
    assert within_tolerance(0.898, 0, data["cm_inv_diff_mom_norm_d1_3d_avg"])
    assert within_tolerance(0.0604, 0, data["cm_inv_var_d1_3d_avg"])
    assert within_tolerance(0.157, 0, data["cm_corr_d1_3d_avg"])
    assert within_tolerance(5.06, 0, data["cm_auto_corr_d1_3d_avg"])
    assert within_tolerance(7.07, 0, data["cm_clust_tend_d1_3d_avg"])
    assert within_tolerance(16.6, 0, data["cm_clust_shade_d1_3d_avg"])
    assert within_tolerance(145, 0, data["cm_clust_prom_d1_3d_avg"])
    assert within_tolerance(-0.157, 0, data["cm_info_corr1_d1_3d_avg"])
    assert within_tolerance(0.52, 0, data["cm_info_corr2_d1_3d_avg"])
    assert within_tolerance(0.509, 0, data["cm_joint_max_d1_3d_v_mrg"])
    assert within_tolerance(2.15, 0, data["cm_joint_avg_d1_3d_v_mrg"])
    assert within_tolerance(3.13, 0, data["cm_joint_var_d1_3d_v_mrg"])
    assert within_tolerance(2.57, 0, data["cm_joint_entr_d1_3d_v_mrg"])
    assert within_tolerance(1.38, 0, data["cm_diff_avg_d1_3d_v_mrg"])
    assert within_tolerance(3.21, 0, data["cm_diff_var_d1_3d_v_mrg"])
    assert within_tolerance(1.64, 0, data["cm_diff_entr_d1_3d_v_mrg"])
    assert within_tolerance(4.3, 0, data["cm_sum_avg_d1_3d_v_mrg"])
    assert within_tolerance(7.41, 0, data["cm_sum_var_d1_3d_v_mrg"])
    assert within_tolerance(2.11, 0, data["cm_sum_entr_d1_3d_v_mrg"])
    assert within_tolerance(0.291, 0, data["cm_energy_d1_3d_v_mrg"])
    assert within_tolerance(5.12, 0, data["cm_contrast_d1_3d_v_mrg"])
    assert within_tolerance(1.38, 0, data["cm_dissimilarity_d1_3d_v_mrg"])
    assert within_tolerance(0.688, 0, data["cm_inv_diff_d1_3d_v_mrg"])
    assert within_tolerance(0.856, 0, data["cm_inv_diff_norm_d1_3d_v_mrg"])
    assert within_tolerance(0.631, 0, data["cm_inv_diff_mom_d1_3d_v_mrg"])
    assert within_tolerance(0.902, 0, data["cm_inv_diff_mom_norm_d1_3d_v_mrg"])
    assert within_tolerance(0.0574, 0, data["cm_inv_var_d1_3d_v_mrg"])
    assert within_tolerance(0.183, 0, data["cm_corr_d1_3d_v_mrg"])
    assert within_tolerance(5.19, 0, data["cm_auto_corr_d1_3d_v_mrg"])
    assert within_tolerance(7.41, 0, data["cm_clust_tend_d1_3d_v_mrg"])
    assert within_tolerance(17.4, 0, data["cm_clust_shade_d1_3d_v_mrg"])
    assert within_tolerance(147, 0, data["cm_clust_prom_d1_3d_v_mrg"])
    assert within_tolerance(-0.0288, 0, data["cm_info_corr1_d1_3d_v_mrg"])
    assert within_tolerance(0.269, 0, data["cm_info_corr2_d1_3d_v_mrg"])
    assert within_tolerance(0.641, 0, data["rlm_sre_2d_avg"])
    assert within_tolerance(3.78, 0, data["rlm_lre_2d_avg"])
    assert within_tolerance(0.604, 0, data["rlm_lgre_2d_avg"])
    assert within_tolerance(9.82, 0, data["rlm_hgre_2d_avg"])
    assert within_tolerance(0.294, 0, data["rlm_srlge_2d_avg"])
    assert within_tolerance(8.57, 0, data["rlm_srhge_2d_avg"])
    assert within_tolerance(3.14, 0, data["rlm_lrlge_2d_avg"])
    assert within_tolerance(17.4, 0, data["rlm_lrhge_2d_avg"])
    assert within_tolerance(5.2, 0, data["rlm_glnu_2d_avg"])
    assert within_tolerance(0.46, 0, data["rlm_glnu_norm_2d_avg"])
    assert within_tolerance(6.12, 0, data["rlm_rlnu_2d_avg"])
    assert within_tolerance(0.492, 0, data["rlm_rlnu_norm_2d_avg"])
    assert within_tolerance(0.627, 0, data["rlm_r_perc_2d_avg"])
    assert within_tolerance(3.35, 0, data["rlm_gl_var_2d_avg"])
    assert within_tolerance(0.761, 0, data["rlm_rl_var_2d_avg"])
    assert within_tolerance(2.17, 0, data["rlm_rl_entr_2d_avg"])
    assert within_tolerance(0.661, 0, data["rlm_sre_2d_s_mrg"])
    assert within_tolerance(3.51, 0, data["rlm_lre_2d_s_mrg"])
    assert within_tolerance(0.609, 0, data["rlm_lgre_2d_s_mrg"])
    assert within_tolerance(9.74, 0, data["rlm_hgre_2d_s_mrg"])
    assert within_tolerance(0.311, 0, data["rlm_srlge_2d_s_mrg"])
    assert within_tolerance(8.67, 0, data["rlm_srhge_2d_s_mrg"])
    assert within_tolerance(2.92, 0, data["rlm_lrlge_2d_s_mrg"])
    assert within_tolerance(16.1, 0, data["rlm_lrhge_2d_s_mrg"])
    assert within_tolerance(20.5, 0, data["rlm_glnu_2d_s_mrg"])
    assert within_tolerance(0.456, 0, data["rlm_glnu_norm_2d_s_mrg"])
    assert within_tolerance(21.6, 0, data["rlm_rlnu_2d_s_mrg"])
    assert within_tolerance(0.441, 0, data["rlm_rlnu_norm_2d_s_mrg"])
    assert within_tolerance(0.627, 0, data["rlm_r_perc_2d_s_mrg"])
    assert within_tolerance(3.37, 0, data["rlm_gl_var_2d_s_mrg"])
    assert within_tolerance(0.778, 0, data["rlm_rl_var_2d_s_mrg"])
    assert within_tolerance(2.57, 0, data["rlm_rl_entr_2d_s_mrg"])
    assert within_tolerance(0.665, 0, data["rlm_sre_2.5d_d_mrg"])
    assert within_tolerance(3.46, 0, data["rlm_lre_2.5d_d_mrg"])
    assert within_tolerance(0.58, 0, data["rlm_lgre_2.5d_d_mrg"])
    assert within_tolerance(10.3, 0, data["rlm_hgre_2.5d_d_mrg"])
    assert within_tolerance(0.296, 0, data["rlm_srlge_2.5d_d_mrg"])
    assert within_tolerance(9.03, 0, data["rlm_srhge_2.5d_d_mrg"])
    assert within_tolerance(2.79, 0, data["rlm_lrlge_2.5d_d_mrg"])
    assert within_tolerance(17.9, 0, data["rlm_lrhge_2.5d_d_mrg"])
    assert within_tolerance(19.5, 0, data["rlm_glnu_2.5d_d_mrg"])
    assert within_tolerance(0.413, 0, data["rlm_glnu_norm_2.5d_d_mrg"])
    assert within_tolerance(22.3, 0, data["rlm_rlnu_2.5d_d_mrg"])
    assert within_tolerance(0.461, 0, data["rlm_rlnu_norm_2.5d_d_mrg"])
    assert within_tolerance(0.632, 0, data["rlm_r_perc_2.5d_d_mrg"])
    assert within_tolerance(3.58, 0, data["rlm_gl_var_2.5d_d_mrg"])
    assert within_tolerance(0.758, 0, data["rlm_rl_var_2.5d_d_mrg"])
    assert within_tolerance(2.52, 0, data["rlm_rl_entr_2.5d_d_mrg"])
    assert within_tolerance(0.68, 0, data["rlm_sre_2.5d_v_mrg"])
    assert within_tolerance(3.27, 0, data["rlm_lre_2.5d_v_mrg"])
    assert within_tolerance(0.585, 0, data["rlm_lgre_2.5d_v_mrg"])
    assert within_tolerance(10.2, 0, data["rlm_hgre_2.5d_v_mrg"])
    assert within_tolerance(0.312, 0, data["rlm_srlge_2.5d_v_mrg"])
    assert within_tolerance(9.05, 0, data["rlm_srhge_2.5d_v_mrg"])
    assert within_tolerance(2.63, 0, data["rlm_lrlge_2.5d_v_mrg"])
    assert within_tolerance(17, 0, data["rlm_lrhge_2.5d_v_mrg"])
    assert within_tolerance(77.1, 0, data["rlm_glnu_2.5d_v_mrg"])
    assert within_tolerance(0.412, 0, data["rlm_glnu_norm_2.5d_v_mrg"])
    assert within_tolerance(83.2, 0, data["rlm_rlnu_2.5d_v_mrg"])
    assert within_tolerance(0.445, 0, data["rlm_rlnu_norm_2.5d_v_mrg"])
    assert within_tolerance(0.632, 0, data["rlm_r_perc_2.5d_v_mrg"])
    assert within_tolerance(3.59, 0, data["rlm_gl_var_2.5d_v_mrg"])
    assert within_tolerance(0.767, 0, data["rlm_rl_var_2.5d_v_mrg"])
    assert within_tolerance(2.76, 0, data["rlm_rl_entr_2.5d_v_mrg"])
    assert within_tolerance(0.705, 0, data["rlm_sre_3d_avg"])
    assert within_tolerance(3.06, 0, data["rlm_lre_3d_avg"])
    assert within_tolerance(0.603, 0, data["rlm_lgre_3d_avg"])
    assert within_tolerance(9.7, 0, data["rlm_hgre_3d_avg"])
    assert within_tolerance(0.352, 0, data["rlm_srlge_3d_avg"])
    assert within_tolerance(8.54, 0, data["rlm_srhge_3d_avg"])
    assert within_tolerance(2.39, 0, data["rlm_lrlge_3d_avg"])
    assert within_tolerance(17.6, 0, data["rlm_lrhge_3d_avg"])
    assert within_tolerance(21.8, 0, data["rlm_glnu_3d_avg"])
    assert within_tolerance(0.43, 0, data["rlm_glnu_norm_3d_avg"])
    assert within_tolerance(26.9, 0, data["rlm_rlnu_3d_avg"])
    assert within_tolerance(0.513, 0, data["rlm_rlnu_norm_3d_avg"])
    assert within_tolerance(0.68, 0, data["rlm_r_perc_3d_avg"])
    assert within_tolerance(3.46, 0, data["rlm_gl_var_3d_avg"])
    assert within_tolerance(0.574, 0, data["rlm_rl_var_3d_avg"])
    assert within_tolerance(2.43, 0, data["rlm_rl_entr_3d_avg"])
    assert within_tolerance(0.729, 0, data["rlm_sre_3d_v_mrg"])
    assert within_tolerance(2.76, 0, data["rlm_lre_3d_v_mrg"])
    assert within_tolerance(0.607, 0, data["rlm_lgre_3d_v_mrg"])
    assert within_tolerance(9.64, 0, data["rlm_hgre_3d_v_mrg"])
    assert within_tolerance(0.372, 0, data["rlm_srlge_3d_v_mrg"])
    assert within_tolerance(8.67, 0, data["rlm_srhge_3d_v_mrg"])
    assert within_tolerance(2.16, 0, data["rlm_lrlge_3d_v_mrg"])
    assert within_tolerance(15.6, 0, data["rlm_lrhge_3d_v_mrg"])
    assert within_tolerance(281, 0, data["rlm_glnu_3d_v_mrg"])
    assert within_tolerance(0.43, 0, data["rlm_glnu_norm_3d_v_mrg"])
    assert within_tolerance(328, 0, data["rlm_rlnu_3d_v_mrg"])
    assert within_tolerance(0.501, 0, data["rlm_rlnu_norm_3d_v_mrg"])
    assert within_tolerance(0.68, 0, data["rlm_r_perc_3d_v_mrg"])
    assert within_tolerance(3.48, 0, data["rlm_gl_var_3d_v_mrg"])
    assert within_tolerance(0.598, 0, data["rlm_rl_var_3d_v_mrg"])
    assert within_tolerance(2.62, 0, data["rlm_rl_entr_3d_v_mrg"])
    assert within_tolerance(0.363, 0, data["szm_sze_2d"])
    assert within_tolerance(43.9, 0, data["szm_lze_2d"])
    assert within_tolerance(0.371, 0, data["szm_lgze_2d"])
    assert within_tolerance(16.4, 0, data["szm_hgze_2d"])
    assert within_tolerance(0.0259, 0, data["szm_szlge_2d"])
    assert within_tolerance(10.3, 0, data["szm_szhge_2d"])
    assert within_tolerance(40.4, 0, data["szm_lzlge_2d"])
    assert within_tolerance(113, 0, data["szm_lzhge_2d"])
    assert within_tolerance(1.41, 0, data["szm_glnu_2d"])
    assert within_tolerance(0.323, 0, data["szm_glnu_norm_2d"])
    assert within_tolerance(1.49, 0, data["szm_zsnu_2d"])
    assert within_tolerance(0.333, 0, data["szm_zsnu_norm_2d"])
    assert within_tolerance(0.24, 0, data["szm_z_perc_2d"])
    assert within_tolerance(3.97, 0, data["szm_gl_var_2d"])
    assert within_tolerance(21, 0, data["szm_zs_var_2d"])
    assert within_tolerance(1.93, 0, data["szm_zs_entr_2d"])
    assert within_tolerance(0.368, 0, data["szm_sze_2.5d"])
    assert within_tolerance(34.2, 0, data["szm_lze_2.5d"])
    assert within_tolerance(0.368, 0, data["szm_lgze_2.5d"])
    assert within_tolerance(16.2, 0, data["szm_hgze_2.5d"])
    assert within_tolerance(0.0295, 0, data["szm_szlge_2.5d"])
    assert within_tolerance(9.87, 0, data["szm_szhge_2.5d"])
    assert within_tolerance(30.6, 0, data["szm_lzlge_2.5d"])
    assert within_tolerance(107, 0, data["szm_lzhge_2.5d"])
    assert within_tolerance(5.44, 0, data["szm_glnu_2.5d"])
    assert within_tolerance(0.302, 0, data["szm_glnu_norm_2.5d"])
    assert within_tolerance(3.44, 0, data["szm_zsnu_2.5d"])
    assert within_tolerance(0.191, 0, data["szm_zsnu_norm_2.5d"])
    assert within_tolerance(0.243, 0, data["szm_z_perc_2.5d"])
    assert within_tolerance(3.92, 0, data["szm_gl_var_2.5d"])
    assert within_tolerance(17.3, 0, data["szm_zs_var_2.5d"])
    assert within_tolerance(3.08, 0, data["szm_zs_entr_2.5d"])
    assert within_tolerance(0.255, 0, data["szm_sze_3d"])
    assert within_tolerance(550, 0, data["szm_lze_3d"])
    assert within_tolerance(0.253, 0, data["szm_lgze_3d"])
    assert within_tolerance(15.6, 0, data["szm_hgze_3d"])
    assert within_tolerance(0.0256, 0, data["szm_szlge_3d"])
    assert within_tolerance(2.76, 0, data["szm_szhge_3d"])
    assert within_tolerance(503, 0, data["szm_lzlge_3d"])
    assert within_tolerance(1490, 0, data["szm_lzhge_3d"])
    assert within_tolerance(1.4, 0, data["szm_glnu_3d"])
    assert within_tolerance(0.28, 0, data["szm_glnu_norm_3d"])
    assert within_tolerance(1, 0, data["szm_zsnu_3d"])
    assert within_tolerance(0.2, 0, data["szm_zsnu_norm_3d"])
    assert within_tolerance(0.0676, 0, data["szm_z_perc_3d"])
    assert within_tolerance(2.64, 0, data["szm_gl_var_3d"])
    assert within_tolerance(331, 0, data["szm_zs_var_3d"])
    assert within_tolerance(2.32, 0, data["szm_zs_entr_3d"])
    assert within_tolerance(0.946, 0, data["dzm_sde_2d"])
    assert within_tolerance(1.21, 0, data["dzm_lde_2d"])
    assert within_tolerance(0.371, 0, data["dzm_lgze_2d"])
    assert within_tolerance(16.4, 0, data["dzm_hgze_2d"])
    assert within_tolerance(0.367, 0, data["dzm_sdlge_2d"])
    assert within_tolerance(15.2, 0, data["dzm_sdhge_2d"])
    assert within_tolerance(0.386, 0, data["dzm_ldlge_2d"])
    assert within_tolerance(21.3, 0, data["dzm_ldhge_2d"])
    assert within_tolerance(1.41, 0, data["dzm_glnu_2d"])
    assert within_tolerance(0.323, 0, data["dzm_glnu_norm_2d"])
    assert within_tolerance(3.79, 0, data["dzm_zdnu_2d"])
    assert within_tolerance(0.898, 0, data["dzm_zdnu_norm_2d"])
    assert within_tolerance(0.24, 0, data["dzm_z_perc_2d"])
    assert within_tolerance(3.97, 0, data["dzm_gl_var_2d"])
    assert within_tolerance(0.051, 0, data["dzm_zd_var_2d"])
    assert within_tolerance(1.73, 0, data["dzm_zd_entr_2d"])
    assert within_tolerance(0.917, 0, data["dzm_sde_2.5d"])
    assert within_tolerance(1.33, 0, data["dzm_lde_2.5d"])
    assert within_tolerance(0.368, 0, data["dzm_lgze_2.5d"])
    assert within_tolerance(16.2, 0, data["dzm_hgze_2.5d"])
    assert within_tolerance(0.362, 0, data["dzm_sdlge_2.5d"])
    assert within_tolerance(14.3, 0, data["dzm_sdhge_2.5d"])
    assert within_tolerance(0.391, 0, data["dzm_ldlge_2.5d"])
    assert within_tolerance(23.7, 0, data["dzm_ldhge_2.5d"])
    assert within_tolerance(5.44, 0, data["dzm_glnu_2.5d"])
    assert within_tolerance(0.302, 0, data["dzm_glnu_norm_2.5d"])
    assert within_tolerance(14.4, 0, data["dzm_zdnu_2.5d"])
    assert within_tolerance(0.802, 0, data["dzm_zdnu_norm_2.5d"])
    assert within_tolerance(0.243, 0, data["dzm_z_perc_2.5d"])
    assert within_tolerance(3.92, 0, data["dzm_gl_var_2.5d"])
    assert within_tolerance(0.0988, 0, data["dzm_zd_var_2.5d"])
    assert within_tolerance(2, 0, data["dzm_zd_entr_2.5d"])
    assert within_tolerance(1, 0, data["dzm_sde_3d"])
    assert within_tolerance(1, 0, data["dzm_lde_3d"])
    assert within_tolerance(0.253, 0, data["dzm_lgze_3d"])
    assert within_tolerance(15.6, 0, data["dzm_hgze_3d"])
    assert within_tolerance(0.253, 0, data["dzm_sdlge_3d"])
    assert within_tolerance(15.6, 0, data["dzm_sdhge_3d"])
    assert within_tolerance(0.253, 0, data["dzm_ldlge_3d"])
    assert within_tolerance(15.6, 0, data["dzm_ldhge_3d"])
    assert within_tolerance(1.4, 0, data["dzm_glnu_3d"])
    assert within_tolerance(0.28, 0, data["dzm_glnu_norm_3d"])
    assert within_tolerance(5, 0, data["dzm_zdnu_3d"])
    assert within_tolerance(1, 0, data["dzm_zdnu_norm_3d"])
    assert within_tolerance(0.0676, 0, data["dzm_z_perc_3d"])
    assert within_tolerance(2.64, 0, data["dzm_gl_var_3d"])
    assert within_tolerance(0, 0, data["dzm_zd_var_3d"])
    assert within_tolerance(1.92, 0, data["dzm_zd_entr_3d"])
    assert within_tolerance(0.121, 0, data["ngt_coarseness_2d"])
    assert within_tolerance(0.925, 0, data["ngt_contrast_2d"])
    assert within_tolerance(2.99, 0, data["ngt_busyness_2d"])
    assert within_tolerance(10.4, 0, data["ngt_complexity_2d"])
    assert within_tolerance(2.88, 0, data["ngt_strength_2d"])
    assert within_tolerance(0.0285, 0, data["ngt_coarseness_2.5d"])
    assert within_tolerance(0.601, 0, data["ngt_contrast_2.5d"])
    assert within_tolerance(6.8, 0, data["ngt_busyness_2.5d"])
    assert within_tolerance(14.1, 0, data["ngt_complexity_2.5d"])
    assert within_tolerance(0.741, 0, data["ngt_strength_2.5d"])
    assert within_tolerance(0.0296, 0, data["ngt_coarseness_3d"])
    assert within_tolerance(0.584, 0, data["ngt_contrast_3d"])
    assert within_tolerance(6.54, 0, data["ngt_busyness_3d"])
    assert within_tolerance(13.5, 0, data["ngt_complexity_3d"])
    assert within_tolerance(0.763, 0, data["ngt_strength_3d"])
    assert within_tolerance(0.158, 0, data["ngl_lde_d1_a0.0_2d"])
    assert within_tolerance(19.2, 0, data["ngl_hde_d1_a0.0_2d"])
    assert within_tolerance(0.702, 0, data["ngl_lgce_d1_a0.0_2d"])
    assert within_tolerance(7.49, 0, data["ngl_hgce_d1_a0.0_2d"])
    assert within_tolerance(0.0473, 0, data["ngl_ldlge_d1_a0.0_2d"])
    assert within_tolerance(3.06, 0, data["ngl_ldhge_d1_a0.0_2d"])
    assert within_tolerance(17.6, 0, data["ngl_hdlge_d1_a0.0_2d"])
    assert within_tolerance(49.5, 0, data["ngl_hdhge_d1_a0.0_2d"])
    assert within_tolerance(10.2, 0, data["ngl_glnu_d1_a0.0_2d"])
    assert within_tolerance(0.562, 0, data["ngl_glnu_norm_d1_a0.0_2d"])
    assert within_tolerance(3.96, 0, data["ngl_dcnu_d1_a0.0_2d"])
    assert within_tolerance(0.212, 0, data["ngl_dcnu_norm_d1_a0.0_2d"])
    assert within_tolerance(1, 0, data["ngl_dc_perc_d1_a0.0_2d"])
    assert within_tolerance(2.7, 0, data["ngl_gl_var_d1_a0.0_2d"])
    assert within_tolerance(2.73, 0, data["ngl_dc_var_d1_a0.0_2d"])
    assert within_tolerance(2.71, 0, data["ngl_dc_entr_d1_a0.0_2d"])
    assert within_tolerance(0.17, 0, data["ngl_dc_energy_d1_a0.0_2d"])
    assert within_tolerance(0.159, 0, data["ngl_lde_d1_a0.0_2.5d"])
    assert within_tolerance(18.8, 0, data["ngl_hde_d1_a0.0_2.5d"])
    assert within_tolerance(0.693, 0, data["ngl_lgce_d1_a0.0_2.5d"])
    assert within_tolerance(7.66, 0, data["ngl_hgce_d1_a0.0_2.5d"])
    assert within_tolerance(0.0477, 0, data["ngl_ldlge_d1_a0.0_2.5d"])
    assert within_tolerance(3.07, 0, data["ngl_ldhge_d1_a0.0_2.5d"])
    assert within_tolerance(17.2, 0, data["ngl_hdlge_d1_a0.0_2.5d"])
    assert within_tolerance(50.8, 0, data["ngl_hdhge_d1_a0.0_2.5d"])
    assert within_tolerance(37.9, 0, data["ngl_glnu_d1_a0.0_2.5d"])
    assert within_tolerance(0.512, 0, data["ngl_glnu_norm_d1_a0.0_2.5d"])
    assert within_tolerance(12.4, 0, data["ngl_dcnu_d1_a0.0_2.5d"])
    assert within_tolerance(0.167, 0, data["ngl_dcnu_norm_d1_a0.0_2.5d"])
    assert within_tolerance(1, 0, data["ngl_dc_perc_d1_a0.0_2.5d"])
    assert within_tolerance(3.05, 0, data["ngl_gl_var_d1_a0.0_2.5d"])
    assert within_tolerance(3.27, 0, data["ngl_dc_var_d1_a0.0_2.5d"])
    assert within_tolerance(3.36, 0, data["ngl_dc_entr_d1_a0.0_2.5d"])
    assert within_tolerance(0.122, 0, data["ngl_dc_energy_d1_a0.0_2.5d"])
    assert within_tolerance(0.045, 0, data["ngl_lde_d1_a0.0_3d"])
    assert within_tolerance(109, 0, data["ngl_hde_d1_a0.0_3d"])
    assert within_tolerance(0.693, 0, data["ngl_lgce_d1_a0.0_3d"])
    assert within_tolerance(7.66, 0, data["ngl_hgce_d1_a0.0_3d"])
    assert within_tolerance(0.00963, 0, data["ngl_ldlge_d1_a0.0_3d"])
    assert within_tolerance(0.736, 0, data["ngl_ldhge_d1_a0.0_3d"])
    assert within_tolerance(102, 0, data["ngl_hdlge_d1_a0.0_3d"])
    assert within_tolerance(235, 0, data["ngl_hdhge_d1_a0.0_3d"])
    assert within_tolerance(37.9, 0, data["ngl_glnu_d1_a0.0_3d"])
    assert within_tolerance(0.512, 0, data["ngl_glnu_norm_d1_a0.0_3d"])
    assert within_tolerance(4.86, 0, data["ngl_dcnu_d1_a0.0_3d"])
    assert within_tolerance(0.0657, 0, data["ngl_dcnu_norm_d1_a0.0_3d"])
    assert within_tolerance(1, 0, data["ngl_dc_perc_d1_a0.0_3d"])
    assert within_tolerance(3.05, 0, data["ngl_gl_var_d1_a0.0_3d"])
    assert within_tolerance(22.1, 0, data["ngl_dc_var_d1_a0.0_3d"])
    assert within_tolerance(4.4, 0, data["ngl_dc_entr_d1_a0.0_3d"])
    assert within_tolerance(0.0533, 0, data["ngl_dc_energy_d1_a0.0_3d"])


def test_ibsi_1_chest_config_a():
    """
    Compare computed feature values with reference values for the chest CT image obtained using image processing
    configuration scheme A.
    """

    # Configure settings used for the digital phantom.
    general_settings = GeneralSettingsClass(
        by_slice=True
    )

    image_interpolation_settings = ImageInterpolationSettingsClass(
        by_slice=general_settings.by_slice,
        anti_aliasing=False
    )

    resegmentation_settings = ResegmentationSettingsClass(
        resegmentation_intensity_range=[-500.0, 400.0]
    )

    feature_computation_parameters = FeatureExtractionSettingsClass(
        by_slice=general_settings.by_slice,
        no_approximation=False,
        base_feature_families="all",
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=25.0,
        ivh_discretisation_method="none",
        glcm_distance=1.0,
        glcm_spatial_method=["2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge"],
        glrlm_spatial_method=["2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge"],
        glszm_spatial_method=["2d", "2.5d"],
        gldzm_spatial_method=["2d", "2.5d"],
        ngtdm_spatial_method=["2d", "2.5d"],
        ngldm_distance=1.0,
        ngldm_spatial_method=["2d", "2.5d"],
        ngldm_difference_level=0.0
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
        roi_resegment_settings=resegmentation_settings,
        perturbation_settings=ImagePerturbationSettingsClass(),
        img_transform_settings=image_transformation_settings,
        feature_extr_settings=feature_computation_parameters
    )

    data = extract_features(
        write_features=False,
        export_features=True,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        settings=settings
    )

    data = data[0]

    assert within_tolerance(358000, 5000, data["morph_volume"])
    assert within_tolerance(359000, 5000, data["morph_vol_approx"])
    assert within_tolerance(35700, 300, data["morph_area_mesh"])
    assert within_tolerance(0.0996, 5e-04, data["morph_av"])
    assert within_tolerance(0.03, 1e-04, data["morph_comp_1"])
    assert within_tolerance(0.319, 0.001, data["morph_comp_2"])
    assert within_tolerance(1.46, 0.01, data["morph_sph_dispr"])
    assert within_tolerance(0.683, 0.001, data["morph_sphericity"])
    assert within_tolerance(0.463, 0.002, data["morph_asphericity"])
    assert within_tolerance(52.9, 28.7, data["morph_com"])
    assert within_tolerance(125, 1, data["morph_diam"])
    assert within_tolerance(92.7, 0.4, data["morph_pca_maj_axis"])
    assert within_tolerance(81.5, 0.4, data["morph_pca_min_axis"])
    assert within_tolerance(70.1, 0.3, data["morph_pca_least_axis"])
    assert within_tolerance(0.879, 0.001, data["morph_pca_elongation"])
    assert within_tolerance(0.756, 0.001, data["morph_pca_flatness"])
    assert within_tolerance(0.486, 0.003, data["morph_vol_dens_aabb"])
    assert within_tolerance(0.725, 0.003, data["morph_area_dens_aabb"])
    assert within_tolerance(1.29, 0.01, data["morph_vol_dens_aee"])
    assert within_tolerance(1.71, 0.01, data["morph_area_dens_aee"])
    assert within_tolerance(0.827, 0.001, data["morph_vol_dens_conv_hull"])
    assert within_tolerance(1.18, 0.01, data["morph_area_dens_conv_hull"])
    assert within_tolerance(4810000, 320000, data["morph_integ_int"])
    # assert within_tolerance(0.0322, 2e-04, data["morph_moran_i"])
    # assert within_tolerance(0.863, 0.001, data["morph_geary_c"])
    assert within_tolerance(-277, 10, data["loc_peak_loc"])
    assert within_tolerance(189, 5, data["loc_peak_glob"])
    assert within_tolerance(13.4, 1.1, data["stat_mean"])
    assert within_tolerance(14200, 400, data["stat_var"])
    assert within_tolerance(-2.47, 0.05, data["stat_skew"])
    assert within_tolerance(5.96, 0.24, data["stat_kurt"])
    assert within_tolerance(46, 0.3, data["stat_median"])
    assert within_tolerance(-500, 0, data["stat_min"])
    assert within_tolerance(-129, 8, data["stat_p10"])
    assert within_tolerance(95, 0, data["stat_p90"])
    assert within_tolerance(377, 9, data["stat_max"])
    assert within_tolerance(56, 0.5, data["stat_iqr"])
    assert within_tolerance(877, 9, data["stat_range"])
    assert within_tolerance(73.6, 1.4, data["stat_mad"])
    assert within_tolerance(27.7, 0.8, data["stat_rmad"])
    assert within_tolerance(64.3, 1, data["stat_medad"])
    assert within_tolerance(8.9, 4.98, data["stat_cov"])
    assert within_tolerance(0.636, 0.008, data["stat_qcod"])
    assert within_tolerance(1.65e+09, 2e+07, data["stat_energy"])
    assert within_tolerance(120, 2, data["stat_rms"])
    assert within_tolerance(21.1, 0.1, data["ih_mean_fbs_w25.0"])
    assert within_tolerance(22.8, 0.6, data["ih_var_fbs_w25.0"])
    assert within_tolerance(-2.46, 0.05, data["ih_skew_fbs_w25.0"])
    assert within_tolerance(5.9, 0.24, data["ih_kurt_fbs_w25.0"])
    assert within_tolerance(22, 0, data["ih_median_fbs_w25.0"])
    assert within_tolerance(1, 0, data["ih_min_fbs_w25.0"])
    assert within_tolerance(15, 0.4, data["ih_p10_fbs_w25.0"])
    assert within_tolerance(24, 0, data["ih_p90_fbs_w25.0"])
    assert within_tolerance(36, 0.4, data["ih_max_fbs_w25.0"])
    assert within_tolerance(23, 0, data["ih_mode_fbs_w25.0"])
    assert within_tolerance(2, 0, data["ih_iqr_fbs_w25.0"])
    assert within_tolerance(35, 0.4, data["ih_range_fbs_w25.0"])
    assert within_tolerance(2.94, 0.06, data["ih_mad_fbs_w25.0"])
    assert within_tolerance(1.18, 0.04, data["ih_rmad_fbs_w25.0"])
    assert within_tolerance(2.58, 0.05, data["ih_medad_fbs_w25.0"])
    assert within_tolerance(0.227, 0.004, data["ih_cov_fbs_w25.0"])
    assert within_tolerance(0.0455, 0, data["ih_qcod_fbs_w25.0"])
    assert within_tolerance(3.36, 0.03, data["ih_entropy_fbs_w25.0"])
    assert within_tolerance(0.15, 0.002, data["ih_uniformity_fbs_w25.0"])
    assert within_tolerance(11000, 100, data["ih_max_grad_fbs_w25.0"])
    assert within_tolerance(21, 0, data["ih_max_grad_g_fbs_w25.0"])
    assert within_tolerance(-10100, 100, data["ih_min_grad_fbs_w25.0"])
    assert within_tolerance(24, 0, data["ih_min_grad_g_fbs_w25.0"])
    assert within_tolerance(0.978, 0.001, data["ivh_v10"])
    assert within_tolerance(6.98e-05, 1.03e-05, data["ivh_v90"])
    assert within_tolerance(96, 0, data["ivh_i10"])
    assert within_tolerance(-128, 8, data["ivh_i90"])
    assert within_tolerance(0.978, 0.001, data["ivh_diff_v10_v90"])
    assert within_tolerance(224, 8, data["ivh_diff_i10_i90"])
    assert within_tolerance(0.109, 0.001, data["cm_joint_max_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(20.6, 0.1, data["cm_joint_avg_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(27, 0.4, data["cm_joint_var_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(5.82, 0.04, data["cm_joint_entr_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(1.58, 0.03, data["cm_diff_avg_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(4.94, 0.19, data["cm_diff_var_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(2.27, 0.03, data["cm_diff_entr_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(41.3, 0.1, data["cm_sum_avg_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(100, 1, data["cm_sum_var_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(4.19, 0.03, data["cm_sum_entr_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.045, 8e-04, data["cm_energy_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(7.85, 0.26, data["cm_contrast_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(1.58, 0.03, data["cm_dissimilarity_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.581, 0.003, data["cm_inv_diff_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.961, 0.001, data["cm_inv_diff_norm_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.544, 0.003, data["cm_inv_diff_mom_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.994, 0.001, data["cm_inv_diff_mom_norm_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.441, 0.001, data["cm_inv_var_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.778, 0.002, data["cm_corr_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(455, 2, data["cm_auto_corr_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(100, 1, data["cm_clust_tend_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(-1040, 20, data["cm_clust_shade_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(52700, 500, data["cm_clust_prom_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(-0.236, 0.001, data["cm_info_corr1_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.863, 0.003, data["cm_info_corr2_d1_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.109, 0.001, data["cm_joint_max_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(20.6, 0.1, data["cm_joint_avg_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(27, 0.4, data["cm_joint_var_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(5.9, 0.04, data["cm_joint_entr_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(1.57, 0.03, data["cm_diff_avg_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(4.96, 0.19, data["cm_diff_var_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(2.28, 0.03, data["cm_diff_entr_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(41.3, 0.1, data["cm_sum_avg_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(100, 1, data["cm_sum_var_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(4.21, 0.03, data["cm_sum_entr_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.0446, 8e-04, data["cm_energy_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(7.82, 0.26, data["cm_contrast_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(1.57, 0.03, data["cm_dissimilarity_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.581, 0.003, data["cm_inv_diff_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.961, 0.001, data["cm_inv_diff_norm_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.544, 0.003, data["cm_inv_diff_mom_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.994, 0.001, data["cm_inv_diff_mom_norm_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.441, 0.001, data["cm_inv_var_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.78, 0.002, data["cm_corr_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(455, 2, data["cm_auto_corr_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(100, 1, data["cm_clust_tend_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(-1050, 20, data["cm_clust_shade_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(52800, 500, data["cm_clust_prom_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(-0.214, 0.001, data["cm_info_corr1_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.851, 0.002, data["cm_info_corr2_d1_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.0943, 8e-04, data["cm_joint_max_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(21.3, 0.1, data["cm_joint_avg_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(18.6, 0.5, data["cm_joint_var_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(5.78, 0.04, data["cm_joint_entr_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(1.35, 0.03, data["cm_diff_avg_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(4.12, 0.2, data["cm_diff_var_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(2.16, 0.03, data["cm_diff_entr_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(42.7, 0.1, data["cm_sum_avg_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(68.5, 1.3, data["cm_sum_var_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(4.17, 0.03, data["cm_sum_entr_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.0429, 7e-04, data["cm_energy_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(5.96, 0.27, data["cm_contrast_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(1.35, 0.03, data["cm_dissimilarity_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.605, 0.003, data["cm_inv_diff_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.966, 0.001, data["cm_inv_diff_norm_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.573, 0.003, data["cm_inv_diff_mom_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.996, 0.001, data["cm_inv_diff_mom_norm_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.461, 0.002, data["cm_inv_var_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.839, 0.003, data["cm_corr_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(471, 2, data["cm_auto_corr_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(68.5, 1.3, data["cm_clust_tend_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(-1490, 30, data["cm_clust_shade_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(47600, 700, data["cm_clust_prom_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(-0.231, 0.001, data["cm_info_corr1_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.879, 0.001, data["cm_info_corr2_d1_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.0943, 8e-04, data["cm_joint_max_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(21.3, 0.1, data["cm_joint_avg_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(18.6, 0.5, data["cm_joint_var_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(5.79, 0.04, data["cm_joint_entr_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(1.35, 0.03, data["cm_diff_avg_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(4.14, 0.2, data["cm_diff_var_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(2.16, 0.03, data["cm_diff_entr_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(42.7, 0.1, data["cm_sum_avg_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(68.5, 1.3, data["cm_sum_var_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(4.18, 0.03, data["cm_sum_entr_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.0427, 7e-04, data["cm_energy_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(5.95, 0.27, data["cm_contrast_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(1.35, 0.03, data["cm_dissimilarity_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.605, 0.003, data["cm_inv_diff_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.966, 0.001, data["cm_inv_diff_norm_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.573, 0.003, data["cm_inv_diff_mom_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.996, 0.001, data["cm_inv_diff_mom_norm_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.461, 0.002, data["cm_inv_var_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.84, 0.003, data["cm_corr_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(471, 2, data["cm_auto_corr_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(68.5, 1.3, data["cm_clust_tend_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(-1490, 30, data["cm_clust_shade_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(47700, 700, data["cm_clust_prom_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(-0.228, 0.001, data["cm_info_corr1_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.88, 0.001, data["cm_info_corr2_d1_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.785, 0.003, data["rlm_sre_2d_avg_fbs_w25.0"])
    assert within_tolerance(2.91, 0.03, data["rlm_lre_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.0264, 3e-04, data["rlm_lgre_2d_avg_fbs_w25.0"])
    assert within_tolerance(428, 3, data["rlm_hgre_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.0243, 3e-04, data["rlm_srlge_2d_avg_fbs_w25.0"])
    assert within_tolerance(320, 1, data["rlm_srhge_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.0386, 3e-04, data["rlm_lrlge_2d_avg_fbs_w25.0"])
    assert within_tolerance(1410, 20, data["rlm_lrhge_2d_avg_fbs_w25.0"])
    assert within_tolerance(432, 1, data["rlm_glnu_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.128, 0.003, data["rlm_glnu_norm_2d_avg_fbs_w25.0"])
    assert within_tolerance(1650, 10, data["rlm_rlnu_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.579, 0.003, data["rlm_rlnu_norm_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.704, 0.003, data["rlm_r_perc_2d_avg_fbs_w25.0"])
    assert within_tolerance(33.7, 0.6, data["rlm_gl_var_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.828, 0.008, data["rlm_rl_var_2d_avg_fbs_w25.0"])
    assert within_tolerance(4.73, 0.02, data["rlm_rl_entr_2d_avg_fbs_w25.0"])
    assert within_tolerance(0.786, 0.003, data["rlm_sre_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(2.89, 0.03, data["rlm_lre_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.0264, 3e-04, data["rlm_lgre_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(428, 3, data["rlm_hgre_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.0243, 3e-04, data["rlm_srlge_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(320, 1, data["rlm_srhge_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.0385, 3e-04, data["rlm_lrlge_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(1400, 20, data["rlm_lrhge_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(1730, 10, data["rlm_glnu_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.128, 0.003, data["rlm_glnu_norm_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(6600, 30, data["rlm_rlnu_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.579, 0.003, data["rlm_rlnu_norm_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.704, 0.003, data["rlm_r_perc_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(33.7, 0.6, data["rlm_gl_var_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.826, 0.008, data["rlm_rl_var_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(4.76, 0.02, data["rlm_rl_entr_2d_s_mrg_fbs_w25.0"])
    assert within_tolerance(0.768, 0.003, data["rlm_sre_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(3.09, 0.03, data["rlm_lre_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.0148, 4e-04, data["rlm_lgre_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(449, 3, data["rlm_hgre_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.0135, 4e-04, data["rlm_srlge_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(332, 1, data["rlm_srhge_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.0229, 4e-04, data["rlm_lrlge_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(1500, 20, data["rlm_lrhge_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(9850, 10, data["rlm_glnu_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.126, 0.003, data["rlm_glnu_norm_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(42700, 200, data["rlm_rlnu_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.548, 0.003, data["rlm_rlnu_norm_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.68, 0.003, data["rlm_r_perc_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(29.1, 0.6, data["rlm_gl_var_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.916, 0.011, data["rlm_rl_var_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(4.87, 0.01, data["rlm_rl_entr_2.5d_d_mrg_fbs_w25.0"])
    assert within_tolerance(0.769, 0.003, data["rlm_sre_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(3.08, 0.03, data["rlm_lre_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.0147, 4e-04, data["rlm_lgre_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(449, 3, data["rlm_hgre_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.0135, 4e-04, data["rlm_srlge_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(333, 1, data["rlm_srhge_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.0228, 4e-04, data["rlm_lrlge_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(1500, 20, data["rlm_lrhge_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(39400, 100, data["rlm_glnu_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.126, 0.003, data["rlm_glnu_norm_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(171000, 1000, data["rlm_rlnu_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.548, 0.003, data["rlm_rlnu_norm_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.68, 0.003, data["rlm_r_perc_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(29.1, 0.6, data["rlm_gl_var_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.914, 0.011, data["rlm_rl_var_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(4.87, 0.01, data["rlm_rl_entr_2.5d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.688, 0.003, data["szm_sze_2d_fbs_w25.0"])
    assert within_tolerance(625, 9, data["szm_lze_2d_fbs_w25.0"])
    assert within_tolerance(0.0368, 5e-04, data["szm_lgze_2d_fbs_w25.0"])
    assert within_tolerance(363, 3, data["szm_hgze_2d_fbs_w25.0"])
    assert within_tolerance(0.0298, 5e-04, data["szm_szlge_2d_fbs_w25.0"])
    assert within_tolerance(226, 1, data["szm_szhge_2d_fbs_w25.0"])
    assert within_tolerance(1.35, 0.03, data["szm_lzlge_2d_fbs_w25.0"])
    assert within_tolerance(316000, 5000, data["szm_lzhge_2d_fbs_w25.0"])
    assert within_tolerance(82.2, 0.1, data["szm_glnu_2d_fbs_w25.0"])
    assert within_tolerance(0.0728, 0.0014, data["szm_glnu_norm_2d_fbs_w25.0"])
    assert within_tolerance(479, 4, data["szm_zsnu_2d_fbs_w25.0"])
    assert within_tolerance(0.44, 0.004, data["szm_zsnu_norm_2d_fbs_w25.0"])
    assert within_tolerance(0.3, 0.003, data["szm_z_perc_2d_fbs_w25.0"])
    assert within_tolerance(42.7, 0.7, data["szm_gl_var_2d_fbs_w25.0"])
    assert within_tolerance(609, 9, data["szm_zs_var_2d_fbs_w25.0"])
    assert within_tolerance(5.92, 0.02, data["szm_zs_entr_2d_fbs_w25.0"])
    assert within_tolerance(0.68, 0.003, data["szm_sze_2.5d_fbs_w25.0"])
    assert within_tolerance(675, 8, data["szm_lze_2.5d_fbs_w25.0"])
    assert within_tolerance(0.0291, 5e-04, data["szm_lgze_2.5d_fbs_w25.0"])
    assert within_tolerance(370, 3, data["szm_hgze_2.5d_fbs_w25.0"])
    assert within_tolerance(0.0237, 5e-04, data["szm_szlge_2.5d_fbs_w25.0"])
    assert within_tolerance(229, 1, data["szm_szhge_2.5d_fbs_w25.0"])
    assert within_tolerance(1.44, 0.02, data["szm_lzlge_2.5d_fbs_w25.0"])
    assert within_tolerance(338000, 5000, data["szm_lzhge_2.5d_fbs_w25.0"])
    assert within_tolerance(1800, 10, data["szm_glnu_2.5d_fbs_w25.0"])
    assert within_tolerance(0.0622, 7e-04, data["szm_glnu_norm_2.5d_fbs_w25.0"])
    assert within_tolerance(12400, 100, data["szm_zsnu_2.5d_fbs_w25.0"])
    assert within_tolerance(0.427, 0.004, data["szm_zsnu_norm_2.5d_fbs_w25.0"])
    assert within_tolerance(0.253, 0.004, data["szm_z_perc_2.5d_fbs_w25.0"])
    assert within_tolerance(47.9, 0.4, data["szm_gl_var_2.5d_fbs_w25.0"])
    assert within_tolerance(660, 8, data["szm_zs_var_2.5d_fbs_w25.0"])
    assert within_tolerance(6.39, 0.01, data["szm_zs_entr_2.5d_fbs_w25.0"])
    assert within_tolerance(0.192, 0.006, data["dzm_sde_2d_fbs_w25.0"])
    assert within_tolerance(161, 1, data["dzm_lde_2d_fbs_w25.0"])
    assert within_tolerance(0.0368, 5e-04, data["dzm_lgze_2d_fbs_w25.0"])
    assert within_tolerance(363, 3, data["dzm_hgze_2d_fbs_w25.0"])
    assert within_tolerance(0.00913, 0.00023, data["dzm_sdlge_2d_fbs_w25.0"])
    assert within_tolerance(60.1, 3.3, data["dzm_sdhge_2d_fbs_w25.0"])
    assert within_tolerance(2.96, 0.02, data["dzm_ldlge_2d_fbs_w25.0"])
    assert within_tolerance(70100, 100, data["dzm_ldhge_2d_fbs_w25.0"])
    assert within_tolerance(82.2, 0.1, data["dzm_glnu_2d_fbs_w25.0"])
    assert within_tolerance(0.0728, 0.0014, data["dzm_glnu_norm_2d_fbs_w25.0"])
    assert within_tolerance(64, 0.4, data["dzm_zdnu_2d_fbs_w25.0"])
    assert within_tolerance(0.0716, 0.0022, data["dzm_zdnu_norm_2d_fbs_w25.0"])
    assert within_tolerance(0.3, 0.003, data["dzm_z_perc_2d_fbs_w25.0"])
    assert within_tolerance(42.7, 0.7, data["dzm_gl_var_2d_fbs_w25.0"])
    assert within_tolerance(69.4, 0.1, data["dzm_zd_var_2d_fbs_w25.0"])
    assert within_tolerance(8, 0.04, data["dzm_zd_entr_2d_fbs_w25.0"])
    assert within_tolerance(0.168, 0.005, data["dzm_sde_2.5d_fbs_w25.0"])
    assert within_tolerance(178, 1, data["dzm_lde_2.5d_fbs_w25.0"])
    assert within_tolerance(0.0291, 5e-04, data["dzm_lgze_2.5d_fbs_w25.0"])
    assert within_tolerance(370, 3, data["dzm_hgze_2.5d_fbs_w25.0"])
    assert within_tolerance(0.00788, 0.00022, data["dzm_sdlge_2.5d_fbs_w25.0"])
    assert within_tolerance(49.5, 2.8, data["dzm_sdhge_2.5d_fbs_w25.0"])
    assert within_tolerance(2.31, 0.01, data["dzm_ldlge_2.5d_fbs_w25.0"])
    assert within_tolerance(79500, 100, data["dzm_ldhge_2.5d_fbs_w25.0"])
    assert within_tolerance(1800, 10, data["dzm_glnu_2.5d_fbs_w25.0"])
    assert within_tolerance(0.0622, 7e-04, data["dzm_glnu_norm_2.5d_fbs_w25.0"])
    assert within_tolerance(1570, 10, data["dzm_zdnu_2.5d_fbs_w25.0"])
    assert within_tolerance(0.0543, 0.0014, data["dzm_zdnu_norm_2.5d_fbs_w25.0"])
    assert within_tolerance(0.253, 0.004, data["dzm_z_perc_2.5d_fbs_w25.0"])
    assert within_tolerance(47.9, 0.4, data["dzm_gl_var_2.5d_fbs_w25.0"])
    assert within_tolerance(78.9, 0.1, data["dzm_zd_var_2.5d_fbs_w25.0"])
    assert within_tolerance(8.87, 0.03, data["dzm_zd_entr_2.5d_fbs_w25.0"])
    assert within_tolerance(0.00629, 0.00046, data["ngt_coarseness_2d_fbs_w25.0"])
    assert within_tolerance(0.107, 0.002, data["ngt_contrast_2d_fbs_w25.0"])
    assert within_tolerance(0.489, 0.001, data["ngt_busyness_2d_fbs_w25.0"])
    assert within_tolerance(438, 9, data["ngt_complexity_2d_fbs_w25.0"])
    assert within_tolerance(3.33, 0.08, data["ngt_strength_2d_fbs_w25.0"])
    assert within_tolerance(9.06e-05, 3.3e-06, data["ngt_coarseness_2.5d_fbs_w25.0"])
    assert within_tolerance(0.0345, 9e-04, data["ngt_contrast_2.5d_fbs_w25.0"])
    assert within_tolerance(8.84, 0.01, data["ngt_busyness_2.5d_fbs_w25.0"])
    assert within_tolerance(580, 19, data["ngt_complexity_2.5d_fbs_w25.0"])
    assert within_tolerance(0.0904, 0.0027, data["ngt_strength_2.5d_fbs_w25.0"])
    assert within_tolerance(0.281, 0.003, data["ngl_lde_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(14.8, 0.1, data["ngl_hde_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(0.0233, 3e-04, data["ngl_lgce_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(446, 2, data["ngl_hgce_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(0.0137, 2e-04, data["ngl_ldlge_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(94.2, 0.4, data["ngl_ldhge_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(0.116, 0.001, data["ngl_hdlge_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(7540, 60, data["ngl_hdhge_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(757, 1, data["ngl_glnu_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(0.151, 0.003, data["ngl_glnu_norm_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(709, 2, data["ngl_dcnu_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(0.175, 0.001, data["ngl_dcnu_norm_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(1, 0, data["ngl_dc_perc_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(31.1, 0.5, data["ngl_gl_var_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(3.12, 0.02, data["ngl_dc_var_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(5.76, 0.02, data["ngl_dc_entr_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(0.0268, 4e-04, data["ngl_dc_energy_d1_a0.0_2d_fbs_w25.0"])
    assert within_tolerance(0.243, 0.004, data["ngl_lde_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(16.1, 0.2, data["ngl_hde_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(0.0115, 3e-04, data["ngl_lgce_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(466, 2, data["ngl_hgce_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(0.00664, 2e-04, data["ngl_ldlge_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(91.9, 0.5, data["ngl_ldhge_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(0.0674, 4e-04, data["ngl_hdlge_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(8100, 60, data["ngl_hdhge_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(17200, 100, data["ngl_glnu_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(0.15, 0.002, data["ngl_glnu_norm_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(17500, 100, data["ngl_dcnu_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(0.153, 0.001, data["ngl_dcnu_norm_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(1, 0, data["ngl_dc_perc_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(22.8, 0.6, data["ngl_gl_var_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(3.37, 0.01, data["ngl_dc_var_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(5.93, 0.02, data["ngl_dc_entr_d1_a0.0_2.5d_fbs_w25.0"])
    assert within_tolerance(0.0245, 3e-04, data["ngl_dc_energy_d1_a0.0_2.5d_fbs_w25.0"])


def test_ibsi_1_chest_config_b():
    """
    Compare computed feature values with reference values for the chest CT image obtained using image processing
    configuration scheme B.
    """

    # Configure settings used for the digital phantom.
    general_settings = GeneralSettingsClass(
        by_slice=True
    )

    image_interpolation_settings = ImageInterpolationSettingsClass(
        by_slice=general_settings.by_slice,
        spline_order=1,
        new_spacing=2.0,
        anti_aliasing=False
    )

    resegmentation_settings = ResegmentationSettingsClass(
        resegmentation_intensity_range=[-500.0, 400.0]
    )

    feature_computation_parameters = FeatureExtractionSettingsClass(
        by_slice=general_settings.by_slice,
        no_approximation=False,
        base_feature_families="all",
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=32,
        ivh_discretisation_method="none",
        glcm_distance=1.0,
        glcm_spatial_method=["2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge"],
        glrlm_spatial_method=["2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge"],
        glszm_spatial_method=["2d", "2.5d"],
        gldzm_spatial_method=["2d", "2.5d"],
        ngtdm_spatial_method=["2d", "2.5d"],
        ngldm_distance=1.0,
        ngldm_spatial_method=["2d", "2.5d"],
        ngldm_difference_level=0.0
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
        roi_resegment_settings=resegmentation_settings,
        perturbation_settings=ImagePerturbationSettingsClass(),
        img_transform_settings=image_transformation_settings,
        feature_extr_settings=feature_computation_parameters
    )

    data = extract_features(
        write_features=False,
        export_features=True,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        settings=settings
    )

    data = data[0]

    assert within_tolerance(358000, 5000, data["morph_volume"])
    assert within_tolerance(358000, 5000, data["morph_vol_approx"])
    assert within_tolerance(33700, 300, data["morph_area_mesh"])
    assert within_tolerance(0.0944, 5e-04, data["morph_av"])
    assert within_tolerance(0.0326, 1e-04, data["morph_comp_1"])
    assert within_tolerance(0.377, 0.001, data["morph_comp_2"])
    assert within_tolerance(1.38, 0.01, data["morph_sph_dispr"])
    assert within_tolerance(0.722, 0.001, data["morph_sphericity"])
    assert within_tolerance(0.385, 0.001, data["morph_asphericity"])
    assert within_tolerance(63.1, 29.6, data["morph_com"])
    assert within_tolerance(125, 1, data["morph_diam"])
    assert within_tolerance(92.6, 0.4, data["morph_pca_maj_axis"])
    assert within_tolerance(81.3, 0.4, data["morph_pca_min_axis"])
    assert within_tolerance(70.2, 0.3, data["morph_pca_least_axis"])
    assert within_tolerance(0.878, 0.001, data["morph_pca_elongation"])
    assert within_tolerance(0.758, 0.001, data["morph_pca_flatness"])
    assert within_tolerance(0.477, 0.003, data["morph_vol_dens_aabb"])
    assert within_tolerance(0.678, 0.003, data["morph_area_dens_aabb"])
    assert within_tolerance(1.29, 0.01, data["morph_vol_dens_aee"])
    assert within_tolerance(1.62, 0.01, data["morph_area_dens_aee"])
    assert within_tolerance(0.829, 0.001, data["morph_vol_dens_conv_hull"])
    assert within_tolerance(1.12, 0.01, data["morph_area_dens_conv_hull"])
    assert within_tolerance(4120000, 320000, data["morph_integ_int"])
    # assert within_tolerance(0.0329, 1e-04, data["morph_moran_i"])
    # assert within_tolerance(0.862, 0.001, data["morph_geary_c"])
    assert within_tolerance(178, 10, data["loc_peak_loc"])
    assert within_tolerance(178, 5, data["loc_peak_glob"])
    assert within_tolerance(11.5, 1.1, data["stat_mean"])
    assert within_tolerance(14400, 400, data["stat_var"])
    assert within_tolerance(-2.49, 0.05, data["stat_skew"])
    assert within_tolerance(5.93, 0.24, data["stat_kurt"])
    assert within_tolerance(45, 0.3, data["stat_median"])
    assert within_tolerance(-500, 0, data["stat_min"])
    assert within_tolerance(-136, 8, data["stat_p10"])
    assert within_tolerance(91, 0, data["stat_p90"])
    assert within_tolerance(391, 9, data["stat_max"])
    assert within_tolerance(52, 0.5, data["stat_iqr"])
    assert within_tolerance(891, 9, data["stat_range"])
    assert within_tolerance(74.4, 1.4, data["stat_mad"])
    assert within_tolerance(27.3, 0.8, data["stat_rmad"])
    assert within_tolerance(63.8, 1, data["stat_medad"])
    assert within_tolerance(10.4, 5.2, data["stat_cov"])
    assert within_tolerance(0.591, 0.008, data["stat_qcod"])
    assert within_tolerance(3.98e+08, 1.1e+07, data["stat_energy"])
    assert within_tolerance(121, 2, data["stat_rms"])
    assert within_tolerance(18.9, 0.3, data["ih_mean_fbn_n32"])
    assert within_tolerance(18.7, 0.2, data["ih_var_fbn_n32"])
    assert within_tolerance(-2.47, 0.05, data["ih_skew_fbn_n32"])
    assert within_tolerance(5.84, 0.24, data["ih_kurt_fbn_n32"])
    assert within_tolerance(20, 0.3, data["ih_median_fbn_n32"])
    assert within_tolerance(1, 0, data["ih_min_fbn_n32"])
    assert within_tolerance(14, 0.5, data["ih_p10_fbn_n32"])
    assert within_tolerance(22, 0.3, data["ih_p90_fbn_n32"])
    assert within_tolerance(32, 0, data["ih_max_fbn_n32"])
    assert within_tolerance(20, 0.3, data["ih_mode_fbn_n32"])
    assert within_tolerance(2, 0, data["ih_iqr_fbn_n32"])
    assert within_tolerance(31, 0, data["ih_range_fbn_n32"])
    assert within_tolerance(2.67, 0.03, data["ih_mad_fbn_n32"])
    assert within_tolerance(1.03, 0.03, data["ih_rmad_fbn_n32"])
    assert within_tolerance(2.28, 0.02, data["ih_medad_fbn_n32"])
    assert within_tolerance(0.229, 0.004, data["ih_cov_fbn_n32"])
    assert within_tolerance(0.05, 5e-04, data["ih_qcod_fbn_n32"])
    assert within_tolerance(3.16, 0.01, data["ih_entropy_fbn_n32"])
    assert within_tolerance(0.174, 0.001, data["ih_uniformity_fbn_n32"])
    assert within_tolerance(3220, 50, data["ih_max_grad_fbn_n32"])
    assert within_tolerance(19, 0.3, data["ih_max_grad_g_fbn_n32"])
    assert within_tolerance(-3020, 50, data["ih_min_grad_fbn_n32"])
    assert within_tolerance(22, 0.3, data["ih_min_grad_g_fbn_n32"])
    assert within_tolerance(0.977, 0.001, data["ivh_v10"])
    assert within_tolerance(7.31e-05, 1.03e-05, data["ivh_v90"])
    assert within_tolerance(92, 0, data["ivh_i10"])
    assert within_tolerance(-135, 8, data["ivh_i90"])
    assert within_tolerance(0.977, 0.001, data["ivh_diff_v10_v90"])
    assert within_tolerance(227, 8, data["ivh_diff_i10_i90"])
    assert within_tolerance(0.156, 0.002, data["cm_joint_max_d1_2d_avg_fbn_n32"])
    assert within_tolerance(18.7, 0.3, data["cm_joint_avg_d1_2d_avg_fbn_n32"])
    assert within_tolerance(21, 0.3, data["cm_joint_var_d1_2d_avg_fbn_n32"])
    assert within_tolerance(5.26, 0.02, data["cm_joint_entr_d1_2d_avg_fbn_n32"])
    assert within_tolerance(1.81, 0.01, data["cm_diff_avg_d1_2d_avg_fbn_n32"])
    assert within_tolerance(7.74, 0.05, data["cm_diff_var_d1_2d_avg_fbn_n32"])
    assert within_tolerance(2.35, 0.01, data["cm_diff_entr_d1_2d_avg_fbn_n32"])
    assert within_tolerance(37.4, 0.5, data["cm_sum_avg_d1_2d_avg_fbn_n32"])
    assert within_tolerance(72.1, 1, data["cm_sum_var_d1_2d_avg_fbn_n32"])
    assert within_tolerance(3.83, 0.01, data["cm_sum_entr_d1_2d_avg_fbn_n32"])
    assert within_tolerance(0.0678, 6e-04, data["cm_energy_d1_2d_avg_fbn_n32"])
    assert within_tolerance(11.9, 0.1, data["cm_contrast_d1_2d_avg_fbn_n32"])
    assert within_tolerance(1.81, 0.01, data["cm_dissimilarity_d1_2d_avg_fbn_n32"])
    assert within_tolerance(0.592, 0.001, data["cm_inv_diff_d1_2d_avg_fbn_n32"])
    assert within_tolerance(0.952, 0.001, data["cm_inv_diff_norm_d1_2d_avg_fbn_n32"])
    assert within_tolerance(0.557, 0.001, data["cm_inv_diff_mom_d1_2d_avg_fbn_n32"])
    assert within_tolerance(0.99, 0.001, data["cm_inv_diff_mom_norm_d1_2d_avg_fbn_n32"])
    assert within_tolerance(0.401, 0.002, data["cm_inv_var_d1_2d_avg_fbn_n32"])
    assert within_tolerance(0.577, 0.002, data["cm_corr_d1_2d_avg_fbn_n32"])
    assert within_tolerance(369, 11, data["cm_auto_corr_d1_2d_avg_fbn_n32"])
    assert within_tolerance(72.1, 1, data["cm_clust_tend_d1_2d_avg_fbn_n32"])
    assert within_tolerance(-668, 17, data["cm_clust_shade_d1_2d_avg_fbn_n32"])
    assert within_tolerance(29400, 1400, data["cm_clust_prom_d1_2d_avg_fbn_n32"])
    assert within_tolerance(-0.239, 0.001, data["cm_info_corr1_d1_2d_avg_fbn_n32"])
    assert within_tolerance(0.837, 0.001, data["cm_info_corr2_d1_2d_avg_fbn_n32"])
    assert within_tolerance(0.156, 0.002, data["cm_joint_max_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(18.7, 0.3, data["cm_joint_avg_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(21, 0.3, data["cm_joint_var_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(5.45, 0.01, data["cm_joint_entr_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(1.81, 0.01, data["cm_diff_avg_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(7.76, 0.05, data["cm_diff_var_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(2.38, 0.01, data["cm_diff_entr_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(37.4, 0.5, data["cm_sum_avg_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(72.3, 1, data["cm_sum_var_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(3.89, 0.01, data["cm_sum_entr_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.0669, 6e-04, data["cm_energy_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(11.8, 0.1, data["cm_contrast_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(1.81, 0.01, data["cm_dissimilarity_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.593, 0.001, data["cm_inv_diff_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.952, 0.001, data["cm_inv_diff_norm_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.558, 0.001, data["cm_inv_diff_mom_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.99, 0.001, data["cm_inv_diff_mom_norm_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.401, 0.002, data["cm_inv_var_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.58, 0.002, data["cm_corr_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(369, 11, data["cm_auto_corr_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(72.3, 1, data["cm_clust_tend_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(-673, 17, data["cm_clust_shade_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(29500, 1400, data["cm_clust_prom_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(-0.181, 0.001, data["cm_info_corr1_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.792, 0.001, data["cm_info_corr2_d1_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.126, 0.002, data["cm_joint_max_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(19.2, 0.3, data["cm_joint_avg_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(14.2, 0.1, data["cm_joint_var_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(5.45, 0.01, data["cm_joint_entr_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(1.47, 0.01, data["cm_diff_avg_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(6.48, 0.06, data["cm_diff_var_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(2.24, 0.01, data["cm_diff_entr_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(38.5, 0.6, data["cm_sum_avg_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(48.1, 0.4, data["cm_sum_var_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(3.91, 0.01, data["cm_sum_entr_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.0581, 6e-04, data["cm_energy_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(8.66, 0.09, data["cm_contrast_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(1.47, 0.01, data["cm_dissimilarity_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.628, 0.001, data["cm_inv_diff_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.96, 0.001, data["cm_inv_diff_norm_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.6, 0.001, data["cm_inv_diff_mom_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.992, 0.001, data["cm_inv_diff_mom_norm_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.424, 0.003, data["cm_inv_var_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.693, 0.003, data["cm_corr_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(380, 11, data["cm_auto_corr_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(48.1, 0.4, data["cm_clust_tend_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(-905, 19, data["cm_clust_shade_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(25200, 1000, data["cm_clust_prom_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(-0.188, 0.001, data["cm_info_corr1_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.821, 0.001, data["cm_info_corr2_d1_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.126, 0.002, data["cm_joint_max_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(19.2, 0.3, data["cm_joint_avg_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(14.2, 0.1, data["cm_joint_var_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(5.46, 0.01, data["cm_joint_entr_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(1.47, 0.01, data["cm_diff_avg_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(6.48, 0.06, data["cm_diff_var_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(2.24, 0.01, data["cm_diff_entr_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(38.5, 0.6, data["cm_sum_avg_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(48.1, 0.4, data["cm_sum_var_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(3.91, 0.01, data["cm_sum_entr_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.058, 6e-04, data["cm_energy_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(8.65, 0.09, data["cm_contrast_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(1.47, 0.01, data["cm_dissimilarity_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.628, 0.001, data["cm_inv_diff_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.96, 0.001, data["cm_inv_diff_norm_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.6, 0.001, data["cm_inv_diff_mom_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.992, 0.001, data["cm_inv_diff_mom_norm_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.424, 0.003, data["cm_inv_var_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.695, 0.003, data["cm_corr_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(380, 11, data["cm_auto_corr_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(48.1, 0.4, data["cm_clust_tend_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(-906, 19, data["cm_clust_shade_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(25300, 1000, data["cm_clust_prom_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(-0.185, 0.001, data["cm_info_corr1_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.819, 0.001, data["cm_info_corr2_d1_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.781, 0.001, data["rlm_sre_2d_avg_fbn_n32"])
    assert within_tolerance(3.52, 0.04, data["rlm_lre_2d_avg_fbn_n32"])
    assert within_tolerance(0.0331, 6e-04, data["rlm_lgre_2d_avg_fbn_n32"])
    assert within_tolerance(342, 11, data["rlm_hgre_2d_avg_fbn_n32"])
    assert within_tolerance(0.0314, 6e-04, data["rlm_srlge_2d_avg_fbn_n32"])
    assert within_tolerance(251, 8, data["rlm_srhge_2d_avg_fbn_n32"])
    assert within_tolerance(0.0443, 8e-04, data["rlm_lrlge_2d_avg_fbn_n32"])
    assert within_tolerance(1390, 30, data["rlm_lrhge_2d_avg_fbn_n32"])
    assert within_tolerance(107, 1, data["rlm_glnu_2d_avg_fbn_n32"])
    assert within_tolerance(0.145, 0.001, data["rlm_glnu_norm_2d_avg_fbn_n32"])
    assert within_tolerance(365, 3, data["rlm_rlnu_2d_avg_fbn_n32"])
    assert within_tolerance(0.578, 0.001, data["rlm_rlnu_norm_2d_avg_fbn_n32"])
    assert within_tolerance(0.681, 0.002, data["rlm_r_perc_2d_avg_fbn_n32"])
    assert within_tolerance(28.3, 0.3, data["rlm_gl_var_2d_avg_fbn_n32"])
    assert within_tolerance(1.22, 0.03, data["rlm_rl_var_2d_avg_fbn_n32"])
    assert within_tolerance(4.53, 0.02, data["rlm_rl_entr_2d_avg_fbn_n32"])
    assert within_tolerance(0.782, 0.001, data["rlm_sre_2d_s_mrg_fbn_n32"])
    assert within_tolerance(3.5, 0.04, data["rlm_lre_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.033, 6e-04, data["rlm_lgre_2d_s_mrg_fbn_n32"])
    assert within_tolerance(342, 11, data["rlm_hgre_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.0313, 6e-04, data["rlm_srlge_2d_s_mrg_fbn_n32"])
    assert within_tolerance(252, 8, data["rlm_srhge_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.0442, 8e-04, data["rlm_lrlge_2d_s_mrg_fbn_n32"])
    assert within_tolerance(1380, 30, data["rlm_lrhge_2d_s_mrg_fbn_n32"])
    assert within_tolerance(427, 1, data["rlm_glnu_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.145, 0.001, data["rlm_glnu_norm_2d_s_mrg_fbn_n32"])
    assert within_tolerance(1460, 10, data["rlm_rlnu_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.578, 0.001, data["rlm_rlnu_norm_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.681, 0.002, data["rlm_r_perc_2d_s_mrg_fbn_n32"])
    assert within_tolerance(28.3, 0.3, data["rlm_gl_var_2d_s_mrg_fbn_n32"])
    assert within_tolerance(1.21, 0.03, data["rlm_rl_var_2d_s_mrg_fbn_n32"])
    assert within_tolerance(4.58, 0.01, data["rlm_rl_entr_2d_s_mrg_fbn_n32"])
    assert within_tolerance(0.759, 0.001, data["rlm_sre_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(3.82, 0.05, data["rlm_lre_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.0194, 6e-04, data["rlm_lgre_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(356, 11, data["rlm_hgre_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.0181, 6e-04, data["rlm_srlge_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(257, 9, data["rlm_srhge_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.0293, 9e-04, data["rlm_lrlge_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(1500, 30, data["rlm_lrhge_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(2400, 10, data["rlm_glnu_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.137, 0.001, data["rlm_glnu_norm_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(9380, 70, data["rlm_rlnu_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.533, 0.001, data["rlm_rlnu_norm_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.642, 0.002, data["rlm_r_perc_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(25.7, 0.2, data["rlm_gl_var_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(1.39, 0.03, data["rlm_rl_var_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(4.84, 0.01, data["rlm_rl_entr_2.5d_d_mrg_fbn_n32"])
    assert within_tolerance(0.759, 0.001, data["rlm_sre_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(3.81, 0.05, data["rlm_lre_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.0194, 6e-04, data["rlm_lgre_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(356, 11, data["rlm_hgre_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.0181, 6e-04, data["rlm_srlge_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(258, 9, data["rlm_srhge_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.0292, 9e-04, data["rlm_lrlge_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(1500, 30, data["rlm_lrhge_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(9600, 20, data["rlm_glnu_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.137, 0.001, data["rlm_glnu_norm_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(37500, 300, data["rlm_rlnu_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.534, 0.001, data["rlm_rlnu_norm_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.642, 0.002, data["rlm_r_perc_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(25.7, 0.2, data["rlm_gl_var_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(1.39, 0.03, data["rlm_rl_var_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(4.84, 0.01, data["rlm_rl_entr_2.5d_v_mrg_fbn_n32"])
    assert within_tolerance(0.745, 0.003, data["szm_sze_2d_fbn_n32"])
    assert within_tolerance(439, 8, data["szm_lze_2d_fbn_n32"])
    assert within_tolerance(0.0475, 0.001, data["szm_lgze_2d_fbn_n32"])
    assert within_tolerance(284, 11, data["szm_hgze_2d_fbn_n32"])
    assert within_tolerance(0.0415, 8e-04, data["szm_szlge_2d_fbn_n32"])
    assert within_tolerance(190, 7, data["szm_szhge_2d_fbn_n32"])
    assert within_tolerance(1.15, 0.04, data["szm_lzlge_2d_fbn_n32"])
    assert within_tolerance(181000, 3000, data["szm_lzhge_2d_fbn_n32"])
    assert within_tolerance(20.5, 0.1, data["szm_glnu_2d_fbn_n32"])
    assert within_tolerance(0.0789, 0.001, data["szm_glnu_norm_2d_fbn_n32"])
    assert within_tolerance(140, 3, data["szm_zsnu_2d_fbn_n32"])
    assert within_tolerance(0.521, 0.004, data["szm_zsnu_norm_2d_fbn_n32"])
    assert within_tolerance(0.324, 0.001, data["szm_z_perc_2d_fbn_n32"])
    assert within_tolerance(36.1, 0.3, data["szm_gl_var_2d_fbn_n32"])
    assert within_tolerance(423, 8, data["szm_zs_var_2d_fbn_n32"])
    assert within_tolerance(5.29, 0.01, data["szm_zs_entr_2d_fbn_n32"])
    assert within_tolerance(0.741, 0.003, data["szm_sze_2.5d_fbn_n32"])
    assert within_tolerance(444, 8, data["szm_lze_2.5d_fbn_n32"])
    assert within_tolerance(0.0387, 0.001, data["szm_lgze_2.5d_fbn_n32"])
    assert within_tolerance(284, 11, data["szm_hgze_2.5d_fbn_n32"])
    assert within_tolerance(0.0335, 9e-04, data["szm_szlge_2.5d_fbn_n32"])
    assert within_tolerance(190, 7, data["szm_szhge_2.5d_fbn_n32"])
    assert within_tolerance(1.16, 0.04, data["szm_lzlge_2.5d_fbn_n32"])
    assert within_tolerance(181000, 3000, data["szm_lzhge_2.5d_fbn_n32"])
    assert within_tolerance(437, 3, data["szm_glnu_2.5d_fbn_n32"])
    assert within_tolerance(0.0613, 5e-04, data["szm_glnu_norm_2.5d_fbn_n32"])
    assert within_tolerance(3630, 70, data["szm_zsnu_2.5d_fbn_n32"])
    assert within_tolerance(0.509, 0.004, data["szm_zsnu_norm_2.5d_fbn_n32"])
    assert within_tolerance(0.26, 0.002, data["szm_z_perc_2.5d_fbn_n32"])
    assert within_tolerance(41, 0.7, data["szm_gl_var_2.5d_fbn_n32"])
    assert within_tolerance(429, 8, data["szm_zs_var_2.5d_fbn_n32"])
    assert within_tolerance(5.98, 0.02, data["szm_zs_entr_2.5d_fbn_n32"])
    assert within_tolerance(0.36, 0.005, data["dzm_sde_2d_fbn_n32"])
    assert within_tolerance(31.6, 0.2, data["dzm_lde_2d_fbn_n32"])
    assert within_tolerance(0.0475, 0.001, data["dzm_lgze_2d_fbn_n32"])
    assert within_tolerance(284, 11, data["dzm_hgze_2d_fbn_n32"])
    assert within_tolerance(0.0192, 5e-04, data["dzm_sdlge_2d_fbn_n32"])
    assert within_tolerance(95.7, 5.5, data["dzm_sdhge_2d_fbn_n32"])
    assert within_tolerance(0.934, 0.018, data["dzm_ldlge_2d_fbn_n32"])
    assert within_tolerance(10600, 300, data["dzm_ldhge_2d_fbn_n32"])
    assert within_tolerance(20.5, 0.1, data["dzm_glnu_2d_fbn_n32"])
    assert within_tolerance(0.0789, 0.001, data["dzm_glnu_norm_2d_fbn_n32"])
    assert within_tolerance(39.8, 0.3, data["dzm_zdnu_2d_fbn_n32"])
    assert within_tolerance(0.174, 0.003, data["dzm_zdnu_norm_2d_fbn_n32"])
    assert within_tolerance(0.324, 0.001, data["dzm_z_perc_2d_fbn_n32"])
    assert within_tolerance(36.1, 0.3, data["dzm_gl_var_2d_fbn_n32"])
    assert within_tolerance(13.5, 0.1, data["dzm_zd_var_2d_fbn_n32"])
    assert within_tolerance(6.47, 0.03, data["dzm_zd_entr_2d_fbn_n32"])
    assert within_tolerance(0.329, 0.004, data["dzm_sde_2.5d_fbn_n32"])
    assert within_tolerance(34.3, 0.2, data["dzm_lde_2.5d_fbn_n32"])
    assert within_tolerance(0.0387, 0.001, data["dzm_lgze_2.5d_fbn_n32"])
    assert within_tolerance(284, 11, data["dzm_hgze_2.5d_fbn_n32"])
    assert within_tolerance(0.0168, 5e-04, data["dzm_sdlge_2.5d_fbn_n32"])
    assert within_tolerance(81.4, 4.6, data["dzm_sdhge_2.5d_fbn_n32"])
    assert within_tolerance(0.748, 0.017, data["dzm_ldlge_2.5d_fbn_n32"])
    assert within_tolerance(11600, 400, data["dzm_ldhge_2.5d_fbn_n32"])
    assert within_tolerance(437, 3, data["dzm_glnu_2.5d_fbn_n32"])
    assert within_tolerance(0.0613, 5e-04, data["dzm_glnu_norm_2.5d_fbn_n32"])
    assert within_tolerance(963, 6, data["dzm_zdnu_2.5d_fbn_n32"])
    assert within_tolerance(0.135, 0.001, data["dzm_zdnu_norm_2.5d_fbn_n32"])
    assert within_tolerance(0.26, 0.002, data["dzm_z_perc_2.5d_fbn_n32"])
    assert within_tolerance(41, 0.7, data["dzm_gl_var_2.5d_fbn_n32"])
    assert within_tolerance(15, 0.1, data["dzm_zd_var_2.5d_fbn_n32"])
    assert within_tolerance(7.58, 0.01, data["dzm_zd_entr_2.5d_fbn_n32"])
    assert within_tolerance(0.0168, 5e-04, data["ngt_coarseness_2d_fbn_n32"])
    assert within_tolerance(0.181, 0.001, data["ngt_contrast_2d_fbn_n32"])
    assert within_tolerance(0.2, 0.005, data["ngt_busyness_2d_fbn_n32"])
    assert within_tolerance(391, 7, data["ngt_complexity_2d_fbn_n32"])
    assert within_tolerance(6.02, 0.23, data["ngt_strength_2d_fbn_n32"])
    assert within_tolerance(0.000314, 4e-06, data["ngt_coarseness_2.5d_fbn_n32"])
    assert within_tolerance(0.0506, 5e-04, data["ngt_contrast_2.5d_fbn_n32"])
    assert within_tolerance(3.45, 0.07, data["ngt_busyness_2.5d_fbn_n32"])
    assert within_tolerance(496, 5, data["ngt_complexity_2.5d_fbn_n32"])
    assert within_tolerance(0.199, 0.009, data["ngt_strength_2.5d_fbn_n32"])
    assert within_tolerance(0.31, 0.001, data["ngl_lde_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(17.3, 0.2, data["ngl_hde_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(0.0286, 4e-04, data["ngl_lgce_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(359, 10, data["ngl_hgce_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(0.0203, 3e-04, data["ngl_ldlge_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(78.9, 2.2, data["ngl_ldhge_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(0.108, 0.003, data["ngl_hdlge_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(7210, 130, data["ngl_hdhge_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(216, 3, data["ngl_glnu_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(0.184, 0.001, data["ngl_glnu_norm_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(157, 1, data["ngl_dcnu_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(0.179, 0.001, data["ngl_dcnu_norm_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(1, 0, data["ngl_dc_perc_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(25.3, 0.4, data["ngl_gl_var_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(4.02, 0.05, data["ngl_dc_var_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(5.38, 0.01, data["ngl_dc_entr_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(0.0321, 2e-04, data["ngl_dc_energy_d1_a0.0_2d_fbn_n32"])
    assert within_tolerance(0.254, 0.002, data["ngl_lde_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(19.6, 0.2, data["ngl_hde_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(0.0139, 5e-04, data["ngl_lgce_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(375, 11, data["ngl_hgce_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(0.00929, 0.00026, data["ngl_ldlge_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(73.4, 2.1, data["ngl_ldhge_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(0.077, 0.0019, data["ngl_hdlge_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(7970, 150, data["ngl_hdhge_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(4760, 50, data["ngl_glnu_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(0.174, 0.001, data["ngl_glnu_norm_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(3710, 30, data["ngl_dcnu_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(0.136, 0.001, data["ngl_dcnu_norm_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(1, 0, data["ngl_dc_perc_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(18.7, 0.2, data["ngl_gl_var_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(4.63, 0.06, data["ngl_dc_var_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(5.78, 0.01, data["ngl_dc_entr_d1_a0.0_2.5d_fbn_n32"])
    assert within_tolerance(0.0253, 1e-04, data["ngl_dc_energy_d1_a0.0_2.5d_fbn_n32"])


def test_ibsi_1_chest_config_c():
    """
    Compare computed feature values with reference values for the chest CT image obtained using image processing
    configuration scheme C.
    """

    # Configure settings used for the digital phantom.
    general_settings = GeneralSettingsClass(
        by_slice=False
    )

    image_interpolation_settings = ImageInterpolationSettingsClass(
        by_slice=general_settings.by_slice,
        spline_order=1,
        new_spacing=2.0,
        anti_aliasing=False
    )

    resegmentation_settings = ResegmentationSettingsClass(
        resegmentation_intensity_range=[-1000.0, 400.0]
    )

    feature_computation_parameters = FeatureExtractionSettingsClass(
        by_slice=general_settings.by_slice,
        no_approximation=False,
        base_feature_families="all",
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=25.0,
        ivh_discretisation_method="fixed_bin_size",
        ivh_discretisation_bin_width=2.5,
        glcm_distance=1.0,
        glcm_spatial_method=["3d_average", "3d_volume_merge"],
        glrlm_spatial_method=["3d_average", "3d_volume_merge"],
        glszm_spatial_method="3d",
        gldzm_spatial_method="3d",
        ngtdm_spatial_method="3d",
        ngldm_distance=1.0,
        ngldm_spatial_method="3d",
        ngldm_difference_level=0.0
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
        roi_resegment_settings=resegmentation_settings,
        perturbation_settings=ImagePerturbationSettingsClass(),
        img_transform_settings=image_transformation_settings,
        feature_extr_settings=feature_computation_parameters
    )

    data = extract_features(
        write_features=False,
        export_features=True,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        settings=settings
    )

    data = data[0]

    assert within_tolerance(367000, 6000, data["morph_volume"])
    assert within_tolerance(368000, 6000, data["morph_vol_approx"])
    assert within_tolerance(34300, 400, data["morph_area_mesh"])
    assert within_tolerance(0.0934, 7e-04, data["morph_av"])
    assert within_tolerance(0.0326, 2e-04, data["morph_comp_1"])
    assert within_tolerance(0.378, 0.004, data["morph_comp_2"])
    assert within_tolerance(1.38, 0.01, data["morph_sph_dispr"])
    assert within_tolerance(0.723, 0.003, data["morph_sphericity"])
    assert within_tolerance(0.383, 0.004, data["morph_asphericity"])
    assert within_tolerance(45.6, 2.8, data["morph_com"])
    assert within_tolerance(125, 1, data["morph_diam"])
    assert within_tolerance(93.3, 0.5, data["morph_pca_maj_axis"])
    assert within_tolerance(82, 0.5, data["morph_pca_min_axis"])
    assert within_tolerance(70.9, 0.4, data["morph_pca_least_axis"])
    assert within_tolerance(0.879, 0.001, data["morph_pca_elongation"])
    assert within_tolerance(0.76, 0.001, data["morph_pca_flatness"])
    assert within_tolerance(0.478, 0.003, data["morph_vol_dens_aabb"])
    assert within_tolerance(0.678, 0.003, data["morph_area_dens_aabb"])
    assert within_tolerance(1.29, 0.01, data["morph_vol_dens_aee"])
    assert within_tolerance(1.62, 0.01, data["morph_area_dens_aee"])
    assert within_tolerance(0.834, 0.002, data["morph_vol_dens_conv_hull"])
    assert within_tolerance(1.13, 0.01, data["morph_area_dens_conv_hull"])
    assert within_tolerance(-1.8e+07, 1400000, data["morph_integ_int"])
    # assert within_tolerance(0.0824, 3e-04, data["morph_moran_i"])
    # assert within_tolerance(0.846, 0.001, data["morph_geary_c"])
    assert within_tolerance(169, 10, data["loc_peak_loc"])
    assert within_tolerance(180, 5, data["loc_peak_glob"])
    assert within_tolerance(-49, 2.9, data["stat_mean"])
    assert within_tolerance(50600, 1400, data["stat_var"])
    assert within_tolerance(-2.14, 0.05, data["stat_skew"])
    assert within_tolerance(3.53, 0.23, data["stat_kurt"])
    assert within_tolerance(40, 0.4, data["stat_median"])
    assert within_tolerance(-939, 4, data["stat_min"])
    assert within_tolerance(-424, 14, data["stat_p10"])
    assert within_tolerance(86, 0.1, data["stat_p90"])
    assert within_tolerance(393, 10, data["stat_max"])
    assert within_tolerance(67, 4.9, data["stat_iqr"])
    assert within_tolerance(1330, 20, data["stat_range"])
    assert within_tolerance(158, 4, data["stat_mad"])
    assert within_tolerance(66.8, 3.5, data["stat_rmad"])
    assert within_tolerance(119, 4, data["stat_medad"])
    assert within_tolerance(-4.59, 0.29, data["stat_cov"])
    assert within_tolerance(1.03, 0.4, data["stat_qcod"])
    assert within_tolerance(2.44e+09, 1.2e+08, data["stat_energy"])
    assert within_tolerance(230, 4, data["stat_rms"])
    assert within_tolerance(38.6, 0.2, data["ih_mean_fbs_w25.0"])
    assert within_tolerance(81.1, 2.1, data["ih_var_fbs_w25.0"])
    assert within_tolerance(-2.14, 0.05, data["ih_skew_fbs_w25.0"])
    assert within_tolerance(3.52, 0.23, data["ih_kurt_fbs_w25.0"])
    assert within_tolerance(42, 0, data["ih_median_fbs_w25.0"])
    assert within_tolerance(3, 0.16, data["ih_min_fbs_w25.0"])
    assert within_tolerance(24, 0.7, data["ih_p10_fbs_w25.0"])
    assert within_tolerance(44, 0, data["ih_p90_fbs_w25.0"])
    assert within_tolerance(56, 0.5, data["ih_max_fbs_w25.0"])
    assert within_tolerance(43, 0.1, data["ih_mode_fbs_w25.0"])
    assert within_tolerance(3, 0.21, data["ih_iqr_fbs_w25.0"])
    assert within_tolerance(53, 0.6, data["ih_range_fbs_w25.0"])
    assert within_tolerance(6.32, 0.15, data["ih_mad_fbs_w25.0"])
    assert within_tolerance(2.59, 0.14, data["ih_rmad_fbs_w25.0"])
    assert within_tolerance(4.75, 0.12, data["ih_medad_fbs_w25.0"])
    assert within_tolerance(0.234, 0.005, data["ih_cov_fbs_w25.0"])
    assert within_tolerance(0.0361, 0.0027, data["ih_qcod_fbs_w25.0"])
    assert within_tolerance(3.73, 0.04, data["ih_entropy_fbs_w25.0"])
    assert within_tolerance(0.14, 0.003, data["ih_uniformity_fbs_w25.0"])
    assert within_tolerance(4750, 30, data["ih_max_grad_fbs_w25.0"])
    assert within_tolerance(41, 0, data["ih_max_grad_g_fbs_w25.0"])
    assert within_tolerance(-4680, 50, data["ih_min_grad_fbs_w25.0"])
    assert within_tolerance(44, 0, data["ih_min_grad_g_fbs_w25.0"])
    assert within_tolerance(0.998, 0.001, data["ivh_v10"])
    assert within_tolerance(0.000152, 2e-05, data["ivh_v90"])
    assert within_tolerance(88.8, 0.2, data["ivh_i10"])
    assert within_tolerance(-421, 14, data["ivh_i90"])
    assert within_tolerance(0.997, 0.001, data["ivh_diff_v10_v90"])
    assert within_tolerance(510, 14, data["ivh_diff_i10_i90"])
    assert within_tolerance(0.111, 0.002, data["cm_joint_max_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(39, 0.2, data["cm_joint_avg_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(73.7, 2, data["cm_joint_var_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(6.39, 0.06, data["cm_joint_entr_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(2.17, 0.05, data["cm_diff_avg_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(14.4, 0.5, data["cm_diff_var_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(2.64, 0.03, data["cm_diff_entr_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(78, 0.3, data["cm_sum_avg_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(276, 8, data["cm_sum_var_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(4.56, 0.04, data["cm_sum_entr_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.045, 0.001, data["cm_energy_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(19.2, 0.7, data["cm_contrast_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(2.17, 0.05, data["cm_dissimilarity_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.582, 0.004, data["cm_inv_diff_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.966, 0.001, data["cm_inv_diff_norm_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.547, 0.004, data["cm_inv_diff_mom_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.994, 0.001, data["cm_inv_diff_mom_norm_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.39, 0.003, data["cm_inv_var_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.869, 0.001, data["cm_corr_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(1580, 10, data["cm_auto_corr_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(276, 8, data["cm_clust_tend_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(-10600, 300, data["cm_clust_shade_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(569000, 11000, data["cm_clust_prom_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(-0.236, 0.001, data["cm_info_corr1_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.9, 0.001, data["cm_info_corr2_d1_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.111, 0.002, data["cm_joint_max_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(39, 0.2, data["cm_joint_avg_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(73.8, 2, data["cm_joint_var_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(6.42, 0.06, data["cm_joint_entr_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(2.16, 0.05, data["cm_diff_avg_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(14.4, 0.5, data["cm_diff_var_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(2.64, 0.03, data["cm_diff_entr_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(78, 0.3, data["cm_sum_avg_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(276, 8, data["cm_sum_var_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(4.56, 0.04, data["cm_sum_entr_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.0447, 0.001, data["cm_energy_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(19.1, 0.7, data["cm_contrast_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(2.16, 0.05, data["cm_dissimilarity_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.583, 0.004, data["cm_inv_diff_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.966, 0.001, data["cm_inv_diff_norm_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.548, 0.004, data["cm_inv_diff_mom_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.994, 0.001, data["cm_inv_diff_mom_norm_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.39, 0.003, data["cm_inv_var_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.871, 0.001, data["cm_corr_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(1580, 10, data["cm_auto_corr_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(276, 8, data["cm_clust_tend_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(-10600, 300, data["cm_clust_shade_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(570000, 11000, data["cm_clust_prom_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(-0.228, 0.001, data["cm_info_corr1_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.899, 0.001, data["cm_info_corr2_d1_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.786, 0.003, data["rlm_sre_3d_avg_fbs_w25.0"])
    assert within_tolerance(3.31, 0.04, data["rlm_lre_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.00155, 5e-05, data["rlm_lgre_3d_avg_fbs_w25.0"])
    assert within_tolerance(1470, 10, data["rlm_hgre_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.00136, 5e-05, data["rlm_srlge_3d_avg_fbs_w25.0"])
    assert within_tolerance(1100, 10, data["rlm_srhge_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.00317, 4e-05, data["rlm_lrlge_3d_avg_fbs_w25.0"])
    assert within_tolerance(5590, 80, data["rlm_lrhge_3d_avg_fbs_w25.0"])
    assert within_tolerance(3180, 10, data["rlm_glnu_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.102, 0.003, data["rlm_glnu_norm_3d_avg_fbs_w25.0"])
    assert within_tolerance(18000, 500, data["rlm_rlnu_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.574, 0.004, data["rlm_rlnu_norm_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.679, 0.003, data["rlm_r_perc_3d_avg_fbs_w25.0"])
    assert within_tolerance(101, 3, data["rlm_gl_var_3d_avg_fbs_w25.0"])
    assert within_tolerance(1.12, 0.02, data["rlm_rl_var_3d_avg_fbs_w25.0"])
    assert within_tolerance(5.35, 0.03, data["rlm_rl_entr_3d_avg_fbs_w25.0"])
    assert within_tolerance(0.787, 0.003, data["rlm_sre_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(3.28, 0.04, data["rlm_lre_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.00155, 5e-05, data["rlm_lgre_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(1470, 10, data["rlm_hgre_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.00136, 5e-05, data["rlm_srlge_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(1100, 10, data["rlm_srhge_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.00314, 4e-05, data["rlm_lrlge_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(5530, 80, data["rlm_lrhge_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(41300, 100, data["rlm_glnu_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.102, 0.003, data["rlm_glnu_norm_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(234000, 6000, data["rlm_rlnu_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.575, 0.004, data["rlm_rlnu_norm_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.679, 0.003, data["rlm_r_perc_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(101, 3, data["rlm_gl_var_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(1.11, 0.02, data["rlm_rl_var_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(5.35, 0.03, data["rlm_rl_entr_3d_v_mrg_fbs_w25.0"])
    assert within_tolerance(0.695, 0.001, data["szm_sze_3d_fbs_w25.0"])
    assert within_tolerance(38900, 900, data["szm_lze_3d_fbs_w25.0"])
    assert within_tolerance(0.00235, 6e-05, data["szm_lgze_3d_fbs_w25.0"])
    assert within_tolerance(971, 7, data["szm_hgze_3d_fbs_w25.0"])
    assert within_tolerance(0.0016, 4e-05, data["szm_szlge_3d_fbs_w25.0"])
    assert within_tolerance(657, 4, data["szm_szhge_3d_fbs_w25.0"])
    assert within_tolerance(21.6, 0.5, data["szm_lzlge_3d_fbs_w25.0"])
    assert within_tolerance(70700000, 1500000, data["szm_lzhge_3d_fbs_w25.0"])
    assert within_tolerance(195, 6, data["szm_glnu_3d_fbs_w25.0"])
    assert within_tolerance(0.0286, 3e-04, data["szm_glnu_norm_3d_fbs_w25.0"])
    assert within_tolerance(3040, 100, data["szm_zsnu_3d_fbs_w25.0"])
    assert within_tolerance(0.447, 0.001, data["szm_zsnu_norm_3d_fbs_w25.0"])
    assert within_tolerance(0.148, 0.003, data["szm_z_perc_3d_fbs_w25.0"])
    assert within_tolerance(106, 1, data["szm_gl_var_3d_fbs_w25.0"])
    assert within_tolerance(38900, 900, data["szm_zs_var_3d_fbs_w25.0"])
    assert within_tolerance(7, 0.01, data["szm_zs_entr_3d_fbs_w25.0"])
    assert within_tolerance(0.531, 0.006, data["dzm_sde_3d_fbs_w25.0"])
    assert within_tolerance(11, 0.3, data["dzm_lde_3d_fbs_w25.0"])
    assert within_tolerance(0.00235, 6e-05, data["dzm_lgze_3d_fbs_w25.0"])
    assert within_tolerance(971, 7, data["dzm_hgze_3d_fbs_w25.0"])
    assert within_tolerance(0.00149, 4e-05, data["dzm_sdlge_3d_fbs_w25.0"])
    assert within_tolerance(476, 11, data["dzm_sdhge_3d_fbs_w25.0"])
    assert within_tolerance(0.0154, 5e-04, data["dzm_ldlge_3d_fbs_w25.0"])
    assert within_tolerance(13400, 200, data["dzm_ldhge_3d_fbs_w25.0"])
    assert within_tolerance(195, 6, data["dzm_glnu_3d_fbs_w25.0"])
    assert within_tolerance(0.0286, 3e-04, data["dzm_glnu_norm_3d_fbs_w25.0"])
    assert within_tolerance(1870, 40, data["dzm_zdnu_3d_fbs_w25.0"])
    assert within_tolerance(0.274, 0.005, data["dzm_zdnu_norm_3d_fbs_w25.0"])
    assert within_tolerance(0.148, 0.003, data["dzm_z_perc_3d_fbs_w25.0"])
    assert within_tolerance(106, 1, data["dzm_gl_var_3d_fbs_w25.0"])
    assert within_tolerance(4.6, 0.06, data["dzm_zd_var_3d_fbs_w25.0"])
    assert within_tolerance(7.56, 0.03, data["dzm_zd_entr_3d_fbs_w25.0"])
    assert within_tolerance(0.000216, 4e-06, data["ngt_coarseness_3d_fbs_w25.0"])
    assert within_tolerance(0.0873, 0.0019, data["ngt_contrast_3d_fbs_w25.0"])
    assert within_tolerance(1.39, 0.01, data["ngt_busyness_3d_fbs_w25.0"])
    assert within_tolerance(1810, 60, data["ngt_complexity_3d_fbs_w25.0"])
    assert within_tolerance(0.651, 0.015, data["ngt_strength_3d_fbs_w25.0"])
    assert within_tolerance(0.137, 0.003, data["ngl_lde_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(126, 2, data["ngl_hde_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(0.0013, 4e-05, data["ngl_lgce_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(1570, 10, data["ngl_hgce_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(0.000306, 1.2e-05, data["ngl_ldlge_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(141, 2, data["ngl_ldhge_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(0.0828, 3e-04, data["ngl_hdlge_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(227000, 3000, data["ngl_hdhge_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(6420, 10, data["ngl_glnu_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(0.14, 0.003, data["ngl_glnu_norm_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(2450, 60, data["ngl_dcnu_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(0.0532, 5e-04, data["ngl_dcnu_norm_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(1, 0, data["ngl_dc_perc_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(81.1, 2.1, data["ngl_gl_var_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(39.2, 0.1, data["ngl_dc_var_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(7.54, 0.03, data["ngl_dc_entr_d1_a0.0_3d_fbs_w25.0"])
    assert within_tolerance(0.00789, 0.00011, data["ngl_dc_energy_d1_a0.0_3d_fbs_w25.0"])


def test_ibsi_1_chest_config_d():
    """
    Compare computed feature values with reference values for the chest CT image obtained using image processing
    configuration scheme D.
    """

    general_settings = GeneralSettingsClass(
        by_slice=False
    )

    image_interpolation_settings = ImageInterpolationSettingsClass(
        by_slice=general_settings.by_slice,
        spline_order=1,
        new_spacing=2.0,
        anti_aliasing=False
    )

    resegmentation_settings = ResegmentationSettingsClass(
        resegmentation_sigma=3.0
    )

    feature_computation_parameters = FeatureExtractionSettingsClass(
        by_slice=general_settings.by_slice,
        no_approximation=False,
        base_feature_families="all",
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=32,
        glcm_distance=1.0,
        glcm_spatial_method=["3d_average", "3d_volume_merge"],
        glrlm_spatial_method=["3d_average", "3d_volume_merge"],
        glszm_spatial_method="3d",
        gldzm_spatial_method="3d",
        ngtdm_spatial_method="3d",
        ngldm_distance=1.0,
        ngldm_spatial_method="3d",
        ngldm_difference_level=0.0
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
        roi_resegment_settings=resegmentation_settings,
        perturbation_settings=ImagePerturbationSettingsClass(),
        img_transform_settings=image_transformation_settings,
        feature_extr_settings=feature_computation_parameters
    )

    data = extract_features(
        write_features=False,
        export_features=True,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        settings=settings
    )

    data = data[0]

    assert within_tolerance(367000, 6000, data["morph_volume"])
    assert within_tolerance(368000, 6000, data["morph_vol_approx"])
    assert within_tolerance(34300, 400, data["morph_area_mesh"])
    assert within_tolerance(0.0934, 7e-04, data["morph_av"])
    assert within_tolerance(0.0326, 2e-04, data["morph_comp_1"])
    assert within_tolerance(0.378, 0.004, data["morph_comp_2"])
    assert within_tolerance(1.38, 0.01, data["morph_sph_dispr"])
    assert within_tolerance(0.723, 0.003, data["morph_sphericity"])
    assert within_tolerance(0.383, 0.004, data["morph_asphericity"])
    assert within_tolerance(64.9, 2.8, data["morph_com"])
    assert within_tolerance(125, 1, data["morph_diam"])
    assert within_tolerance(93.3, 0.5, data["morph_pca_maj_axis"])
    assert within_tolerance(82, 0.5, data["morph_pca_min_axis"])
    assert within_tolerance(70.9, 0.4, data["morph_pca_least_axis"])
    assert within_tolerance(0.879, 0.001, data["morph_pca_elongation"])
    assert within_tolerance(0.76, 0.001, data["morph_pca_flatness"])
    assert within_tolerance(0.478, 0.003, data["morph_vol_dens_aabb"])
    assert within_tolerance(0.678, 0.003, data["morph_area_dens_aabb"])
    assert within_tolerance(1.29, 0.01, data["morph_vol_dens_aee"])
    assert within_tolerance(1.62, 0.01, data["morph_area_dens_aee"])
    assert within_tolerance(0.834, 0.002, data["morph_vol_dens_conv_hull"])
    assert within_tolerance(1.13, 0.01, data["morph_area_dens_conv_hull"])
    assert within_tolerance(-8640000, 1560000, data["morph_integ_int"])
    # assert within_tolerance(0.0622, 0.0013, data["morph_moran_i"])
    # assert within_tolerance(0.851, 0.001, data["morph_geary_c"])
    assert within_tolerance(201, 10, data["loc_peak_loc"])
    assert within_tolerance(201, 5, data["loc_peak_glob"])
    assert within_tolerance(-23.5, 3.9, data["stat_mean"])
    assert within_tolerance(32800, 2100, data["stat_var"])
    assert within_tolerance(-2.28, 0.06, data["stat_skew"])
    assert within_tolerance(4.35, 0.32, data["stat_kurt"])
    assert within_tolerance(42, 0.4, data["stat_median"])
    assert within_tolerance(-724, 12, data["stat_min"])
    assert within_tolerance(-304, 20, data["stat_p10"])
    assert within_tolerance(86, 0.1, data["stat_p90"])
    assert within_tolerance(521, 22, data["stat_max"])
    assert within_tolerance(57, 4.1, data["stat_iqr"])
    assert within_tolerance(1240, 40, data["stat_range"])
    assert within_tolerance(123, 6, data["stat_mad"])
    assert within_tolerance(46.8, 3.6, data["stat_rmad"])
    assert within_tolerance(94.7, 3.8, data["stat_medad"])
    assert within_tolerance(-7.7, 1.01, data["stat_cov"])
    assert within_tolerance(0.74, 0.011, data["stat_qcod"])
    assert within_tolerance(1.48e+09, 1.4e+08, data["stat_energy"])
    assert within_tolerance(183, 7, data["stat_rms"])
    assert within_tolerance(18.5, 0.5, data["ih_mean_fbn_n32"])
    assert within_tolerance(21.7, 0.4, data["ih_var_fbn_n32"])
    assert within_tolerance(-2.27, 0.06, data["ih_skew_fbn_n32"])
    assert within_tolerance(4.31, 0.32, data["ih_kurt_fbn_n32"])
    assert within_tolerance(20, 0.5, data["ih_median_fbn_n32"])
    assert within_tolerance(1, 0, data["ih_min_fbn_n32"])
    assert within_tolerance(11, 0.7, data["ih_p10_fbn_n32"])
    assert within_tolerance(21, 0.5, data["ih_p90_fbn_n32"])
    assert within_tolerance(32, 0, data["ih_max_fbn_n32"])
    assert within_tolerance(20, 0.4, data["ih_mode_fbn_n32"])
    assert within_tolerance(2, 0.06, data["ih_iqr_fbn_n32"])
    assert within_tolerance(31, 0, data["ih_range_fbn_n32"])
    assert within_tolerance(3.15, 0.05, data["ih_mad_fbn_n32"])
    assert within_tolerance(1.33, 0.06, data["ih_rmad_fbn_n32"])
    assert within_tolerance(2.41, 0.04, data["ih_medad_fbn_n32"])
    assert within_tolerance(0.252, 0.006, data["ih_cov_fbn_n32"])
    assert within_tolerance(0.05, 0.0021, data["ih_qcod_fbn_n32"])
    assert within_tolerance(2.94, 0.01, data["ih_entropy_fbn_n32"])
    assert within_tolerance(0.229, 0.003, data["ih_uniformity_fbn_n32"])
    assert within_tolerance(7260, 200, data["ih_max_grad_fbn_n32"])
    assert within_tolerance(19, 0.4, data["ih_max_grad_g_fbn_n32"])
    assert within_tolerance(-6670, 230, data["ih_min_grad_fbn_n32"])
    assert within_tolerance(22, 0.4, data["ih_min_grad_g_fbn_n32"])
    assert within_tolerance(0.972, 0.003, data["ivh_v10"])
    assert within_tolerance(9e-05, 0.000415, data["ivh_v90"])
    assert within_tolerance(87, 0.1, data["ivh_i10"])
    assert within_tolerance(-303, 20, data["ivh_i90"])
    assert within_tolerance(0.971, 0.001, data["ivh_diff_v10_v90"])
    assert within_tolerance(390, 20, data["ivh_diff_i10_i90"])
    assert within_tolerance(0.232, 0.007, data["cm_joint_max_d1_3d_avg_fbn_n32"])
    assert within_tolerance(18.9, 0.5, data["cm_joint_avg_d1_3d_avg_fbn_n32"])
    assert within_tolerance(17.6, 0.4, data["cm_joint_var_d1_3d_avg_fbn_n32"])
    assert within_tolerance(4.95, 0.03, data["cm_joint_entr_d1_3d_avg_fbn_n32"])
    assert within_tolerance(1.29, 0.01, data["cm_diff_avg_d1_3d_avg_fbn_n32"])
    assert within_tolerance(5.37, 0.11, data["cm_diff_var_d1_3d_avg_fbn_n32"])
    assert within_tolerance(2.13, 0.01, data["cm_diff_entr_d1_3d_avg_fbn_n32"])
    assert within_tolerance(37.7, 0.8, data["cm_sum_avg_d1_3d_avg_fbn_n32"])
    assert within_tolerance(63.4, 1.3, data["cm_sum_var_d1_3d_avg_fbn_n32"])
    assert within_tolerance(3.68, 0.02, data["cm_sum_entr_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.11, 0.003, data["cm_energy_d1_3d_avg_fbn_n32"])
    assert within_tolerance(7.07, 0.13, data["cm_contrast_d1_3d_avg_fbn_n32"])
    assert within_tolerance(1.29, 0.01, data["cm_dissimilarity_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.682, 0.003, data["cm_inv_diff_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.965, 0.001, data["cm_inv_diff_norm_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.656, 0.003, data["cm_inv_diff_mom_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.994, 0.001, data["cm_inv_diff_mom_norm_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.341, 0.005, data["cm_inv_var_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.798, 0.005, data["cm_corr_d1_3d_avg_fbn_n32"])
    assert within_tolerance(370, 16, data["cm_auto_corr_d1_3d_avg_fbn_n32"])
    assert within_tolerance(63.4, 1.3, data["cm_clust_tend_d1_3d_avg_fbn_n32"])
    assert within_tolerance(-1270, 40, data["cm_clust_shade_d1_3d_avg_fbn_n32"])
    assert within_tolerance(35700, 1400, data["cm_clust_prom_d1_3d_avg_fbn_n32"])
    assert within_tolerance(-0.231, 0.003, data["cm_info_corr1_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.845, 0.003, data["cm_info_corr2_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.232, 0.007, data["cm_joint_max_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(18.9, 0.5, data["cm_joint_avg_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(17.6, 0.4, data["cm_joint_var_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(4.96, 0.03, data["cm_joint_entr_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(1.29, 0.01, data["cm_diff_avg_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(5.38, 0.11, data["cm_diff_var_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(2.14, 0.01, data["cm_diff_entr_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(37.7, 0.8, data["cm_sum_avg_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(63.5, 1.3, data["cm_sum_var_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(3.68, 0.02, data["cm_sum_entr_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.109, 0.003, data["cm_energy_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(7.05, 0.13, data["cm_contrast_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(1.29, 0.01, data["cm_dissimilarity_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.682, 0.003, data["cm_inv_diff_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.965, 0.001, data["cm_inv_diff_norm_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.657, 0.003, data["cm_inv_diff_mom_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.994, 0.001, data["cm_inv_diff_mom_norm_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.34, 0.005, data["cm_inv_var_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.8, 0.005, data["cm_corr_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(370, 16, data["cm_auto_corr_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(63.5, 1.3, data["cm_clust_tend_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(-1280, 40, data["cm_clust_shade_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(35700, 1500, data["cm_clust_prom_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(-0.225, 0.003, data["cm_info_corr1_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.846, 0.003, data["cm_info_corr2_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.734, 0.001, data["rlm_sre_3d_avg_fbn_n32"])
    assert within_tolerance(6.66, 0.18, data["rlm_lre_3d_avg_fbn_n32"])
    assert within_tolerance(0.0257, 0.0012, data["rlm_lgre_3d_avg_fbn_n32"])
    assert within_tolerance(326, 17, data["rlm_hgre_3d_avg_fbn_n32"])
    assert within_tolerance(0.0232, 0.001, data["rlm_srlge_3d_avg_fbn_n32"])
    assert within_tolerance(219, 13, data["rlm_srhge_3d_avg_fbn_n32"])
    assert within_tolerance(0.0484, 0.0031, data["rlm_lrlge_3d_avg_fbn_n32"])
    assert within_tolerance(2670, 30, data["rlm_lrhge_3d_avg_fbn_n32"])
    assert within_tolerance(3290, 10, data["rlm_glnu_3d_avg_fbn_n32"])
    assert within_tolerance(0.133, 0.002, data["rlm_glnu_norm_3d_avg_fbn_n32"])
    assert within_tolerance(12400, 200, data["rlm_rlnu_3d_avg_fbn_n32"])
    assert within_tolerance(0.5, 0.001, data["rlm_rlnu_norm_3d_avg_fbn_n32"])
    assert within_tolerance(0.554, 0.005, data["rlm_r_perc_3d_avg_fbn_n32"])
    assert within_tolerance(31.5, 0.4, data["rlm_gl_var_3d_avg_fbn_n32"])
    assert within_tolerance(3.35, 0.14, data["rlm_rl_var_3d_avg_fbn_n32"])
    assert within_tolerance(5.08, 0.02, data["rlm_rl_entr_3d_avg_fbn_n32"])
    assert within_tolerance(0.736, 0.001, data["rlm_sre_3d_v_mrg_fbn_n32"])
    assert within_tolerance(6.56, 0.18, data["rlm_lre_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.0257, 0.0012, data["rlm_lgre_3d_v_mrg_fbn_n32"])
    assert within_tolerance(326, 17, data["rlm_hgre_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.0232, 0.001, data["rlm_srlge_3d_v_mrg_fbn_n32"])
    assert within_tolerance(219, 13, data["rlm_srhge_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.0478, 0.0031, data["rlm_lrlge_3d_v_mrg_fbn_n32"])
    assert within_tolerance(2630, 30, data["rlm_lrhge_3d_v_mrg_fbn_n32"])
    assert within_tolerance(42800, 200, data["rlm_glnu_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.134, 0.002, data["rlm_glnu_norm_3d_v_mrg_fbn_n32"])
    assert within_tolerance(160000, 3000, data["rlm_rlnu_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.501, 0.001, data["rlm_rlnu_norm_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.554, 0.005, data["rlm_r_perc_3d_v_mrg_fbn_n32"])
    assert within_tolerance(31.4, 0.4, data["rlm_gl_var_3d_v_mrg_fbn_n32"])
    assert within_tolerance(3.29, 0.13, data["rlm_rl_var_3d_v_mrg_fbn_n32"])
    assert within_tolerance(5.08, 0.02, data["rlm_rl_entr_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.637, 0.005, data["szm_sze_3d_fbn_n32"])
    assert within_tolerance(99100, 2800, data["szm_lze_3d_fbn_n32"])
    assert within_tolerance(0.0409, 5e-04, data["szm_lgze_3d_fbn_n32"])
    assert within_tolerance(188, 10, data["szm_hgze_3d_fbn_n32"])
    assert within_tolerance(0.0248, 4e-04, data["szm_szlge_3d_fbn_n32"])
    assert within_tolerance(117, 7, data["szm_szhge_3d_fbn_n32"])
    assert within_tolerance(241, 14, data["szm_lzlge_3d_fbn_n32"])
    assert within_tolerance(41400000, 3e+05, data["szm_lzhge_3d_fbn_n32"])
    assert within_tolerance(212, 6, data["szm_glnu_3d_fbn_n32"])
    assert within_tolerance(0.0491, 8e-04, data["szm_glnu_norm_3d_fbn_n32"])
    assert within_tolerance(1630, 10, data["szm_zsnu_3d_fbn_n32"])
    assert within_tolerance(0.377, 0.006, data["szm_zsnu_norm_3d_fbn_n32"])
    assert within_tolerance(0.0972, 7e-04, data["szm_z_perc_3d_fbn_n32"])
    assert within_tolerance(32.7, 1.6, data["szm_gl_var_3d_fbn_n32"])
    assert within_tolerance(99000, 2800, data["szm_zs_var_3d_fbn_n32"])
    assert within_tolerance(6.52, 0.01, data["szm_zs_entr_3d_fbn_n32"])
    assert within_tolerance(0.579, 0.004, data["dzm_sde_3d_fbn_n32"])
    assert within_tolerance(10.3, 0.1, data["dzm_lde_3d_fbn_n32"])
    assert within_tolerance(0.0409, 5e-04, data["dzm_lgze_3d_fbn_n32"])
    assert within_tolerance(188, 10, data["dzm_hgze_3d_fbn_n32"])
    assert within_tolerance(0.0302, 6e-04, data["dzm_sdlge_3d_fbn_n32"])
    assert within_tolerance(99.3, 5.1, data["dzm_sdhge_3d_fbn_n32"])
    assert within_tolerance(0.183, 0.004, data["dzm_ldlge_3d_fbn_n32"])
    assert within_tolerance(2620, 110, data["dzm_ldhge_3d_fbn_n32"])
    assert within_tolerance(212, 6, data["dzm_glnu_3d_fbn_n32"])
    assert within_tolerance(0.0491, 8e-04, data["dzm_glnu_norm_3d_fbn_n32"])
    assert within_tolerance(1370, 20, data["dzm_zdnu_3d_fbn_n32"])
    assert within_tolerance(0.317, 0.004, data["dzm_zdnu_norm_3d_fbn_n32"])
    assert within_tolerance(0.0972, 7e-04, data["dzm_z_perc_3d_fbn_n32"])
    assert within_tolerance(32.7, 1.6, data["dzm_gl_var_3d_fbn_n32"])
    assert within_tolerance(4.61, 0.04, data["dzm_zd_var_3d_fbn_n32"])
    assert within_tolerance(6.61, 0.03, data["dzm_zd_entr_3d_fbn_n32"])
    assert within_tolerance(0.000208, 4e-06, data["ngt_coarseness_3d_fbn_n32"])
    assert within_tolerance(0.046, 5e-04, data["ngt_contrast_3d_fbn_n32"])
    assert within_tolerance(5.14, 0.14, data["ngt_busyness_3d_fbn_n32"])
    assert within_tolerance(400, 5, data["ngt_complexity_3d_fbn_n32"])
    assert within_tolerance(0.162, 0.008, data["ngt_strength_3d_fbn_n32"])
    assert within_tolerance(0.0912, 7e-04, data["ngl_lde_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(223, 5, data["ngl_hde_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(0.0168, 9e-04, data["ngl_lgce_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(364, 16, data["ngl_hgce_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(0.00357, 4e-05, data["ngl_ldlge_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(18.9, 1.1, data["ngl_ldhge_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(0.798, 0.072, data["ngl_hdlge_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(92800, 1300, data["ngl_hdhge_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(10200, 300, data["ngl_glnu_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(0.229, 0.003, data["ngl_glnu_norm_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(1840, 30, data["ngl_dcnu_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(0.0413, 3e-04, data["ngl_dcnu_norm_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(1, 0, data["ngl_dc_perc_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(21.7, 0.4, data["ngl_gl_var_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(63.9, 1.3, data["ngl_dc_var_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(6.98, 0.01, data["ngl_dc_entr_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(0.0113, 2e-04, data["ngl_dc_energy_d1_a0.0_3d_fbn_n32"])


def test_ibsi_1_chest_config_e():
    """
    Compare computed feature values with reference values for the chest CT image obtained using image processing
    configuration scheme E.
    """

    general_settings = GeneralSettingsClass(
        by_slice=False
    )

    image_interpolation_settings = ImageInterpolationSettingsClass(
        by_slice=general_settings.by_slice,
        spline_order=3,
        new_spacing=2.0,
        anti_aliasing=False
    )

    resegmentation_settings = ResegmentationSettingsClass(
        resegmentation_intensity_range=[-1000.0, 400.0],
        resegmentation_sigma=3.0
    )

    feature_computation_parameters = FeatureExtractionSettingsClass(
        by_slice=general_settings.by_slice,
        no_approximation=False,
        base_feature_families="all",
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=32,
        ivh_discretisation_method="fixed_bin_number",
        ivh_discretisation_n_bins=1000,
        glcm_distance=1.0,
        glcm_spatial_method=["3d_average", "3d_volume_merge"],
        glrlm_spatial_method=["3d_average", "3d_volume_merge"],
        glszm_spatial_method="3d",
        gldzm_spatial_method="3d",
        ngtdm_spatial_method="3d",
        ngldm_distance=1.0,
        ngldm_spatial_method="3d",
        ngldm_difference_level=0.0
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
        roi_resegment_settings=resegmentation_settings,
        perturbation_settings=ImagePerturbationSettingsClass(),
        img_transform_settings=image_transformation_settings,
        feature_extr_settings=feature_computation_parameters
    )

    data = extract_features(
        write_features=False,
        export_features=True,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        settings=settings
    )

    data = data[0]

    assert within_tolerance(367000, 6000, data["morph_volume"])
    assert within_tolerance(368000, 6000, data["morph_vol_approx"])
    assert within_tolerance(34300, 400, data["morph_area_mesh"])
    assert within_tolerance(0.0934, 7e-04, data["morph_av"])
    assert within_tolerance(0.0326, 2e-04, data["morph_comp_1"])
    assert within_tolerance(0.378, 0.004, data["morph_comp_2"])
    assert within_tolerance(1.38, 0.01, data["morph_sph_dispr"])
    assert within_tolerance(0.723, 0.003, data["morph_sphericity"])
    assert within_tolerance(0.383, 0.004, data["morph_asphericity"])
    assert within_tolerance(68.5, 2.1, data["morph_com"])
    assert within_tolerance(125, 1, data["morph_diam"])
    assert within_tolerance(93.3, 0.5, data["morph_pca_maj_axis"])
    assert within_tolerance(82, 0.5, data["morph_pca_min_axis"])
    assert within_tolerance(70.9, 0.4, data["morph_pca_least_axis"])
    assert within_tolerance(0.879, 0.001, data["morph_pca_elongation"])
    assert within_tolerance(0.76, 0.001, data["morph_pca_flatness"])
    assert within_tolerance(0.478, 0.003, data["morph_vol_dens_aabb"])
    assert within_tolerance(0.678, 0.003, data["morph_area_dens_aabb"])
    assert within_tolerance(1.29, 0.01, data["morph_vol_dens_aee"])
    assert within_tolerance(1.62, 0.01, data["morph_area_dens_aee"])
    assert within_tolerance(0.834, 0.002, data["morph_vol_dens_conv_hull"])
    assert within_tolerance(1.13, 0.01, data["morph_area_dens_conv_hull"])
    assert within_tolerance(-8310000, 1600000, data["morph_integ_int"])
    # assert within_tolerance(0.0596, 0.0014, data["morph_moran_i"])
    # assert within_tolerance(0.853, 0.001, data["morph_geary_c"])
    assert within_tolerance(181, 13, data["loc_peak_loc"])
    assert within_tolerance(181, 5, data["loc_peak_glob"])
    assert within_tolerance(-22.6, 4.1, data["stat_mean"])
    assert within_tolerance(35100, 2200, data["stat_var"])
    assert within_tolerance(-2.3, 0.07, data["stat_skew"])
    assert within_tolerance(4.44, 0.33, data["stat_kurt"])
    assert within_tolerance(43, 0.5, data["stat_median"])
    assert within_tolerance(-743, 13, data["stat_min"])
    assert within_tolerance(-310, 21, data["stat_p10"])
    assert within_tolerance(93, 0.2, data["stat_p90"])
    assert within_tolerance(345, 9, data["stat_max"])
    assert within_tolerance(62, 3.5, data["stat_iqr"])
    assert within_tolerance(1090, 30, data["stat_range"])
    assert within_tolerance(125, 6, data["stat_mad"])
    assert within_tolerance(46.5, 3.7, data["stat_rmad"])
    assert within_tolerance(97.9, 3.9, data["stat_medad"])
    assert within_tolerance(-8.28, 0.95, data["stat_cov"])
    assert within_tolerance(0.795, 0.337, data["stat_qcod"])
    assert within_tolerance(1.58e+09, 1.4e+08, data["stat_energy"])
    assert within_tolerance(189, 7, data["stat_rms"])
    assert within_tolerance(21.7, 0.3, data["ih_mean_fbn_n32"])
    assert within_tolerance(30.4, 0.8, data["ih_var_fbn_n32"])
    assert within_tolerance(-2.29, 0.07, data["ih_skew_fbn_n32"])
    assert within_tolerance(4.4, 0.33, data["ih_kurt_fbn_n32"])
    assert within_tolerance(24, 0.2, data["ih_median_fbn_n32"])
    assert within_tolerance(1, 0, data["ih_min_fbn_n32"])
    assert within_tolerance(13, 0.7, data["ih_p10_fbn_n32"])
    assert within_tolerance(25, 0.2, data["ih_p90_fbn_n32"])
    assert within_tolerance(32, 0, data["ih_max_fbn_n32"])
    assert within_tolerance(24, 0.1, data["ih_mode_fbn_n32"])
    assert within_tolerance(1, 0.06, data["ih_iqr_fbn_n32"])
    assert within_tolerance(31, 0, data["ih_range_fbn_n32"])
    assert within_tolerance(3.69, 0.1, data["ih_mad_fbn_n32"])
    assert within_tolerance(1.46, 0.09, data["ih_rmad_fbn_n32"])
    assert within_tolerance(2.89, 0.07, data["ih_medad_fbn_n32"])
    assert within_tolerance(0.254, 0.006, data["ih_cov_fbn_n32"])
    assert within_tolerance(0.0213, 0.0015, data["ih_qcod_fbn_n32"])
    assert within_tolerance(3.22, 0.02, data["ih_entropy_fbn_n32"])
    assert within_tolerance(0.184, 0.001, data["ih_uniformity_fbn_n32"])
    assert within_tolerance(6010, 130, data["ih_max_grad_fbn_n32"])
    assert within_tolerance(23, 0.2, data["ih_max_grad_g_fbn_n32"])
    assert within_tolerance(-6110, 180, data["ih_min_grad_fbn_n32"])
    assert within_tolerance(25, 0.2, data["ih_min_grad_g_fbn_n32"])
    assert within_tolerance(0.975, 0.002, data["ivh_v10"])
    assert within_tolerance(0.000157, 0.000248, data["ivh_v90"])
    assert within_tolerance(770, 5, data["ivh_i10"])
    assert within_tolerance(399, 17, data["ivh_i90"])
    assert within_tolerance(0.974, 0.001, data["ivh_diff_v10_v90"])
    assert within_tolerance(371, 13, data["ivh_diff_i10_i90"])
    assert within_tolerance(0.153, 0.003, data["cm_joint_max_d1_3d_avg_fbn_n32"])
    assert within_tolerance(22.1, 0.3, data["cm_joint_avg_d1_3d_avg_fbn_n32"])
    assert within_tolerance(24.4, 0.9, data["cm_joint_var_d1_3d_avg_fbn_n32"])
    assert within_tolerance(5.6, 0.03, data["cm_joint_entr_d1_3d_avg_fbn_n32"])
    assert within_tolerance(1.7, 0.01, data["cm_diff_avg_d1_3d_avg_fbn_n32"])
    assert within_tolerance(8.22, 0.06, data["cm_diff_var_d1_3d_avg_fbn_n32"])
    assert within_tolerance(2.39, 0.01, data["cm_diff_entr_d1_3d_avg_fbn_n32"])
    assert within_tolerance(44.3, 0.4, data["cm_sum_avg_d1_3d_avg_fbn_n32"])
    assert within_tolerance(86.6, 3.3, data["cm_sum_var_d1_3d_avg_fbn_n32"])
    assert within_tolerance(3.96, 0.02, data["cm_sum_entr_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.0638, 9e-04, data["cm_energy_d1_3d_avg_fbn_n32"])
    assert within_tolerance(11.1, 0.1, data["cm_contrast_d1_3d_avg_fbn_n32"])
    assert within_tolerance(1.7, 0.01, data["cm_dissimilarity_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.608, 0.001, data["cm_inv_diff_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.955, 0.001, data["cm_inv_diff_norm_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.576, 0.001, data["cm_inv_diff_mom_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.99, 0.001, data["cm_inv_diff_mom_norm_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.41, 0.004, data["cm_inv_var_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.771, 0.006, data["cm_corr_d1_3d_avg_fbn_n32"])
    assert within_tolerance(509, 8, data["cm_auto_corr_d1_3d_avg_fbn_n32"])
    assert within_tolerance(86.6, 3.3, data["cm_clust_tend_d1_3d_avg_fbn_n32"])
    assert within_tolerance(-2070, 70, data["cm_clust_shade_d1_3d_avg_fbn_n32"])
    assert within_tolerance(68900, 2100, data["cm_clust_prom_d1_3d_avg_fbn_n32"])
    assert within_tolerance(-0.181, 0.003, data["cm_info_corr1_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.813, 0.004, data["cm_info_corr2_d1_3d_avg_fbn_n32"])
    assert within_tolerance(0.153, 0.003, data["cm_joint_max_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(22.1, 0.3, data["cm_joint_avg_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(24.4, 0.9, data["cm_joint_var_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(5.61, 0.03, data["cm_joint_entr_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(1.7, 0.01, data["cm_diff_avg_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(8.23, 0.06, data["cm_diff_var_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(2.4, 0.01, data["cm_diff_entr_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(44.3, 0.4, data["cm_sum_avg_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(86.7, 3.3, data["cm_sum_var_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(3.97, 0.02, data["cm_sum_entr_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.0635, 9e-04, data["cm_energy_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(11.1, 0.1, data["cm_contrast_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(1.7, 0.01, data["cm_dissimilarity_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.608, 0.001, data["cm_inv_diff_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.955, 0.001, data["cm_inv_diff_norm_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.577, 0.001, data["cm_inv_diff_mom_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.99, 0.001, data["cm_inv_diff_mom_norm_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.41, 0.004, data["cm_inv_var_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.773, 0.006, data["cm_corr_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(509, 8, data["cm_auto_corr_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(86.7, 3.3, data["cm_clust_tend_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(-2080, 70, data["cm_clust_shade_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(69000, 2100, data["cm_clust_prom_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(-0.175, 0.003, data["cm_info_corr1_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.813, 0.004, data["cm_info_corr2_d1_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.776, 0.001, data["rlm_sre_3d_avg_fbn_n32"])
    assert within_tolerance(3.55, 0.07, data["rlm_lre_3d_avg_fbn_n32"])
    assert within_tolerance(0.0204, 8e-04, data["rlm_lgre_3d_avg_fbn_n32"])
    assert within_tolerance(471, 9, data["rlm_hgre_3d_avg_fbn_n32"])
    assert within_tolerance(0.0187, 7e-04, data["rlm_srlge_3d_avg_fbn_n32"])
    assert within_tolerance(346, 7, data["rlm_srhge_3d_avg_fbn_n32"])
    assert within_tolerance(0.0313, 0.0016, data["rlm_lrlge_3d_avg_fbn_n32"])
    assert within_tolerance(1900, 20, data["rlm_lrhge_3d_avg_fbn_n32"])
    assert within_tolerance(4000, 10, data["rlm_glnu_3d_avg_fbn_n32"])
    assert within_tolerance(0.135, 0.003, data["rlm_glnu_norm_3d_avg_fbn_n32"])
    assert within_tolerance(16600, 300, data["rlm_rlnu_3d_avg_fbn_n32"])
    assert within_tolerance(0.559, 0.001, data["rlm_rlnu_norm_3d_avg_fbn_n32"])
    assert within_tolerance(0.664, 0.003, data["rlm_r_perc_3d_avg_fbn_n32"])
    assert within_tolerance(39.8, 0.9, data["rlm_gl_var_3d_avg_fbn_n32"])
    assert within_tolerance(1.26, 0.05, data["rlm_rl_var_3d_avg_fbn_n32"])
    assert within_tolerance(4.87, 0.03, data["rlm_rl_entr_3d_avg_fbn_n32"])
    assert within_tolerance(0.777, 0.001, data["rlm_sre_3d_v_mrg_fbn_n32"])
    assert within_tolerance(3.52, 0.07, data["rlm_lre_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.0204, 8e-04, data["rlm_lgre_3d_v_mrg_fbn_n32"])
    assert within_tolerance(471, 9, data["rlm_hgre_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.0186, 7e-04, data["rlm_srlge_3d_v_mrg_fbn_n32"])
    assert within_tolerance(347, 7, data["rlm_srhge_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.0311, 0.0016, data["rlm_lrlge_3d_v_mrg_fbn_n32"])
    assert within_tolerance(1890, 20, data["rlm_lrhge_3d_v_mrg_fbn_n32"])
    assert within_tolerance(51900, 200, data["rlm_glnu_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.135, 0.003, data["rlm_glnu_norm_3d_v_mrg_fbn_n32"])
    assert within_tolerance(215000, 4000, data["rlm_rlnu_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.56, 0.001, data["rlm_rlnu_norm_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.664, 0.003, data["rlm_r_perc_3d_v_mrg_fbn_n32"])
    assert within_tolerance(39.7, 0.9, data["rlm_gl_var_3d_v_mrg_fbn_n32"])
    assert within_tolerance(1.25, 0.05, data["rlm_rl_var_3d_v_mrg_fbn_n32"])
    assert within_tolerance(4.87, 0.03, data["rlm_rl_entr_3d_v_mrg_fbn_n32"])
    assert within_tolerance(0.676, 0.003, data["szm_sze_3d_fbn_n32"])
    assert within_tolerance(58600, 800, data["szm_lze_3d_fbn_n32"])
    assert within_tolerance(0.034, 4e-04, data["szm_lgze_3d_fbn_n32"])
    assert within_tolerance(286, 6, data["szm_hgze_3d_fbn_n32"])
    assert within_tolerance(0.0224, 4e-04, data["szm_szlge_3d_fbn_n32"])
    assert within_tolerance(186, 4, data["szm_szhge_3d_fbn_n32"])
    assert within_tolerance(105, 4, data["szm_lzlge_3d_fbn_n32"])
    assert within_tolerance(33600000, 3e+05, data["szm_lzhge_3d_fbn_n32"])
    assert within_tolerance(231, 6, data["szm_glnu_3d_fbn_n32"])
    assert within_tolerance(0.0414, 3e-04, data["szm_glnu_norm_3d_fbn_n32"])
    assert within_tolerance(2370, 40, data["szm_zsnu_3d_fbn_n32"])
    assert within_tolerance(0.424, 0.004, data["szm_zsnu_norm_3d_fbn_n32"])
    assert within_tolerance(0.126, 0.001, data["szm_z_perc_3d_fbn_n32"])
    assert within_tolerance(50.8, 0.9, data["szm_gl_var_3d_fbn_n32"])
    assert within_tolerance(58500, 800, data["szm_zs_var_3d_fbn_n32"])
    assert within_tolerance(6.57, 0.01, data["szm_zs_entr_3d_fbn_n32"])
    assert within_tolerance(0.527, 0.004, data["dzm_sde_3d_fbn_n32"])
    assert within_tolerance(12.6, 0.1, data["dzm_lde_3d_fbn_n32"])
    assert within_tolerance(0.034, 4e-04, data["dzm_lgze_3d_fbn_n32"])
    assert within_tolerance(286, 6, data["dzm_hgze_3d_fbn_n32"])
    assert within_tolerance(0.0228, 3e-04, data["dzm_sdlge_3d_fbn_n32"])
    assert within_tolerance(136, 4, data["dzm_sdhge_3d_fbn_n32"])
    assert within_tolerance(0.179, 0.004, data["dzm_ldlge_3d_fbn_n32"])
    assert within_tolerance(4850, 60, data["dzm_ldhge_3d_fbn_n32"])
    assert within_tolerance(231, 6, data["dzm_glnu_3d_fbn_n32"])
    assert within_tolerance(0.0414, 3e-04, data["dzm_glnu_norm_3d_fbn_n32"])
    assert within_tolerance(1500, 30, data["dzm_zdnu_3d_fbn_n32"])
    assert within_tolerance(0.269, 0.003, data["dzm_zdnu_norm_3d_fbn_n32"])
    assert within_tolerance(0.126, 0.001, data["dzm_z_perc_3d_fbn_n32"])
    assert within_tolerance(50.8, 0.9, data["dzm_gl_var_3d_fbn_n32"])
    assert within_tolerance(5.56, 0.05, data["dzm_zd_var_3d_fbn_n32"])
    assert within_tolerance(7.06, 0.01, data["dzm_zd_entr_3d_fbn_n32"])
    assert within_tolerance(0.000188, 4e-06, data["ngt_coarseness_3d_fbn_n32"])
    assert within_tolerance(0.0752, 0.0019, data["ngt_contrast_3d_fbn_n32"])
    assert within_tolerance(4.65, 0.1, data["ngt_busyness_3d_fbn_n32"])
    assert within_tolerance(574, 1, data["ngt_complexity_3d_fbn_n32"])
    assert within_tolerance(0.167, 0.006, data["ngt_strength_3d_fbn_n32"])
    assert within_tolerance(0.118, 0.001, data["ngl_lde_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(134, 3, data["ngl_hde_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(0.0154, 7e-04, data["ngl_lgce_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(502, 8, data["ngl_hgce_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(0.00388, 4e-05, data["ngl_ldlge_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(36.7, 0.5, data["ngl_ldhge_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(0.457, 0.031, data["ngl_hdlge_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(76000, 600, data["ngl_hdhge_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(8170, 130, data["ngl_glnu_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(0.184, 0.001, data["ngl_glnu_norm_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(2250, 30, data["ngl_dcnu_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(0.0505, 3e-04, data["ngl_dcnu_norm_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(1, 0, data["ngl_dc_perc_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(30.4, 0.8, data["ngl_gl_var_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(39.4, 1, data["ngl_dc_var_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(7.06, 0.02, data["ngl_dc_entr_d1_a0.0_3d_fbn_n32"])
    assert within_tolerance(0.0106, 1e-04, data["ngl_dc_energy_d1_a0.0_3d_fbn_n32"])


def test_all_features():
    """
    Compute all features, including non-ibsi compliant ones.
    """

    # Configure settings used for the digital phantom.
    general_settings = GeneralSettingsClass(by_slice=False, ibsi_compliant=False)

    image_interpolation_settings = ImageInterpolationSettingsClass(
        by_slice=general_settings.by_slice,
        anti_aliasing=False
    )

    feature_computation_parameters = FeatureExtractionSettingsClass(
        by_slice=general_settings.by_slice,
        no_approximation=False,
        ibsi_compliant=False,
        base_feature_families="all",
        base_discretisation_method="none",
        ivh_discretisation_method="none",
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
        roi_resegment_settings=ResegmentationSettingsClass(),
        perturbation_settings=ImagePerturbationSettingsClass(),
        img_transform_settings=image_transformation_settings,
        feature_extr_settings=feature_computation_parameters
    )

    data = extract_features(
        write_features=False,
        export_features=True,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_digital_phantom", "nifti", "image", "phantom.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_digital_phantom", "nifti", "mask", "mask.nii.gz"),
        settings=settings
    )

    data = data[0]

    # Check that data for non-ibsi-compliant features are computed.
    assert "morph_vol_dens_ombb" in data.columns.values
    assert "morph_area_dens_ombb" in data.columns.values
    assert "morph_vol_dens_mvee" in data.columns.values
    assert "morph_area_dens_mvee" in data.columns.values
    assert "ivh_auc" in data.columns.values
