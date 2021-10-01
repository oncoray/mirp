import pytest
import os
from mirp.experimentClass import ExperimentClass
from mirp.importSettings import SettingsClass, GeneralSettingsClass, ImagePostProcessingClass,\
    ImageInterpolationSettingsClass, RoiInterpolationSettingsClass, ResegmentationSettingsClass,\
    ImagePerturbationSettingsClass, ImageTransformationSettingsClass, FeatureExtractionSettingsClass

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_ibsi_1_digital_phantom():

    # Configure settings used for the digital phantom.
    general_settings = GeneralSettingsClass()
    general_settings.by_slice = False

    image_interpolation_settings = ImageInterpolationSettingsClass()
    image_interpolation_settings.interpolate = False
    image_interpolation_settings.anti_aliasing = False

    feature_computation_parameters = FeatureExtractionSettingsClass()
    feature_computation_parameters.discr_method = ["none"]
    feature_computation_parameters.ivh_discr_method = "none"
    feature_computation_parameters.glcm_dist = 1.0
    feature_computation_parameters.glcm_spatial_method = ["2d", "2.5d", "3d"]
    feature_computation_parameters.glcm_merge_method = ["average", "slice_merge", "dir_merge", "vol_merge"]
    feature_computation_parameters.glrlm_spatial_method = ["2d", "2.5d", "3d"]
    feature_computation_parameters.glrlm_merge_method = ["average", "slice_merge", "dir_merge", "vol_merge"]
    feature_computation_parameters.glszm_spatial_method = ["2d", "2.5d", "3d"]
    feature_computation_parameters.gldzm_spatial_method = ["2d", "2.5d", "3d"]
    feature_computation_parameters.ngtdm_spatial_method = ["2d", "2.5d", "3d"]
    feature_computation_parameters.ngldm_dist = 1.0
    feature_computation_parameters.ngldm_spatial_method = ["2d", "2.5d", "3d"]
    feature_computation_parameters.ngldm_diff_lvl = 0.0

    settings = SettingsClass(general_settings=general_settings,
                             post_process_settings=ImagePostProcessingClass(),
                             img_interpolate_settings=image_interpolation_settings,
                             roi_interpolate_settings=RoiInterpolationSettingsClass(),
                             roi_resegment_settings=ResegmentationSettingsClass(),
                             vol_adapt_settings=ImagePerturbationSettingsClass(),
                             img_transform_settings=ImageTransformationSettingsClass(),
                             feature_extr_settings=feature_computation_parameters)

    main_experiment = ExperimentClass(modality="CT",
                                      subject="phantom",
                                      cohort=None,
                                      write_path=None,
                                      image_folder=os.path.join(CURRENT_DIR, "data", "ibsi_1_digital_phantom", "nifti",
                                                                "image"),
                                      roi_folder=os.path.join(CURRENT_DIR, "data", "ibsi_1_digital_phantom", "nifti",
                                                              "mask"),
                                      roi_reg_img_folder=None,
                                      image_file_name_pattern=None,
                                      registration_image_file_name_pattern=None,
                                      roi_names=["mask"],
                                      data_str=None,
                                      provide_diagnostics=True,
                                      settings=settings,
                                      compute_features=True,
                                      extract_images=False,
                                      plot_images=False,
                                      keep_images_in_memory=False)

    data = main_experiment.process()

    1

test_ibsi_1_digital_phantom()
