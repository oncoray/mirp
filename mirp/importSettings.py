import xml.etree.ElementTree as ElemTree

import numpy as np
import pandas as pd


class SettingsClass:

    def __init__(self, general_settings, img_interpolate_settings, roi_interpolate_settings,
                 vol_adapt_settings, roi_resegment_settings, feature_extr_settings, img_transform_settings,
                 deep_learning_settings):
        self.general         = general_settings
        self.img_interpolate = img_interpolate_settings
        self.roi_interpolate = roi_interpolate_settings
        self.vol_adapt       = vol_adapt_settings
        self.roi_resegment   = roi_resegment_settings
        self.feature_extr    = feature_extr_settings
        self.img_transform   = img_transform_settings
        self.deep_learning   = deep_learning_settings


class GeneralSettingsClass:

    def __init__(self):
        self.by_slice = None
        self.config_str = ""
        self.divide_disconnected_roi = "combine"
        self.no_approximation = False


class RoiInterpolationSettingsClass:

    def __init__(self):
        # self.interpolate = None
        # self.method = "grid"        # Alternative: mesh for mesh-based interpolation
        self.spline_order = None
        self.incl_threshold = None
        # self.new_spacing = None


class ImagePerturbationSettingsClass:

    def __init__(self):

        self.roi_adapt_size = [0.0]
        self.roi_adapt_type = "distance"  # Alternatively, fraction for fractional volume growth and decreases
        self.rot_angles = [0.0]
        self.eroded_vol_fract = 0.8
        self.crop = False
        self.translate_frac = [0.0]
        self.add_noise = False
        self.noise_repetitions = 0
        self.noise_level = None
        self.randomise_roi = False
        self.roi_random_rep = 0
        self.drop_out_slice = 0

        # Division of roi into bulk and boundary
        self.bulk_min_vol_fract = 0.4
        self.roi_boundary_size = [0.0]

        # Selection of heterogeneous supervoxels
        self.heterogeneous_svx_count = 0.0
        self.heterogeneity_features  = ["rlm_sre"]
        self.heterogen_low_values    = [False]

        # Initially local variables
        self.translate_x = None
        self.translate_y = None


class ImageInterpolationSettingsClass:

    def __init__(self):
        self.interpolate = None
        self.spline_order = None
        self.new_spacing = [None]
        self.new_non_iso_spacing = [None]
        self.anti_aliasing = True
        self.smoothing_beta = 0.95


class ResegmentationSettingsClass:

    def __init__(self):
        self.method   = None
        self.g_thresh = [np.nan, np.nan]
        self.sigma    = None


class FeatureExtractionSettingsClass:

    def __init__(self):
        self.families = ["all"]
        self.discr_method = None
        self.discr_n_bins = None
        self.discr_bin_width = None
        self.loc_int_cval = None
        self.ivh_discr_method = "none"
        self.ivh_discr_n_bins     = 1000
        self.ivh_discr_bin_width  = None
        self.glcm_dist = None
        self.glcm_spatial_method = None
        self.glcm_merge_method = None
        self.glrlm_spatial_method = None
        self.glrlm_merge_method = None
        self.glszm_spatial_method = None
        self.gldzm_spatial_method = None
        self.gldzm_merge_method = None
        self.ngtdm_spatial_method = None
        self.ngtdm_merge_method = None
        self.ngldm_dist = None
        self.ngldm_diff_lvl = None
        self.ngldm_spatial_method = None
        self.ngldm_merge_method = None


class ImageTransformationSettingsClass:

    def __init__(self):
        self.perform_img_transform = False
        # self.interp_prior_filter = True
        self.spatial_filters = None
        self.wavelet_fam = None
        self.wavelet_rot_invar = False
        self.wavelet_stationary = True
        self.log_sigma = None
        self.log_average = False
        self.log_sigma_truncate = 4.0
        self.laws_calculate_energy = True
        self.laws_kernel = ["all"]
        self.laws_delta  = 7
        self.laws_rot_invar = True
        self.mean_filter_size = None
        self.boundary_condition = "nearest"


class DeepLearningSettingsClass:

    def __init__(self):
        self.expected_size = [np.nan, np.nan, np.nan]
        self.normalisation = "none"
        self.intensity_range = [np.nan, np.nan]


def str2list(strx, data_type, default=None):
    """ Function for splitting strings read from the xml file """

    # Check if strx is none
    if strx is None and default is None: return None
    elif strx is None and type(default) in [list, tuple]: return default
    elif strx is None and not type(default) in [list, tuple]: return [default]

    # If strx is an element, read string
    if type(strx) is ElemTree.Element:
        strx = strx.text

    # Repeat check
    if strx is None and default is None: return None
    elif strx is None and type(default) in [list, tuple]: return default
    elif strx is None and not type(default) in [list, tuple]: return [default]

    contents = strx.split(",")
    content_list = []

    if (len(contents) == 1) and (contents[0] == ""): return content_list

    for i in np.arange(0, len(contents)):
        append_data = str2type(contents[i], data_type)

        # Check append data for None
        if append_data is None and type(default) in [list, tuple]: return default
        elif append_data is None and not type(default) in [list, tuple]:  return [default]
        else: content_list.append(append_data)

    return content_list


def str2type(strx, data_type, default=None):
    # Check if strx is none
    if strx is None and default is None: return None
    elif strx is None: return default

    # If strx is an element, read string
    if type(strx) is ElemTree.Element:
        strx = strx.text

    # Test if the requested data type is not a string or path, but is empty
    if data_type not in ["str", "path"] and (strx == "" or strx is None):
        return default
    elif data_type in ["str", "path"] and (strx == "" or strx is None) and default is not None:
        return default

    # Casting of strings to different data types
    if data_type == "int":                                       return int(strx)
    if data_type == "bool":
        if strx in ("true", "True", "TRUE", "T", "t", "1"):      return True
        elif strx in ("false", "False", "FALSE", "F", "f", "0"): return False
    if data_type == "float":
        if strx in ("na", "nan", "NA", "NaN"):                   return np.nan
        elif strx in ("-inf", "-Inf", "-INF"):                   return -np.inf
        elif strx in ("inf", "Inf", "INF"):                      return np.inf
        else:                                                    return float(strx)
    if data_type == "str":                                       return strx
    if data_type == "path":                                      return strx


def import_configuration_settings(path):

    def slice_to_spatial(by_slice):
        if by_slice:
            return "2d"
        else:
            return "3d"

    # Load xml
    tree = ElemTree.parse(path)
    root = tree.getroot()

    # Empty list for settings
    settings_list = []

    for branch in root.findall("config"):

        # Process general settings
        general_branch   = branch.find("general")
        general_settings = GeneralSettingsClass()

        general_settings.by_slice = str2type(general_branch.find("by_slice"), "bool", False)
        general_settings.config_str = str2type(general_branch.find("config_str"), "str", "")
        general_settings.no_approximation = str2type(general_branch.find("no_approximation"), "bool", False)

        # Process image interpolation settings
        img_interp_branch   = branch.find("img_interpolate")
        img_interp_settings = ImageInterpolationSettingsClass()

        img_interp_settings.interpolate  = str2type(img_interp_branch.find("interpolate"), "bool")
        img_interp_settings.spline_order = str2type(img_interp_branch.find("spline_order"), "int")
        img_interp_settings.new_spacing  = np.unique(str2list(img_interp_branch.find("new_spacing"), "float"))
        img_interp_settings.new_non_iso_spacing = str2list(img_interp_branch.find("new_non_iso_spacing"), "float")
        img_interp_settings.anti_aliasing = str2type(img_interp_branch.find("anti_aliasing"), "bool", True)
        img_interp_settings.smoothing_beta = str2type(img_interp_branch.find("smoothing_beta"), "float", 0.95)

        # Process roi interpolation settings
        roi_interp_branch   = branch.find("roi_interpolate")
        roi_interp_settings = RoiInterpolationSettingsClass()
        roi_interp_settings.spline_order   = str2type(roi_interp_branch.find("spline_order"), "int", 1)
        roi_interp_settings.incl_threshold = str2type(roi_interp_branch.find("incl_threshold"), "float", 0.5)

        # Image and roi volume adaptation settings
        vol_adapt_branch = branch.find("vol_adapt")
        vol_adapt_settings = ImagePerturbationSettingsClass()

        if vol_adapt_branch is not None:
            vol_adapt_settings.roi_adapt_size          = str2list(vol_adapt_branch.find("roi_adapt_size"), "float", 0.0)
            vol_adapt_settings.roi_adapt_type          = str2type(vol_adapt_branch.find("roi_adapt_type"), "str", "distance")
            vol_adapt_settings.eroded_vol_fract        = str2type(vol_adapt_branch.find("eroded_vol_fract"), "float", 0.8)
            vol_adapt_settings.crop                    = str2type(vol_adapt_branch.find("resect"), "bool", False)
            vol_adapt_settings.rot_angles              = str2list(vol_adapt_branch.find("rot_angles"), "float", 0.0)
            vol_adapt_settings.translate_frac          = str2list(vol_adapt_branch.find("translate_frac"), "float", 0.0)
            vol_adapt_settings.noise_repetitions       = str2type(vol_adapt_branch.find("noise_repetitions"), "int", 0)
            vol_adapt_settings.noise_level             = str2type(vol_adapt_branch.find("noise_level"), "float")
            vol_adapt_settings.roi_random_rep          = str2type(vol_adapt_branch.find("roi_randomise_repetitions"), "int", 0)
            vol_adapt_settings.bulk_min_vol_fract      = str2type(vol_adapt_branch.find("bulk_min_vol_fract"), "float", 0.4)
            vol_adapt_settings.roi_boundary_size       = str2list(vol_adapt_branch.find("roi_boundary_size"), "float", 0.0)
            vol_adapt_settings.heterogeneous_svx_count = str2type(vol_adapt_branch.find("heterogeneous_svx_count"), "float", 0.0)
            vol_adapt_settings.heterogeneity_features  = str2list(vol_adapt_branch.find("heterogeneity_features"), "str", "rlm_sre")
            vol_adapt_settings.heterogen_low_values    = str2list(vol_adapt_branch.find("heterogeneity_low_values"), "bool", False)
            vol_adapt_settings.drop_out_slice = str2type(vol_adapt_branch.find("drop_out_slice"), "int", 0)
            if vol_adapt_settings.noise_repetitions > 0: vol_adapt_settings.add_noise = True
            if vol_adapt_settings.roi_random_rep > 0:    vol_adapt_settings.randomise_roi = True

        # Process roi segmentation settings
        roi_resegment_branch   = branch.find("roi_resegment")
        roi_resegment_settings = ResegmentationSettingsClass()

        if roi_resegment_branch is not None:
            roi_resegment_settings.method   = str2list(roi_resegment_branch.find("method"), "str")
            roi_resegment_settings.g_thresh = str2list(roi_resegment_branch.find("g_thresh"), "float")
            roi_resegment_settings.sigma    = str2type(roi_resegment_branch.find("sigma"), "float")

        # Process feature extraction settings
        feature_extr_branch   = branch.find("feature_extr")
        feature_extr_settings = FeatureExtractionSettingsClass()

        if feature_extr_branch is not None:
            feature_extr_settings.families             = str2list(feature_extr_branch.find("families"), "str", "all")
            feature_extr_settings.discr_method         = str2list(feature_extr_branch.find("discr_method"), "str", "none")
            feature_extr_settings.discr_n_bins         = str2list(feature_extr_branch.find("discr_n_bins"), "float")
            feature_extr_settings.discr_bin_width      = str2list(feature_extr_branch.find("discr_bin_width"), "float")
            feature_extr_settings.ivh_discr_method     = str2type(feature_extr_branch.find("ivh_discr_method"), "str")
            feature_extr_settings.ivh_discr_n_bins     = str2type(feature_extr_branch.find("ivh_discr_n_bins"), "float", 1000.0)
            feature_extr_settings.ivh_discr_bin_width  = str2type(feature_extr_branch.find("ivh_discr_bin_width"), "float")
            feature_extr_settings.glcm_dist            = str2list(feature_extr_branch.find("glcm_dist"), "float")
            feature_extr_settings.glcm_spatial_method  = str2list(feature_extr_branch.find("glcm_spatial_method"), "str", slice_to_spatial(general_settings.by_slice))
            feature_extr_settings.glcm_merge_method    = str2list(feature_extr_branch.find("glcm_merge_method"), "str")
            feature_extr_settings.glrlm_spatial_method = str2list(feature_extr_branch.find("glrlm_spatial_method"), "str", slice_to_spatial(general_settings.by_slice))
            feature_extr_settings.glrlm_merge_method   = str2list(feature_extr_branch.find("glrlm_merge_method"), "str")
            feature_extr_settings.glszm_spatial_method = str2list(feature_extr_branch.find("glszm_spatial_method"), "str", slice_to_spatial(general_settings.by_slice))
            feature_extr_settings.gldzm_spatial_method = str2list(feature_extr_branch.find("gldzm_spatial_method"), "str", slice_to_spatial(general_settings.by_slice))
            feature_extr_settings.ngtdm_spatial_method = str2list(feature_extr_branch.find("ngtdm_spatial_method"), "str", slice_to_spatial(general_settings.by_slice))
            feature_extr_settings.ngldm_dist           = str2list(feature_extr_branch.find("ngldm_dist"), "float")
            feature_extr_settings.ngldm_diff_lvl       = str2list(feature_extr_branch.find("ngldm_diff_lvl"), "float")
            feature_extr_settings.ngldm_spatial_method = str2list(feature_extr_branch.find("ngldm_spatial_method"), "str", slice_to_spatial(general_settings.by_slice))

        # Process image transformation settings
        img_transform_branch = branch.find("img_transform")
        img_transform_settings = ImageTransformationSettingsClass()

        if img_transform_branch is not None:
            img_transform_settings.perform_img_transform = str2type(img_transform_branch.find("perform_img_transform"), "bool", False)
            img_transform_settings.spatial_filters       = str2list(img_transform_branch.find("spatial_filters"), "str")
            img_transform_settings.wavelet_fam           = str2type(img_transform_branch.find("wavelet_fam"), "str")
            img_transform_settings.wavelet_rot_invar     = str2type(img_transform_branch.find("wavelet_rot_invar"), "bool", True)
            img_transform_settings.wavelet_stationary    = str2type(img_transform_branch.find("wavelet_stationary"), "bool", True)
            img_transform_settings.laws_calculate_energy = str2type(img_transform_branch.find("laws_calculate_energy"), "bool", True)
            img_transform_settings.laws_kernel           = str2list(img_transform_branch.find("laws_kernel"), "str", "all")
            img_transform_settings.laws_delta            = str2type(img_transform_branch.find("laws_delta"), "int", 7)
            img_transform_settings.laws_rot_invar        = str2type(img_transform_branch.find("laws_rot_invar"), "bool", True)
            img_transform_settings.log_sigma             = str2list(img_transform_branch.find("log_sigma"), "float")
            img_transform_settings.log_sigma_truncate    = str2type(img_transform_branch.find("log_sigma_truncate"), "float", 4.0)
            img_transform_settings.log_average           = str2type(img_transform_branch.find("log_average"), "bool")
            img_transform_settings.mean_filter_size      = str2type(img_transform_branch.find("mean_filter_size"), "int")
            img_transform_settings.boundary_condition    = str2type(img_transform_branch.find("boundary_condition"), "str", "nearest")

            if img_transform_settings.spatial_filters is None:
                img_transform_settings.perform_img_transform = False
        else:
            img_transform_settings.perform_img_transform = False

        # Deep learning branch
        deep_learning_branch = branch.find("deep_learning")
        deep_learning_settings = DeepLearningSettingsClass()

        if deep_learning_branch is not None:
            deep_learning_settings.expected_size = str2list(deep_learning_branch.find("expected_size"), "int", [np.nan, np.nan, np.nan])
            deep_learning_settings.normalisation = str2type(deep_learning_branch.find("normalisation"), "str", "none")
            deep_learning_settings.intensity_range = str2list(deep_learning_branch.find("intensity_range"), "float", [np.nan, np.nan])

        # Parse to settings
        settings_list.append(SettingsClass(general_settings=general_settings,
                                           img_interpolate_settings=img_interp_settings, roi_interpolate_settings=roi_interp_settings,
                                           vol_adapt_settings=vol_adapt_settings, roi_resegment_settings=roi_resegment_settings,
                                           feature_extr_settings=feature_extr_settings, img_transform_settings=img_transform_settings,
                                           deep_learning_settings=deep_learning_settings))

    return settings_list


def import_data_settings(path, config_settings, compute_features=False, extract_images=False, plot_images=False, keep_images_in_memory=False, file_structure=False):

    from mirp.experimentClass import ExperimentClass
    import os
    import logging

    def find_sub_directories(dir_path):
        sub_dir = []
        for dir_file in os.listdir(dir_path):
            if os.path.isdir(os.path.join(dir_path, dir_file)):
                sub_dir.append(dir_file)
        return sub_dir

    def find_imaging_files(dir_path):
        file_found = False
        for dir_file in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, dir_file)):
                if dir_file.lower().endswith((".dcm", ".ima", ".nii", ".nii.gz", ".nifti", ".nifti.gz", ".nrrd")):
                    file_found = True
                    break
        return file_found

    # Configure logger
    logging.basicConfig(format="%(levelname)s\t: %(processName)s \t %(asctime)s \t %(message)s", level=logging.INFO)

    # Load xml
    tree = ElemTree.parse(path)
    root = tree.getroot()

    # Empty list for iteratively storing data objects
    data_obj_list = []

    # Iterate over configurations
    for branch in root.findall("config"):

        # Read data from xml file
        paths_branch = branch.find("paths")
        project_path = os.path.normpath(str2type(paths_branch.find("project_folder"), "path"))
        write_path   = os.path.normpath(str2type(paths_branch.find("write_folder"), "path"))
        excl_subj    = str2list(paths_branch.find("subject_exclude"), "str")
        incl_subj    = str2list(paths_branch.find("subject_include"), "str")
        provide_diagnostics = str2type(paths_branch.find("provide_diagnostics"), "bool")

        # Read cohort name or ID
        cohort_id = str2type(paths_branch.find("cohort"), "str", "NA")

        # Identify subject folders
        folder_list = find_sub_directories(project_path)

        # Iterate over data branches
        for data_branch in branch.findall("data"):

            # Read current data branch
            image_modality = str2type(data_branch.find("modality"), "str")
            image_folder = str2type(data_branch.find("image_folder"), "path")
            roi_folder = str2type(data_branch.find("roi_folder"), "path")
            roi_reg_img_folder = str2type(data_branch.find("roi_reg_img_folder"), "str")
            roi_names = str2list(data_branch.find("roi_names"), "str")
            roi_list_path = str2type(data_branch.find("roi_list_path"), "str")
            divide_disconnected_roi = str2type(data_branch.find("divide_disconnected_roi"), "str", "combine")
            data_string = str2type(data_branch.find("data_str"), "str")
            extraction_config = str2list(data_branch.find("extraction_config"), "str")

            # Check if extraction config has been set -- this allows setting configurations for mixed modalities
            if extraction_config is not None and config_settings is not None:
                new_config_settings = []

                # Iterate over configuration names mentioned in the data and compare those to the configuration strings in the settings
                # If a match is found, the configuration is set to the new configuration list.
                for config_name in extraction_config:
                    for ii in np.arange(len(config_settings)):
                        if config_settings[ii].general.config_str == config_name:
                            new_config_settings.append(config_settings[ii])

                if len(new_config_settings) == 0:
                    raise ValueError("No matching configuration strings were found in the settings file.")

            elif type(config_settings) is list:
                new_config_settings = config_settings
            else:
                new_config_settings = [config_settings]

            if not file_structure:
                # Check if image_folder has been set
                if image_folder is None:
                    logging.warning("No image folder was set. If images are located directly in the patient folder, use subject_dir as tag")
                    continue

                # Set correct paths in case folders are tagged with subject_dir. This tag indicates that the data is directly in the subject folder
                if image_folder == "subject_dir": image_folder = ""
                if roi_folder   == "subject_dir": roi_folder   = ""

                # Perform consistency check for roi folder, roi_names
                if roi_folder is None:
                    logging.info("No roi folder was configured. The roi folder reverts to the image folder.")
                    roi_folder = image_folder

                if roi_folder is not None and roi_names is None and roi_list_path is None:
                    logging.warning("No roi names were provided with the configuration.")

                if roi_reg_img_folder is None:
                    roi_reg_img_folder = image_folder

            else:
                image_folder = roi_folder = roi_reg_img_folder = ""

            # A separate file with roi names per sample may be provided in case there is no standardized name
            if roi_list_path is not None:
                roi_list = pd.read_csv(roi_list_path, sep=None, engine="python")

                # Convert sample_id to string
                roi_list["sample_id"] = roi_list["sample_id"].astype(str)

                # Check if column names are correct
                if not all([ii in roi_list.columns.values for ii in ["sample_id", "roi_name"]]):
                    raise ValueError("Column names in the provided roi list do not match \"sample_id\" or \"roi_name\".")
            else:
                roi_list = None

            # Iterate over subjects
            for curr_subj in folder_list:

                # Check if the current subject is included in the analysis
                if incl_subj is not None:
                    if curr_subj not in incl_subj:
                        logging.info("%s was excluded as per configuration.", curr_subj)
                        continue

                # Check if the current subject is excluded from the analysis
                if excl_subj is not None:
                    if curr_subj in excl_subj:
                        logging.info("%s was excluded as per configuration.", curr_subj)
                        continue

                # Set image folder and roi folder paths
                image_dir_subj   = os.path.normpath(os.path.join(project_path, curr_subj, image_folder))
                roi_dir_subj     = os.path.normpath(os.path.join(project_path, curr_subj, roi_folder))
                roi_reg_img_subj = os.path.normpath(os.path.join(project_path, curr_subj, roi_reg_img_folder))

                # Check if image and roi folders exist on the path
                if not os.path.isdir(image_dir_subj):
                    logging.info("%s was excluded as the image folder %s was not found.", curr_subj, os.path.join(image_dir_subj))
                    continue

                if not os.path.isdir(roi_dir_subj):
                    logging.info("%s was excluded as the roi folder %s was not found.", curr_subj, os.path.join(roi_dir_subj))
                    continue

                # Check if the image and roi folders contain files
                if not find_imaging_files(image_dir_subj) and not file_structure:
                    logging.info("%s was excluded as the image folder did not contain image files.", curr_subj)
                    continue

                if not find_imaging_files(roi_dir_subj) and not file_structure:
                    logging.info("%s was excluded as the roi folder did not contain image files.", curr_subj)
                    continue

                # Get rois for the current sample in case a roi_list was provided.
                if roi_list is not None:
                    roi_names = roi_list.loc[roi_list.sample_id == curr_subj, "roi_name"].values

                # Create data class object and add to list
                # For better parallellisation performance a data object only contains a single configurations file
                for curr_config_setting in new_config_settings:

                    # Set divide_disconnected_roi setting
                    if curr_config_setting is not None:
                        curr_config_setting.general.divide_disconnected_roi = divide_disconnected_roi

                    data_obj = ExperimentClass(modality=image_modality, subject=curr_subj, cohort=cohort_id, write_path=write_path,
                                               image_folder=image_dir_subj, roi_folder=roi_dir_subj, roi_reg_img_folder=roi_reg_img_subj,
                                               roi_names=roi_names, data_str=data_string, provide_diagnostics=provide_diagnostics,
                                               settings=curr_config_setting, compute_features=compute_features, extract_images=extract_images,
                                               plot_images=plot_images, keep_images_in_memory=keep_images_in_memory)
                    data_obj_list.append(data_obj)

    return data_obj_list
