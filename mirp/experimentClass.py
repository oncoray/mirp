# from mirp.configThreading import disable_multi_threading
# disable_multi_threading()

import numpy as np
import os
import pandas as pd
import sys

from warnings import warn
from typing import List, Optional
from mirp.settings.settingsGeneric import SettingsClass
from mirp.utilities import expand_grid


class ExperimentClass:

    def __init__(self,
                 modality: str,
                 subject: str,
                 cohort: Optional[str],
                 image_folder: Optional[str],
                 roi_folder: Optional[str],
                 roi_reg_img_folder: Optional[str],
                 image_file_name_pattern: Optional[str],
                 registration_image_file_name_pattern: Optional[str],
                 roi_names: Optional[List[str]],
                 data_str: Optional[List[str]],
                 write_path: Optional[str],
                 settings: SettingsClass,
                 provide_diagnostics: bool = False,
                 compute_features: bool = False,
                 extract_images: bool = False,
                 plot_images: bool = False,
                 keep_images_in_memory: bool = False):
        """
        Attributes for an experiment.
        :param modality: modality of the requested image
        :param subject: sample identifier
        :param cohort: cohort identifier
        :param image_folder: full path to folder containing the requested image
        :param roi_folder: full path to folder containing the region/volume of interest definition
        :param roi_reg_img_folder: (optional) full path to folder containing the image on which the roi was defined. If the image used for roi registration and the requested image
        have the same coordinate system, the roi is transferred to the requested image.
        :param roi_names: name(s) of the requested rois
        :param data_str: string that is used as a data descriptor
        :param write_path: full path to folder used for writing output
        :param settings: settings object used for providing the configuration
        :param provide_diagnostics: flag to extract diagnostic features (default: False)
        :param compute_features: flag to compute features (default: False)
        :param extract_images: flag to extract images and mask in Nifti format (default: False)
        :param plot_images: flag to plot images and masks as .png (default: False)
        :param keep_images_in_memory: flag to keep images in memory. This avoids repeated loading of images, but at the expense of memory.
        """
        import datetime

        # General data
        self.modality = modality  # Image modality
        self.subject = subject  # Patient ID
        self.cohort = cohort  # Cohort name or id

        # Path for writing data
        self.write_path = write_path
        if self.write_path is not None:
            if not os.path.isdir(self.write_path):
                os.makedirs(self.write_path)

        # Paths to image and segmentation folders
        self.image_folder = image_folder
        self.roi_folder = roi_folder
        self.roi_reg_img_folder = roi_reg_img_folder  # Folder containing the image on which the roi was registered

        # File name patterns
        self.image_file_name_pattern = image_file_name_pattern  # Main image
        self.registration_image_file_name_pattern = registration_image_file_name_pattern  # Image against which segmentation was registered.

        # Segmentation names
        self.roi_names: List[str] = roi_names

        # Identifier strings
        self.data_str: List[str] = [] if data_str is None else data_str

        # Date at analysis start
        self.date = datetime.date.today().isoformat()

        # Settings and iteration settings
        self.settings = settings
        self.iter_settings = None

        # Process parameters
        self.provide_diagnostics: bool = provide_diagnostics  # Flag for writing diagnostics features
        self.compute_features: bool = compute_features
        self.extract_images: bool = extract_images
        self.plot_images: bool = plot_images
        self.keep_images_in_memory: bool = keep_images_in_memory

    def get_roi_list(self):
        """ Extracts the available region of interest from the roi folder. This function allows identification of
         regions of interest in new projects. """

        import logging
        from mirp.imageRead import find_regions_of_interest

        # Notify log of roi name extraction
        logging.info("Starting extraction of rois for %s.", self.subject)

        # Extract regions of interest
        roi_table = find_regions_of_interest(roi_folder=self.roi_folder, subject=self.subject)

        return roi_table

    def get_imaging_parameter_table(self):
        """
        Extracts image metadata from the image folder. This enables collection of scanner acquisition protocols and settings.
        :return: pandas table with imaging parameters
        """

        import logging
        from mirp.imageRead import find_imaging_parameters

        # Notify for start extraction of image meta data
        logging.info("Starting extraction of image metadata for %s.", self.subject)

        # Find files and extract meta data to a dataframe
        metadata_table = find_imaging_parameters(image_folder=self.image_folder,
                                                 modality=self.modality,
                                                 subject=self.subject,
                                                 plot_images=self.plot_images,
                                                 write_folder=self.write_path,
                                                 roi_folder=self.roi_folder,
                                                 registration_image_folder=self.roi_reg_img_folder,
                                                 settings=self.settings,
                                                 roi_names=self.roi_names)

        return metadata_table

    def get_file_structure_information(self, include_path_info=False):
        # Extract image metadata
        import logging
        import os

        from mirp.imageMetaData import get_image_directory_meta_data

        # Iterate over directories
        sub_dirs = [dirs[0] for dirs in os.walk(self.image_folder)]

        image_in_dir = []

        logging.info(f"Starting extraction of image metadata for {self.subject}.")

        # Find directories with images, based on file extensions
        for curr_dirr in sub_dirs:
            files_in_dir = os.listdir(path=os.path.join(self.image_folder, curr_dirr))
            image_in_dir += [any([file_name.lower().endswith((".dcm", ".ima", ".nii", ".nii.gz", ".nrrd")) for file_name in files_in_dir])]

        # Remove subdirectories without image files
        sub_dirs = [os.path.join(self.image_folder, sub_dirs[ii]) for ii in range(len(sub_dirs)) if image_in_dir[ii]]

        # Check if the list is empty
        if len(sub_dirs) == 0:
            return None

        dir_meta_list = []

        for curr_dirr in sub_dirs:
            dir_meta_list += [get_image_directory_meta_data(image_folder=curr_dirr, subject=self.subject)]

        if len(dir_meta_list) > 0:
            # Concatenate list of sub-directory meta data
            df_meta = pd.concat(dir_meta_list, axis=0, sort=False)

            # Sort values
            df_meta.sort_values(by=["patient_name", "study_date", "study_instance_uid", "modality", "instance_number"], inplace=True)

            if not include_path_info:
                # Drop path
                df_meta.drop(labels=["file_path"], axis=1, inplace=True)

                # Drop duplicates
                df_meta.drop_duplicates(subset=["patient_name", "study_instance_uid", "series_instance_uid", "modality"], keep="last", inplace=True)
        else:
            logging.warning("No image data could be found for %s.", self.subject)
            df_meta = []

        return df_meta

    def restructure_files(self, file, use_folder_name=True):
        import os
        import shutil

        # Read the csv file
        df_assign = pd.read_csv(filepath_or_buffer=os.path.join(self.write_path, file), sep=";",
                                usecols=["patient_name", "study_instance_uid", "series_instance_uid", "modality", "assigned_folder"],
                                dtype=object)

        # Keep only non-NA assignments
        df_assign: pd.DataFrame = df_assign.loc[df_assign.assigned_folder.notna(), ]

        # Pad uids with leading 0s in case they were dropped in the csv.
        df_assign.study_instance_uid = np.array([val_str.rjust(6, "0") for val_str in df_assign.study_instance_uid.values])
        df_assign.series_instance_uid = np.array([val_str.rjust(6, "0") for val_str in df_assign.series_instance_uid.values])

        # Get meta data
        df_files = self.get_file_structure_information(include_path_info=True)

        # Merge both sets and keep only the relevant columns
        df_assign = df_assign.merge(right=df_files, on=["patient_name", "study_instance_uid", "series_instance_uid", "modality"], how="inner")
        df_assign = df_assign[["file_path", "assigned_folder", "patient_name"]]

        for idx, row in df_assign.iterrows():
            if use_folder_name:
                # Get name for new directory where the images are copied to
                new_dir = os.path.join(self.write_path, self.subject, row.assigned_folder)
            else:
                new_dir = os.path.join(self.write_path, row.patient_name, row.assigned_folder)

            # Create new directory, if required
            os.makedirs(new_dir, exist_ok=True)

            # Copy file
            shutil.copy2(src=row.file_path, dst=new_dir)

    @staticmethod
    def get_iterable_parameters(settings: SettingsClass):
        """
        Settings may be iterated, e.g. to extract multi-scale features or for image perturbations.
        There are two types of iterable settings. One type involves changing the image intensities in the object.
        The other type merely involves changing the ROI. The first type requires an outer loop as smaller loops
        are not feasible for parallel processing due to memory requirements.

        :param settings: settingsClass object that contains configuration settings
        :return:
        """

        #################################################################
        # Iterables that require outer looping
        #################################################################

        # Image rotation
        rot_angles = settings.perturbation.rotation_angles

        # Noise addition
        noise_reps = settings.perturbation.noise_repetitions
        if noise_reps > 0:
            noise_reps = np.arange(0, noise_reps)
        else:
            noise_reps = [0]

        # Image translation
        translate_frac = settings.perturbation.translation_fraction
        if not settings.general.by_slice:
            translate_frac_z = settings.perturbation.translation_fraction
        else:
            translate_frac_z = [0.0]

        # Multi-scale features
        vox_spacing = settings.img_interpolate.new_spacing
        if vox_spacing is None:
            vox_spacing = [None]

        # Generate outer loop permutations
        iter_settings = expand_grid({"rot_angle": rot_angles,
                                     "noise_repetition": noise_reps,
                                     "translate_x": translate_frac,
                                     "translate_y": translate_frac,
                                     "translate_z": translate_frac_z,
                                     "vox_spacing": vox_spacing})
        iter_settings.transpose()

        #################################################################
        # Determine number of iterations requires
        #################################################################

        # Outer loop iterations
        n_outer_iter = len(rot_angles) * len(noise_reps) * len(translate_frac)**2 * len(translate_frac_z) * len(vox_spacing)

        # Roi iterations
        if settings.perturbation.randomise_roi:
            roi_random_rep = settings.perturbation.roi_random_rep
        else:
            roi_random_rep = 1

        n_inner_iter = len(settings.perturbation.roi_adapt_size) * roi_random_rep

        return iter_settings, n_outer_iter, n_inner_iter

    def process(self):
        """ Main pipeline """
        return None

    def process_deep_learning(self,
                              output_slices=False,
                              crop_size=None,
                              center_crops_per_slice=True,
                              remove_empty_crops=True,
                              intensity_range=None,
                              normalisation="none",
                              as_numpy=False):
        return None

    def collect_features(self, img_obj, roi_list, feat_list, settings: SettingsClass):
        """
        Combine separate feature tables into one single table.
        :param img_obj:
        :param roi_list:
        :param feat_list:
        :param settings:
        :return:
        """

        ########################################################################################################
        # Additional descriptors for experimental settings and diagnostics
        ########################################################################################################

        # Get voxel spacing
        if img_obj.is_missing:
            voxel_size = np.nan
        else:
            voxel_size = [np.min(roi_obj.roi.spacing) for roi_obj in roi_list]

        # Add descriptions for experimental settings
        df_img_data = pd.DataFrame({"id_subject": self.subject,
                                    "id_cohort": self.cohort,
                                    "img_data_settings_id": settings.general.config_str,
                                    "img_data_modality": self.modality,
                                    "img_data_config": "_".join(self.data_str),
                                    "img_data_noise_level": img_obj.noise,
                                    "img_data_noise_iter": img_obj.noise_iter,
                                    "img_data_rotation_angle": settings.perturbation.rotation_angles,
                                    "img_data_roi_randomise_iter": [roi_obj.svx_randomisation_id for roi_obj in roi_list],
                                    "img_data_roi_adapt_size": [roi_obj.adapt_size for roi_obj in roi_list],
                                    "img_data_translate_x": settings.perturbation.translate_x,
                                    "img_data_translate_y": settings.perturbation.translate_y,
                                    "img_data_translate_z": settings.perturbation.translate_z,
                                    "img_data_voxel_size": voxel_size,
                                    "img_data_roi": [roi_obj.name for roi_obj in roi_list]},
                                   index=np.arange(len(roi_list)))

        if len(df_img_data) == 0:
            df_img_data = None

        feat_list.insert(0, df_img_data)  # Add to front of list

        # Add diagnostic features (optional)
        if self.provide_diagnostics:
            # Parse diagnostic data attached to the different rois to one single list
            df_diag_data = self.collect_diagnostic_data(roi_list=roi_list)

            # Add to second position in the list
            feat_list.insert(1, df_diag_data)

        ########################################################################################################
        # Concatenation of results for the current image adaptation
        ########################################################################################################

        # Strip empty entries.
        feat_list = [df.reset_index(drop=True) for df in feat_list if df is not None]

        if len(feat_list) > 0:
            # Concatenate features - first reset indices for correct alignment
            return pd.concat(feat_list, axis=1)
        else:
            return None

    def extract_diagnostic_features(self, img_obj, roi_list=None, append_str=""):
        """ Extracts diagnostics features from image objects and lists of roi objects """

        if self.provide_diagnostics and roi_list is not None:

            # The image diagnostic features are calculated only once, yet are written on every row by concatenation
            # roi diagnostic features
            img_diag_feat = img_obj.compute_diagnostic_features(append_str="_".join([append_str, "img", self.modality]).strip("_"))

            for curr_roi in roi_list:
                # Add image diagnostic data to roi
                curr_roi.diagnostic_list += [img_diag_feat]

                # Add roi diagnostic data to roi
                curr_roi.compute_diagnostic_features(img_obj=img_obj, append_str="_".join([append_str, "roi"]).strip("_"))

    def collect_diagnostic_data(self, roi_list):
        """ Compiles diagnostic data from regions of interest """

        if self.provide_diagnostics:
            # Instantiate feature list
            diag_feat_list = []

            # Iterate over rois and combine lists of diagnostic tables into single frame, representing a row
            for curr_roi in roi_list:

                diag_feat_list += [pd.concat(curr_roi.diagnostic_list, axis=1)]

            # Combine rows into single table
            if len(diag_feat_list) > 0:
                df_diag_feat = pd.concat(diag_feat_list, axis=0)

                return df_diag_feat
            else:
                return None

    def set_image_name(self, img_obj):
        """
        Sets the name of image based on parameters of the corresponding data object
        :param img_obj:
        :return:
        """

        name_string = []

        # Add cohort and subject or just subject
        if self.cohort != "NA" and self.cohort is not None:
            name_string += [self.cohort, self.subject]
        else:
            name_string += [self.subject]

        # Add modality
        name_string += [self.modality]

        # Add data string
        if self.data_str is not None:
            name_string += self.data_str

        # Add configuration string
        if self.settings.general.config_str != "":
            name_string += [self.settings.general.config_str]

        # Set name
        img_obj.name = "_".join(name_string).replace(" ", "_")

    def _message_computation_initialisation(self):

        image_descriptor = "_".join(self.data_str).strip("_")

        message_str = ["Initialising"]
        if self.compute_features and self.extract_images:
            message_str += ["feature computation and image extraction"]
        elif self.compute_features:
            message_str += ["feature computation"]
        elif self.extract_images:
            message_str += ["image extraction"]

        message_str += [f"using {image_descriptor} images"]

        if self.settings.general.config_str != "":
            message_str += [f"and configuration \"{self.settings.general.config_str}\""]

        message_str += [f"for {self.subject}."]

        return " ".join(message_str)

    def _message_warning_no_features_extracted(self):
        image_descriptor = "_".join(self.data_str).strip("_")

        message_str = [f"No features were extracted from {image_descriptor} images"]

        if self.settings.general.config_str != "":
            message_str += [f"using configuration \"{self.settings.general.config_str}\""]

        message_str += [f"for {self.subject}."]

        return " ".join(message_str)

    def _message_feature_extraction_finished(self):
        image_descriptor = "_".join(self.data_str).strip("_")

        message_str = [f"Features were successfully extracted from {image_descriptor} images"]

        if self.settings.general.config_str != "":
            message_str += [f"using configuration \"{self.settings.general.config_str}\""]

        message_str += [f"for {self.subject}."]

        return " ".join(message_str)

    def _create_base_file_name(self):

        basename = []
        if self.subject is not None and self.subject != "":
            basename += [self.subject]

        if self.data_str is not None and self.data_str != "":
            basename += self.data_str

        if self.date is not None and self.date != "":
            basename += [self.date]

        if self.settings.general.config_str is not None and self.settings.general.config_str != "":
            basename += [self.settings.general.config_str]

        return "_".join(basename).replace(" ", "_")
