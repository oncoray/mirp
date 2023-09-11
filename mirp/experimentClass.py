# from mirp.configThreading import disable_multi_threading
# disable_multi_threading()

import numpy as np
import os
import pandas as pd
from typing import List, Optional
from mirp.settings.settingsGeneric import SettingsClass


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