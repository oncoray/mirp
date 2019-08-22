# from mirp.configThreading import disable_multi_threading
# disable_multi_threading()

import numpy as np
import pandas as pd

from mirp.utilities import expand_grid


class ExperimentClass:

    def __init__(self, modality, subject, cohort, image_folder, roi_folder, roi_reg_img_folder, roi_names, data_str, write_path,
                 settings, provide_diagnostics=False, compute_features=False, extract_images=False, plot_images=False,
                 keep_images_in_memory=False):
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

        # Initial settings
        self.modality     = modality                        # Image modality
        self.subject      = subject                         # Patient ID
        self.cohort       = cohort                          # Cohort name or id
        self.write_path   = write_path                      # Path for writing results
        self.image_folder = image_folder                    # Path to images
        self.roi_folder   = roi_folder                      # Path to roi
        self.roi_reg_img_folder = roi_reg_img_folder        # Path to image on which the roi is registered
        self.roi_names    = roi_names                       # Roi names
        if data_str is None: data_str = ""                  # Data string to specify particular settings, e.g. 0W or pt
        self.data_str     = data_str
        self.date = datetime.date.today().isoformat()       # Date at analysis start

        self.settings = settings
        self.iter_settings = None

        # Process parameters
        self.provide_diagnostics = provide_diagnostics  # Flag for writing diagnostics features
        self.compute_features = compute_features
        self.extract_images = extract_images
        self.plot_images = plot_images
        self.keep_images_in_memory = keep_images_in_memory

    def getRoiList(self):
        """ Extracts the available region of interest from the roi folder. This function allows identification of
         regions of interest in new projects. """

        import logging
        from mirp.imageRead import find_regions_of_interest

        # Notify log of roi name extraction
        logging.info("Starting extraction of rois for %s.", self.subject)

        # Extract regions of interest
        df_roi = find_regions_of_interest(roi_folder=self.roi_folder, subject=self.subject)

        return df_roi

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
        df_meta = find_imaging_parameters(image_folder=self.image_folder, modality=self.modality, subject=self.subject, plot_images=self.plot_images,
                                          write_folder=self.write_path, roi_folder=self.roi_folder, roi_reg_img_folder=self.roi_reg_img_folder,
                                          settings=self.settings, roi_names=self.roi_names)

        return df_meta

    def get_file_structure_information(self, include_path_info=False):
        # Extract image metadata
        import logging
        import os

        from mirp.imageMetaData import get_image_directory_meta_data

        # Iterate over directories
        sub_dirs = [dirs[0] for dirs in os.walk(self.image_folder)]

        image_in_dir = []

        logging.info("Starting extraction of image metadata for %s.", self.subject)

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
        df_assign = df_assign.loc[df_assign.assigned_folder.notna(), ]

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

    def get_iterable_parameters(self, settings):
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
        rot_angles = settings.vol_adapt.rot_angles

        # Noise addition
        noise_reps  = settings.vol_adapt.noise_repetitions
        if noise_reps > 0:
            noise_reps = np.arange(0, noise_reps)
        else:
            noise_reps = [0]

        # Image translation
        translate_frac = settings.vol_adapt.translate_frac
        if not settings.general.by_slice:
            translate_frac_z = settings.vol_adapt.translate_frac
        else:
            translate_frac_z = [0.0]

        # Multi-scale features
        vox_spacing = settings.img_interpolate.new_spacing

        # Generate outer loop permutations
        iter_settings = expand_grid({"rot_angle":      rot_angles,
                                     "noise_repetition": noise_reps,
                                     "translate_x":    translate_frac,
                                     "translate_y":    translate_frac,
                                     "translate_z":    translate_frac_z,
                                     "vox_spacing":    vox_spacing})
        iter_settings.transpose()

        #################################################################
        # Determine number of iterations requires
        #################################################################

        # Outer loop iterations
        n_outer_iter = len(rot_angles) * len(noise_reps) * len(translate_frac)**2 * len(translate_frac_z) * len(vox_spacing)

        # Roi iterations
        if settings.vol_adapt.randomise_roi:
            roi_random_rep = settings.vol_adapt.roi_random_rep
        else:
            roi_random_rep = 1

        n_inner_iter = len(settings.vol_adapt.roi_adapt_size) * roi_random_rep

        return iter_settings, n_outer_iter, n_inner_iter

    def process(self):
        """ Main pipeline """

        import os
        import logging
        from mirp.imageRead import load_image
        from mirp.imageProcess import crop_image, estimate_image_noise, interpolate_image, interpolate_roi, divide_tumour_regions, resegmentise, calculate_features, transform_images
        from mirp.imagePerturbations import rotate_image, adapt_roi_size, randomise_roi_contours
        import copy

        # Configure logger
        logging.basicConfig(format="%(levelname)s\t: %(processName)s \t %(asctime)s \t %(message)s", level=logging.INFO)

        # Initialise empty feature list
        feat_list = []

        # Notifications
        if self.compute_features and self.extract_images:
            logging.info("Initialising feature computation and image extraction using %s images and configuration %s for %s.", self.modality + "_" + self.data_str + "_", self.settings.general.config_str, self.subject)
        elif self.compute_features:
            logging.info("Initialising feature computation using %s images and configuration %s for %s.", self.modality + "_" + self.data_str + "_", self.settings.general.config_str, self.subject)
        elif self.extract_images:
            logging.info("Initialising image extraction using %s images and configuration %s for %s.", self.modality + "_" + self.data_str + "_", self.settings.general.config_str, self.subject)

        # Get iterables from current settings which lead to different image adaptations
        iter_set, n_outer_iter, n_inner_iter = self.get_iterable_parameters(settings=self.settings)

        # Load image and roi
        if self.keep_images_in_memory:
            base_img_obj, base_roi_list = load_image(image_folder=self.image_folder, roi_folder=self.roi_folder, roi_reg_img_folder=self.roi_reg_img_folder,
                                                     settings=self.settings, modality=self.modality, roi_names=self.roi_names)
            self.set_image_name(img_obj=base_img_obj)
        else:
            base_img_obj = base_roi_list = None

        # Iterate over iterable settings
        for ii in np.arange(0, n_outer_iter):

            # Log current iteration
            if n_outer_iter * n_inner_iter > 1:
                if n_inner_iter > 1:
                    logging.info("Starting computations for %s to %s of %s adaptations.", str(ii*n_inner_iter+1), str((ii+1)*n_inner_iter), str(n_outer_iter*n_inner_iter))
                else:
                    logging.info("Starting computations for %s of %s adaptations.", str(ii+1), str(n_outer_iter))
            else:
                logging.info("Starting computations.")

            ########################################################################################################
            # Load and pre-process image and roi
            ########################################################################################################

            # Set verbosity for loading roi - only report issues on first image adaptation
            if ii == 0:
                verbosity_roi = True
            else:
                verbosity_roi = False

            # Use pre-loaded base image and roi_list (more memory used, but may be faster if loading over a network), or load from disk.
            if self.keep_images_in_memory:
                img_obj = base_img_obj.copy()
                roi_list = copy.deepcopy(base_roi_list)
            else:
                # Read image and ROI segmentations
                img_obj, roi_list = load_image(image_folder=self.image_folder, roi_folder=self.roi_folder, roi_reg_img_folder=self.roi_reg_img_folder,
                                               settings=self.settings, modality=self.modality, roi_names=self.roi_names)
                self.set_image_name(img_obj=img_obj)

            # Check integrity of the list of regions of interest
            # roi_integrity_flag, roi_list = self.checkRoiListIntegrity(roi_list=roi_list, verbose=verbosity_roi)
            # if not roi_integrity_flag: continue

            # Crop slice stack (z only)
            if self.settings.vol_adapt.crop:
                img_obj, roi_list = crop_image(img_obj=img_obj, roi_list=roi_list, boundary=150.0, z_only=True)

            # Extract diagnostic features from initial image and rois
            self.extractDiagnosticFeatures(img_obj=img_obj, roi_list=roi_list, append_str="init")

            ########################################################################################################
            # Update settings and initialise
            ########################################################################################################

            # Copy settings for current iteration run - this allows local changes to curr_setting
            curr_setting = copy.deepcopy(self.settings)

            # Update settings object with iterable settings
            curr_setting.vol_adapt.rot_angles = [iter_set.rot_angle[ii]]
            curr_setting.img_interpolate.new_spacing = [iter_set.vox_spacing[ii]]
            curr_setting.vol_adapt.translate_x = [iter_set.translate_x[ii]]
            curr_setting.vol_adapt.translate_y = [iter_set.translate_y[ii]]
            curr_setting.vol_adapt.translate_z = [iter_set.translate_z[ii]]

            ########################################################################################################
            # Determine image noise levels (optional)
            ########################################################################################################

            # Initialise noise level with place holder value
            est_noise_level = -1.0

            # Determine image noise levels
            if curr_setting.vol_adapt.add_noise and curr_setting.vol_adapt.noise_level is None and est_noise_level == -1.0:
                est_noise_level = estimate_image_noise(img_obj=img_obj, settings=curr_setting, method="chang")
            elif curr_setting.vol_adapt.add_noise:
                est_noise_level = curr_setting.vol_adapt.noise_level

            ########################################################################################################
            # Base image-based operations - basic operations on base image (rotation, cropping, noise addition)
            # Note interpolation and translation are performed simultaneously, and interpolation is only done after
            # application of spatial filters
            ########################################################################################################

            # Rotate object
            img_obj, roi_list = rotate_image(img_obj=img_obj, roi_list=roi_list, settings=curr_setting)

            # Crop image to a box extending at most 15 cm around the combined ROI
            if curr_setting.vol_adapt.crop:
                img_obj, roi_list = crop_image(img_obj=img_obj, roi_list=roi_list, boundary=150.0, z_only=False)

            # Add random noise to an image
            if curr_setting.vol_adapt.add_noise:
                img_obj.add_noise(noise_level=est_noise_level, noise_iter=ii)

            ########################################################################################################
            # Interpolation of base image
            ########################################################################################################

            # Translate and interpolate image to isometric voxels
            img_obj = interpolate_image(img_obj=img_obj, settings=curr_setting)
            roi_list = interpolate_roi(roi_list=roi_list, img_obj=img_obj, settings=curr_setting)
            self.extractDiagnosticFeatures(img_obj=img_obj, roi_list=roi_list, append_str="interp")

            ########################################################################################################
            # ROI-based operations
            # These operations only affect the regions of interest
            ########################################################################################################

            # Adapt roi sizes by dilation and erosion
            roi_list = adapt_roi_size(roi_list=roi_list, settings=curr_setting)

            # Update roi using SLIC
            roi_list = randomise_roi_contours(roi_list=roi_list, img_obj=img_obj, settings=curr_setting)

            # Extract boundaries and tumour bulk
            roi_list = divide_tumour_regions(roi_list=roi_list, settings=curr_setting)

            # Resegmentise ROI based on intensities in the base images
            roi_list = resegmentise(img_obj=img_obj, roi_list=roi_list, settings=curr_setting)
            self.extractDiagnosticFeatures(img_obj=img_obj, roi_list=roi_list, append_str="reseg")

            # Compose ROI of heterogeneous supervoxels
            # roi_list = imageProcess.selectHeterogeneousSuperVoxels(img_obj=img_obj, roi_list=roi_list, settings=curr_setting,
            #                                                        file_str=os.path.join(self.write_path, self.subject + "_" + self.modality + "_" + self.data_str + "_" + self.date))

            ########################################################################################################
            # Base image computations and exports
            ########################################################################################################

            if self.extract_images:
                img_obj.export(file_path=self.write_path)
                for roi_obj in roi_list:
                    roi_obj.export(img_obj=img_obj, file_path=self.write_path)

            iter_feat_list = []
            if self.compute_features:
                iter_feat_list.append(calculate_features(img_obj=img_obj, roi_list=roi_list, settings=curr_setting))

            ########################################################################################################
            # Image transformations
            ########################################################################################################

            if self.settings.img_transform.perform_img_transform:
                # Get image features from transformed images (may be empty if no features are computed)
                iter_feat_list += transform_images(img_obj=img_obj, roi_list=roi_list, settings=curr_setting,
                                                   compute_features=self.compute_features, extract_images=self.extract_images,
                                                   file_path=self.write_path)

            ########################################################################################################
            # Collect and combine features for current iteration
            ########################################################################################################

            if self.compute_features:
                feat_list.append(self.collect_features(img_obj=img_obj, roi_list=roi_list, feat_list=iter_feat_list, settings=curr_setting))

            # Clean up
            del img_obj, roi_list

        ########################################################################################################
        # Feature aggregation over settings
        ########################################################################################################

        if self.compute_features:
            # Check if features were extracted
            if len(feat_list) == 0:
                logging.warning("No features were extracted from %s images for %s.", self.modality + "_" + self.data_str, self.subject)
                return None

            # Concatenate feat list
            df_feat = pd.concat(feat_list, axis=0)

            # Write to file
            file_name = self.subject + "_" + self.modality + "_" + self.data_str + "_" + self.date + "_" + self.settings.general.config_str + "_features.csv"
            df_feat.to_csv(path_or_buf=os.path.join(self.write_path, file_name), sep=";", na_rep="NA", index=False, decimal=".")

            # Write successful completion to console or log
            logging.info("Features were extracted from %s images for %s.", self.modality + "_" + self.data_str + "_" + self.settings.general.config_str, self.subject)

    def process_deep_learning(self, output_slices=False):

        import logging
        from mirp.imageRead import load_image
        from mirp.imageProcess import estimate_image_noise, interpolate_image, interpolate_roi, crop_image_to_size, saturate_image, normalise_image
        from mirp.imagePerturbations import rotate_image, adapt_roi_size, randomise_roi_contours
        import copy

        from mirp.imagePlot import plot_image

        # Configure logger
        logging.basicConfig(format="%(levelname)s\t: %(processName)s \t %(asctime)s \t %(message)s", level=logging.INFO)

        # Notifications
        logging.info("Initialising image and mask processing using %s images for %s.", self.modality + "_" + self.data_str + "_", self.subject)

        # Get iterables from current settings which lead to different image adaptations
        iter_set, n_outer_iter, n_inner_iter = self.get_iterable_parameters(settings=self.settings)

        # Load image and roi
        if self.keep_images_in_memory:
            base_img_obj, base_roi_list = load_image(image_folder=self.image_folder, roi_folder=self.roi_folder, roi_reg_img_folder=self.roi_reg_img_folder,
                                                     settings=self.settings, modality=self.modality, roi_names=self.roi_names)
            self.set_image_name(img_obj=base_img_obj)
        else:
            base_img_obj = base_roi_list = None

        # Create lists for image objects and rois
        processed_image_list = []

        # Iterate over iterable settings
        for ii in np.arange(0, n_outer_iter):

            # Log current iteration
            if n_outer_iter * n_inner_iter > 1:
                if n_inner_iter > 1:
                    logging.info("Processing image and mask for %s to %s of %s adaptations.", str(ii * n_inner_iter + 1), str((ii + 1) * n_inner_iter),
                                 str(n_outer_iter * n_inner_iter))
                else:
                    logging.info("Processing image and mask for %s of %s adaptations.", str(ii + 1), str(n_outer_iter))
            else:
                logging.info("Starting image and mask processing.")

            ########################################################################################################
            # Load and pre-process image and roi
            ########################################################################################################

            # Set verbosity for loading roi - only report issues on first image adaptation
            if ii == 0:
                verbosity_roi = True
            else:
                verbosity_roi = False

            # Use pre-loaded base image and roi_list (more memory used, but may be faster if loading over a network), or load from disk.
            if self.keep_images_in_memory:
                img_obj = base_img_obj.copy()
                roi_list = copy.deepcopy(base_roi_list)
            else:
                # Read image and ROI segmentations
                img_obj, roi_list = load_image(image_folder=self.image_folder, roi_folder=self.roi_folder, roi_reg_img_folder=self.roi_reg_img_folder,
                                               settings=self.settings, modality=self.modality, roi_names=self.roi_names)
                self.set_image_name(img_obj=img_obj)

            # Check integrity of the list of regions of interest
            # roi_integrity_flag, roi_list = self.checkRoiListIntegrity(roi_list=roi_list, verbose=verbosity_roi)
            # if not roi_integrity_flag: continue

            ########################################################################################################
            # Update settings and initialise
            ########################################################################################################

            # Copy settings for current iteration run - this allows local changes to curr_setting
            curr_setting = copy.deepcopy(self.settings)

            # Update settings object with iterable settings
            curr_setting.vol_adapt.rot_angles = [iter_set.rot_angle[ii]]
            curr_setting.img_interpolate.new_spacing = [iter_set.vox_spacing[ii]]
            curr_setting.vol_adapt.translate_x = [iter_set.translate_x[ii]]
            curr_setting.vol_adapt.translate_y = [iter_set.translate_y[ii]]
            curr_setting.vol_adapt.translate_z = [iter_set.translate_z[ii]]

            ########################################################################################################
            # Determine image noise levels (optional)
            ########################################################################################################

            # Initialise noise level with place holder value
            est_noise_level = -1.0

            # Determine image noise levels
            if curr_setting.vol_adapt.add_noise and curr_setting.vol_adapt.noise_level is None and est_noise_level == -1.0:
                est_noise_level = estimate_image_noise(img_obj=img_obj, settings=curr_setting, method="chang")
            elif curr_setting.vol_adapt.add_noise:
                est_noise_level = curr_setting.vol_adapt.noise_level

            ########################################################################################################
            # Base image-based operations - basic operations on base image (rotation, cropping, noise addition)
            # Note interpolation and translation are performed simultaneously, and interpolation is only done after
            # application of spatial filters
            ########################################################################################################

            # Rotate object
            img_obj, roi_list = rotate_image(img_obj=img_obj, roi_list=roi_list, settings=curr_setting)

            # Add random noise to an image
            if curr_setting.vol_adapt.add_noise:
                img_obj.add_noise(noise_level=est_noise_level, noise_iter=ii)

            ########################################################################################################
            # Interpolation of base image
            ########################################################################################################

            # Translate and interpolate image to isometric voxels
            img_obj = interpolate_image(img_obj=img_obj, settings=curr_setting)
            roi_list = interpolate_roi(roi_list=roi_list, img_obj=img_obj, settings=curr_setting)

            ########################################################################################################
            # ROI-based operations
            # These operations only affect the regions of interest
            ########################################################################################################

            # Adapt roi sizes by dilation and erosion
            roi_list = adapt_roi_size(roi_list=roi_list, settings=curr_setting)

            # Update roi using SLIC
            roi_list = randomise_roi_contours(roi_list=roi_list, img_obj=img_obj, settings=curr_setting)

            ########################################################################################################
            # Standardise output
            ########################################################################################################

            # Crop image
            img_obj, roi_list = crop_image_to_size(img_obj=img_obj, crop_size=curr_setting.deep_learning.expected_size, roi_list=roi_list)

            # Set intensity range
            img_obj = saturate_image(img_obj=img_obj, intensity_range=curr_setting.deep_learning.intensity_range, fill_value=None)

            # Normalise the image to a standard range
            img_obj = normalise_image(img_obj=img_obj, norm_method=curr_setting.deep_learning.normalisation, intensity_range=curr_setting.deep_learning.intensity_range)

            ########################################################################################################
            # Collect output
            ########################################################################################################

            if self.extract_images:
                img_obj.export(file_path=self.write_path)
                for roi_obj in roi_list:
                    roi_obj.export(img_obj=img_obj, file_path=self.write_path)

            # Store processed imaging
            if output_slices:
                # 2D slices
                slice_img_obj_list = img_obj.get_slices()
                for jj in np.arange(len(slice_img_obj_list)):
                    for roi_obj in roi_list:
                        processed_image_list += [{"image": slice_img_obj_list[jj], "mask": roi_obj.get_slices(slice_number=jj)}]
            else:
                # 3D volumes
                for roi_obj in roi_list:
                    processed_image_list += [{"image": img_obj, "mask": roi_obj}]

            # Plot images
            if self.plot_images:
                plot_image(img_obj=img_obj, roi_list=roi_list, slice_id="all", file_path=self.write_path, file_name="plot", g_range=[np.nan, np.nan])

        # Return list of processed images and masks
        return processed_image_list

    def collect_features(self, img_obj, roi_list, feat_list, settings):
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
                                    "img_data_config": self.data_str,
                                    "img_data_noise_level": img_obj.noise,
                                    "img_data_noise_iter": img_obj.noise_iter,
                                    "img_data_rotation_angle": settings.vol_adapt.rot_angles[0],
                                    "img_data_roi_randomise_iter": [roi_obj.svx_randomisation_id for roi_obj in roi_list],
                                    "img_data_roi_adapt_size": [roi_obj.adapt_size for roi_obj in roi_list],
                                    "img_data_translate_x": settings.vol_adapt.translate_x[0],
                                    "img_data_translate_y": settings.vol_adapt.translate_y[0],
                                    "img_data_translate_z": settings.vol_adapt.translate_z[0],
                                    "img_data_voxel_size": voxel_size,
                                    "img_data_roi": [roi_obj.name for roi_obj in roi_list]},
                                   index=np.arange(len(roi_list)))

        feat_list.insert(0, df_img_data)  # Add to front of list

        # Add diagnostic features (optional)
        if self.provide_diagnostics:
            # Parse diagnostic data attached to the different rois to one single list
            df_diag_data = self.compileDiagnosticData(roi_list=roi_list)

            # Add to second position in the list
            feat_list.insert(1, df_diag_data)

        ########################################################################################################
        # Concatenation of results for the current image adaptation
        ########################################################################################################

        # Concatenate features - first reset indices for correct alignment
        df_iter_feat = pd.concat([df.reset_index(drop=True) for df in feat_list], axis=1)

        return df_iter_feat

    def checkRoiListIntegrity(self, roi_list, verbose=False):
        # Provides a number of tests to check the integrity of the selected regions of interest
        # TODO: improve this function and re-enable it. It should be able to parse combined rois, e.g. {roi_1 & roi_2 & roi_3}
        import logging

        # Log error on completely missing rois
        if len(roi_list) == 0:
            if verbose:
                logging.error("No regions of interest were found within %s images for %s.", self.modality + "_" + self.data_str, self.subject)

            return False, None

        # Check found roi names
        found_roi_names = np.asarray([roi_obj.name for roi_obj in roi_list])
        req_roi_names   = np.asarray(self.roi_names)

        # Check roi names that were not found
        if verbose and not np.all(np.in1d(req_roi_names, found_roi_names)):
            logging.warning("Some regions of interest were not found within %s images for %s: %s", self.modality + "_" + self.data_str,
                            self.subject, ", ".join(req_roi_names[~np.in1d(req_roi_names, found_roi_names)]))

        # Check duplicate roi names - maintain only unique rois
        uniq_roi_names, uniq_index, uniq_counts  = np.unique(np.asarray(found_roi_names), return_index=True, return_counts=True)
        if np.size(uniq_index) != len(found_roi_names):
            if verbose:
                logging.warning("Some duplicate regions of interest were found within %s images for %s: %s",
                                self.modality + "_" + self.data_str, self.subject,
                                ", ".join(uniq_roi_names[uniq_counts > 1]))
                logging.info("Only first non-unique regions of interest are used.")

            roi_list = [roi_list[ii] for ii in uniq_index]

        return True, roi_list

    def extractDiagnosticFeatures(self, img_obj, roi_list=None, append_str=""):
        """ Extracts diagnostics features from image objects and lists of roi objects """

        if self.provide_diagnostics and roi_list is not None:

            # The image diagnostic features are calculated only once, yet are written on every row by concatenation
            # roi diagnostic features
            img_diag_feat = img_obj.compute_diagnostic_features(append_str="_" + append_str + "_img_" + self.modality)

            for curr_roi in roi_list:
                # Add image diagnostic data to roi
                curr_roi.diagnostic_list += [img_diag_feat]

                # Add roi diagnostic data to roi
                curr_roi.compute_diagnostic_features(img_obj=img_obj, append_str="_" + append_str + "_roi")

    def compileDiagnosticData(self, roi_list):
        """ Compiles diagnostic data from regions of interest """

        if self.provide_diagnostics:
            # Instantiate feature list
            diag_feat_list = []

            # Iterate over rois and combine lists of diagnostic tables into single frame, representing a row
            for curr_roi in roi_list:
                diag_feat_list += [pd.concat(curr_roi.diagnostic_list, axis=1)]

            # Combine rows into single table
            df_diag_feat = pd.concat(diag_feat_list, axis=0)

            return df_diag_feat

    def set_image_name(self, img_obj):
        """
        Sets the name of image based on parameters of the corresponding data object
        :param img_obj:
        :return:
        """
        import copy

        # Add cohort and subject or just subject
        if self.cohort != "NA":
            name_str = self.cohort + "_" + self.subject
        else:
            name_str = copy.deepcopy(self.subject)

        # Add modality
        name_str += "_" + self.modality

        # Add data string
        if self.data_str != "":
            name_str += "_" + self.data_str

        # Add configuration strin
        if self.settings.general.config_str is not None:
            name_str += "_" + self.settings.general.config_str

        # Set name
        img_obj.name = name_str
