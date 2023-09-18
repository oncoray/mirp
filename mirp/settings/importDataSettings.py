import os
import numpy as np
import pandas as pd
from xml.etree import ElementTree as ElemTree

from mirp.settings.utilities import str2list, str2type


def import_data_settings(
    path: str,
    is_mask: bool
):
    if os.path.isfile(path):
        raise ValueError(f"The {path} data settings file does not exist. Please check spelling of the file path.")
    if not path.endswith(".xml"):
        raise ValueError(f"The {path} data settings file is not an xml file.")

    # Load xml
    tree = ElemTree.parse(path)
    root = tree.getroot()

    # Empty list for iteratively storing data objects
    data_arguments_list = []

    # Iterate over configurations
    for branch in root.findall("config"):
        # Set data arguments.
        data_arguments = []

        # Read data from xml file
        paths_branch = branch.find("paths")

        # Set main directory.
        project_path = os.path.normpath(str2type(paths_branch.find("project_folder"), "path"))
        if is_mask:
            data_arguments += [("mask", project_path)]
        else:
            data_arguments += [("image", project_path)]

        # Set sample name.
        sample_name = str2list(paths_branch.find("subject_include"), "str")
        data_arguments += [("sample_name", sample_name)]

        # Deprecated stuff.
        # TODO: Add deprecation warnings.
        # excl_subj = str2list(paths_branch.find("subject_exclude"), "str")
        # write_path = os.path.normpath(str2type(paths_branch.find("write_folder"), "path"))
        # provide_diagnostics = str2type(paths_branch.find("provide_diagnostics"), "bool")
        # cohort_id = str2type(paths_branch.find("cohort"), "str", "NA")
       
        # Check if there are multiple data branches with the same modality, because then we need to update the subject.
        image_data_identifier_list = []
        for data_branch in branch.findall("data"):
            # Collect modality, image_folder and image_file_name_pattern
            image_data_identifier_list += [pd.DataFrame.from_dict(dict({
                "modality": [str2type(data_branch.find("modality"), "str")],
                "folder": [str2type(data_branch.find("image_folder"), "path")],
                "image_file_name_pattern": [str2type(data_branch.find("image_filename_pattern"), "str")]
            }))]

        # Concatenate to single data frame.
        image_data_identifier_list = pd.concat(image_data_identifier_list, ignore_index=True)

        # Populate image data identifiers aside from subject/
        n_unique_sets = image_data_identifier_list.shape[0]
        if image_data_identifier_list.drop_duplicates(
                subset=["modality"],
                inplace=False).shape[0] == n_unique_sets:
            image_data_identifiers = ["modality"]

        elif image_data_identifier_list.drop_duplicates(
                subset=["modality", "folder"],
                inplace=False).shape[0] == n_unique_sets:
            image_data_identifiers = ["modality", "folder"]

        elif image_data_identifier_list.drop_duplicates(
                subset=["modality", "image_file_name_pattern"],
                inplace=False).shape[0] == n_unique_sets:
            image_data_identifiers = ["modality", "file_name"]

        else:
            image_data_identifiers = ["modality", "folder", "file_name"]

        # Iterate over data branches
        for data_branch in branch.findall("data"):

            # Read current data branch
            image_modality = str2type(data_branch.find("modality"), "str")
            image_folder = str2type(data_branch.find("image_folder"), "path")
            roi_folder = str2type(data_branch.find("roi_folder"), "path")
            roi_reg_img_folder = str2type(data_branch.find("registration_image_folder"), "str")
            image_file_name_pattern = str2type(data_branch.find("image_filename_pattern"), "str")
            registration_image_file_name_pattern = str2type(
                data_branch.find("registration_image_filename_pattern"), "str")
            roi_names = str2list(data_branch.find("roi_names"), "str")
            roi_list_path: str = str2type(data_branch.find("roi_list_path"), "str")
            divide_disconnected_roi = str2type(data_branch.find("divide_disconnected_roi"), "str", "combine")
            extraction_config = str2list(data_branch.find("extraction_config"), "str")

            # Check if extraction config has been set -- this allows setting configurations for mixed modalities
            if extraction_config is not None and config_settings is not None:
                new_config_settings = []

                # Iterate over configuration names mentioned in the data and compare those to the configuration strings
                # in the settings. If a match is found, the configuration is set to the new configuration list.
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
                    logging.warning(
                        "No image folder was set. If images are located directly in the patient folder, use subject_dir as tag")
                    continue

                # Set correct paths in case folders are tagged with subject_dir. This tag indicates that the data is
                # directly in the subject folder.
                if image_folder == "subject_dir":
                    image_folder = ""
                if roi_folder == "subject_dir":
                    roi_folder = ""

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
                    raise ValueError(
                        "Column names in the provided roi list do not match \"sample_id\" or \"roi_name\".")
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
                image_dir_subj = os.path.normpath(os.path.join(project_path, curr_subj, image_folder))
                roi_dir_subj = os.path.normpath(os.path.join(project_path, curr_subj, roi_folder))
                roi_reg_img_subj = os.path.normpath(os.path.join(project_path, curr_subj, roi_reg_img_folder))

                # Check if image and roi folders exist on the path
                if not os.path.isdir(image_dir_subj):
                    logging.info("%s was excluded as the image folder %s was not found.", curr_subj,
                                 os.path.join(image_dir_subj))
                    continue

                if not os.path.isdir(roi_dir_subj):
                    logging.info("%s was excluded as the roi folder %s was not found.", curr_subj,
                                 os.path.join(roi_dir_subj))
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

                    # Identify data string.
                    data_string = []
                    for data_identifier in image_data_identifiers:
                        if data_identifier == "modality":
                            data_string += [image_modality]

                        elif data_identifier == "folder":
                            if image_folder is not None:
                                image_folder_str = image_folder.replace("/", "_").replace("\\", "_")
                                data_string += [image_folder_str]

                        elif data_identifier == "file_name":
                            if image_file_name_pattern is not None:
                                data_string += [image_file_name_pattern]
                        else:
                            raise ValueError(f"Encountered an unexpected data_identifier: {data_identifier}")

                    data_obj = ExperimentClass(
                        modality=image_modality,
                        subject=curr_subj,
                        cohort=cohort_id,
                        write_path=write_path,
                        image_folder=image_dir_subj,
                        roi_folder=roi_dir_subj,
                        roi_reg_img_folder=roi_reg_img_subj,
                        image_file_name_pattern=image_file_name_pattern,
                        registration_image_file_name_pattern=registration_image_file_name_pattern,
                        roi_names=roi_names,
                        data_str=data_string,
                        provide_diagnostics=provide_diagnostics,
                        settings=curr_config_setting,
                        compute_features=compute_features,
                        extract_images=extract_images,
                        plot_images=plot_images,
                        keep_images_in_memory=keep_images_in_memory)

                    data_obj_list.append(data_obj)

    return data_obj_list
