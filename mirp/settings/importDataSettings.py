import copy
import os

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
       
        # Iterate over data branches
        for data_branch in branch.findall("data"):
            current_data_arguments = copy.deepcopy(data_arguments)

            if is_mask:
                mask_sub_folder = str2type(data_branch.find("roi_folder"), "path")
                current_data_arguments += [("mask_sub_folder", mask_sub_folder)]

                roi_name = str2list(data_branch.find("roi_names"), "str")
                current_data_arguments += [("roi_name", roi_name)]

            else:
                image_modality = str2type(data_branch.find("modality"), "str")
                current_data_arguments += [("image_modality", image_modality)]

                image_sub_folder = str2type(data_branch.find("image_folder"), "path")
                current_data_arguments += [("image_sub_folder", image_sub_folder)]

                image_name = str2type(data_branch.find("image_filename_pattern"), "str")
                current_data_arguments += [("image_name", image_name)]

            # More deprecated stuff.
            # TODO: add deprecation warnings.
            # roi_reg_img_folder = str2type(data_branch.find("registration_image_folder"), "str")
            # registration_image_file_name_pattern = str2type(
            #     data_branch.find("registration_image_filename_pattern"), "str")
            # roi_list_path: str = str2type(data_branch.find("roi_list_path"), "str")
            # divide_disconnected_roi = str2type(data_branch.find("divide_disconnected_roi"), "str", "combine")
            # extraction_config = str2list(data_branch.find("extraction_config"), "str")

            data_arguments_list += [current_data_arguments]

    return data_arguments_list
