import copy
import os
import warnings
from typing import Any
from xml.etree import ElementTree as ElemTree

from mirp.settings.utilities import str2list, str2type, read_node


def import_data_settings(
    path: str,
    is_mask: bool
) -> list[dict[str, Any]]:
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
        if is_mask:
            mask = os.path.normpath(
                str2type(read_node(paths_branch, ["project_folder", "mask"]), "path"))
            data_arguments += [("mask", mask)]
        else:
            image = os.path.normpath(
                str2type(read_node(paths_branch, ["project_folder", "image"]), "path"))
            data_arguments += [("image", image)]

        # Set sample name.
        sample_name = str2list(read_node(paths_branch, ["sample_name", "subject_include"]), "str")
        data_arguments += [("sample_name", sample_name)]

        # Deprecated items.
        if paths_branch.find("subject_exclude") is not None:
            # noinspection PyDeprecation
            _import_data_settings_deprecation_warning("subject_exclude")
        if paths_branch.find("write_folder") is not None:
            # noinspection PyDeprecation
            _import_data_settings_deprecation_warning("write_path")
        if paths_branch.find("provide_diagnostics") is not None:
            # noinspection PyDeprecation
            _import_data_settings_deprecation_warning("provide_diagnostics")
        if paths_branch.find("cohort") is not None:
            # noinspection PyDeprecation
            _import_data_settings_deprecation_warning("cohort")
       
        # Iterate over data branches
        for data_branch in branch.findall("data"):
            current_data_arguments = copy.deepcopy(data_arguments)

            if is_mask:
                mask_name = str2list(data_branch.find("mask_name"), str)
                current_data_arguments += [("mask_name", mask_name)]

                mask_file_type = str2type(data_branch.find("mask_file_type"), "str")
                current_data_arguments += [("mask_file_type", mask_file_type)]

                mask_sub_folder = str2type(read_node(data_branch, ["mask_sub_folder", "roi_folder"]), "path")
                current_data_arguments += [("mask_sub_folder", mask_sub_folder)]

                mask_modality = str2list(data_branch.find("mask_modality"), "str")
                current_data_arguments += [("mask_modality", mask_modality)]

                roi_name = str2list(data_branch.find("roi_names"), "str")
                current_data_arguments += [("roi_name", roi_name)]

            else:
                image_name = str2list(read_node(data_branch, ["image_name", "image_filename_pattern"]), "str")
                current_data_arguments += [("image_name", image_name)]

                image_file_type = str2type(data_branch.find("image_file_type"), "str")
                current_data_arguments = [("image_file_type", image_file_type)]

                image_sub_folder = str2type(read_node(data_branch, ["image_sub_folder", "image_folder"]), "path")
                current_data_arguments += [("image_sub_folder", image_sub_folder)]

                image_modality = str2type(read_node(data_branch, ["image_modality", "modality"]), "str")
                current_data_arguments += [("image_modality", image_modality)]

            # More deprecated items.
            if data_branch.find("registration_image_folder") is not None:
                # noinspection PyDeprecation
                _import_data_settings_deprecation_warning("registration_image_folder")
            if data_branch.find("registration_image_filename_pattern") is not None:
                # noinspection PyDeprecation
                _import_data_settings_deprecation_warning("registration_image_filename_pattern")
            if data_branch.find("roi_list_path") is not None:
                # noinspection PyDeprecation
                _import_data_settings_deprecation_warning("roi_list_path")
            if data_branch.find("divide_disconnected_roi") is not None:
                # noinspection PyDeprecation
                _import_data_settings_deprecation_warning("divide_disconnected_roi")
            if data_branch.find("extraction_config") is not None:
                # noinspection PyDeprecation
                _import_data_settings_deprecation_warning("extraction_config")

            data_arguments_list += [current_data_arguments]

    return data_arguments_list


def _import_data_settings_deprecation_warning(tag: str) -> None:
    warnings.warn(
        f"The {tag} tag was deprecated in version 2.0, and will not be parsed.",
        DeprecationWarning
    )
