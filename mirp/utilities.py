import os
from collections import Counter
from itertools import product

import pandas as pd


def index_to_world(index, origin, spacing):
    """"Translates index to world coordinates"""
    return origin + index * spacing


def world_to_index(coord, origin, spacing):
    """"Translates world coordinates to index"""
    return (coord - origin) / spacing


def extract_roi_names(roi_list):
    """
    Extract the names of the regions of interest in roi_list
    :param roi_list: List of roi objects
    :return: List of roi names
    """
    return [roi.name for roi in roi_list]


def parse_roi_name(roi: str):

    # Determine if the input roi is surrounded by curly brackets
    if roi.startswith("{") and roi.endswith("}"):
        # Strip curly brackets
        roi = roi.strip("{}")

        # Get individual rois. & and | are valid splitting variables.
        if "&" in roi:
            indiv_roi = roi.split("&")

        elif "|" in roi:
            indiv_roi = roi.split("|")

        else:
            indiv_roi = [roi]

        # Remove white space around individual rois
        indiv_roi = [curr_roi.strip() for curr_roi in indiv_roi]

    else:
        indiv_roi = [roi.strip()]

    return indiv_roi


def expand_grid(data_dict):
    rows = product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def check_string(input_string):
    """Updates string characters that may lead to errors when written."""
    input_string = input_string.replace(" ", "_")
    input_string = input_string.replace(",", "_")
    input_string = input_string.replace(";", "_")
    input_string = input_string.replace(":", "_")
    input_string = input_string.replace("\"", "_")
    input_string = input_string.replace("=", "_equal_")
    input_string = input_string.replace(">", "_greater_")
    input_string = input_string.replace("<", "_smaller_")
    input_string = input_string.replace("&", "_and_")
    input_string = input_string.replace("|", "_or_")

    return input_string


def get_valid_elements(input_list):

    filt = [elem_ii is not None for elem_ii in input_list]

    return [elem_ii for (elem_ii, flag) in zip(input_list, filt) if flag]


def get_most_common_element(input_list):

    if len(input_list) == 0:
        return None

    counts = Counter(input_list)

    return counts.most_common(n=1)[0][0]


def get_version():
    with open(os.path.join("..", 'VERSION.txt')) as version_file:
        version = version_file.read().strip()

    return version


def get_spherical_structure(radius, spacing):
    import numpy as np

    # Define extent in index coordinates
    extent = np.ceil(np.divide(radius, spacing)).astype(int)

    # Generate coordinate grids. Note that this will generate a grid with 0,0,0 as center voxel.
    grid_z, grid_y, grid_x = np.mgrid[-extent[0]:extent[0]+1, -extent[1]:extent[1]+1, -extent[2]:extent[2]+1]

    # Transform coordinates back to real world space. We then square the values to compute the Euclidean norm later on.
    grid_z = np.power(grid_z * spacing[0], 2.0)
    grid_y = np.power(grid_y * spacing[1], 2.0)
    grid_x = np.power(grid_x * spacing[2], 2.0)

    # Compute the Euclidean distance to each voxel center and filter by radius.
    geom_struct = np.sqrt(grid_z + grid_y + grid_x) <= radius

    return geom_struct


def makedirs_check(path):
    """
    Checks if the given path is an existing directory
    structure, otherwise creates it.
    """

    if not os.path.isdir(path):
        os.makedirs(path)
