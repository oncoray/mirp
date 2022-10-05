import os
import warnings
from typing import Union

import numpy as np
import itk

from mirp.imageClass import ImageClass
from mirp.roiClass import RoiClass, merge_roi_objects
from mirp.utilities import parse_roi_name


class NoITKSegmentationFileFound(Exception):
    """Error generated when no unique files are found"""


def read_itk_image(image_folder, modality=None, name_contains=None):

    # Identify the ITK (NIfTI or NRRD) file.
    itk_file = _find_itk_images(
        image_folder=image_folder,
        name_contains=name_contains,
        is_mask=False
    )

    # Load the image
    itk_img = itk.imread(os.path.join(image_folder, itk_file))

    # Import the image volume
    voxel_grid = itk.GetArrayFromImage(itk_img).astype(float)

    # Determine origin, spacing, and orientation
    image_origin = np.array(itk_img.GetOrigin())[::-1]
    image_spacing = np.array(itk_img.GetSpacing())[::-1]
    image_orientation = np.reshape(np.ravel(itk.array_from_matrix(itk_img.GetDirection()))[::-1], [3, 3])

    # Create an ImageClass object from the input image.
    image_obj = ImageClass(
        voxel_grid=voxel_grid,
        origin=image_origin,
        spacing=image_spacing,
        orientation=image_orientation,
        modality=modality,
        spat_transform="base",
        no_image=False
    )

    return image_obj


def read_itk_segmentations(image_folder, roi):

    roi_list = []

    # Iterate over segmentation names.
    for roi_name in roi:
        roi_obj: Union[RoiClass, None] = _load_itk_segmentation(image_folder=image_folder, roi=roi_name)

        # Add ROI object to list if it is not None.
        if roi_obj is not None:
            roi_list += [roi_obj]

    return roi_list


def _find_itk_images(image_folder, name_contains=None, is_mask=False):

    # Check folder contents, keep only files that are recognised as DICOM images.
    file_list = os.listdir(image_folder)
    file_list = [file_name for file_name in file_list if not os.path.isdir(os.path.join(image_folder, file_name))]
    file_list = [file_name for file_name in file_list if file_name.lower().endswith((".nii", ".nii.gz", ".nrrd"))]

    if len(file_list) == 0:
        raise FileNotFoundError(f"The folder ({image_folder}) does not contain NIfTI or NRRD files.")

    if name_contains is None and len(file_list) > 1:
        raise ValueError(f"The folder ({image_folder}) contains multiple valid files: {file_list}. Please assign the intended file to a separate folder or provide a string for"
                         f"(partial) matching with the filename.")

    elif len(file_list) > 1:
        # Find files that (partially) match the name_contains string
        file_list = [file_name for file_name in file_list if name_contains in file_name]

        if len(file_list) == 0:
            raise NoITKSegmentationFileFound(f"Could not match the filename pattern {name_contains} against the files in the folder ({image_folder}).")

        elif len(file_list) > 1:
            warnings.warn(f"Multiple (partial) matches of filename pattern {name_contains} were found in the folder ({image_folder}): {file_list}. We will"
                          f"use the match with the shortest filename.")

            # Determine file lengths and find out the smallest file name. This will be used as a match.
            file_name_lengths = [len(file_name) for file_name in file_list]
            min_length = min(file_name_lengths)

            # Load file list
            file_list = [file_name for ii, file_name in enumerate(file_list) if file_name_lengths[ii] == min_length]

            if len(file_list) > 1:
                raise ValueError(f"Could not use the filename pattern {name_contains} to select a single file in the folder ({image_folder}).")
        else:
            pass

    else:
        pass

    # Only a single file should remain.
    if is_mask:
        # Check if the image is actually a mask.

        # Load the file and convert to numpy.
        itk_img = itk.imread(os.path.join(image_folder, file_list[0]))
        img_volume = itk.GetArrayFromImage(itk_img)

        # Check if the image is a mask.
        if (np.min(img_volume) == 0 or np.min(img_volume) == 1) and np.max(img_volume) == 1:
            pass
        else:
            raise ValueError(f"The image file ({file_list[0]}) in the image folder ({image_folder}) is not a mask consisting of 0s and 1s.")

    return file_list[0]


def _load_itk_segmentation(image_folder, roi: str):

    # Deparse roi
    deparsed_roi = parse_roi_name(roi=roi)

    # Create an empty roi_list. This list will be iteratively expanded.
    roi_list = []

    for single_roi in deparsed_roi:
        # Not every roi may be found as a file, e.g. in a shotgun approach. We capture
        # NoITKSegmentationFileFound errors that only occur if the segmentation cannot be found.
        try:
            file_name = _find_itk_images(image_folder=image_folder, name_contains=single_roi, is_mask=True)
        except NoITKSegmentationFileFound:
            continue

        # Load the segmentation file
        itk_img = itk.imread(os.path.join(image_folder, file_name))

        # Obtain mask
        mask = itk.GetArrayFromImage(itk_img).astype(bool)

        # Determine origin, spacing, and orientation
        mask_origin = np.array(itk_img.GetOrigin())[::-1]
        mask_spacing = np.array(itk_img.GetSpacing())[::-1]
        mask_orientation = np.reshape(np.ravel(itk.array_from_matrix(itk_img.GetDirection()))[::-1], [3, 3])

        # Create an ImageClass object using the mask.
        roi_mask_obj = ImageClass(
            voxel_grid=mask,
            origin=mask_origin,
            spacing=mask_spacing,
            orientation=mask_orientation,
            modality="SEG",
            spat_transform="base",
            no_image=False
        )

        roi_list += [RoiClass(name=single_roi,
                              contour=None,
                              roi_mask=roi_mask_obj)]

    # Attempt to merge deparsed roi objects.
    if len(roi_list) > 0:
        roi_obj = merge_roi_objects(roi_list=roi_list)

    else:
        roi_obj = None

    return roi_obj
