import logging

import numpy as np

from mirp.imageProcess import crop_image, get_supervoxels, get_supervoxel_overlap
from mirp.utilities import extract_roi_names
from mirp.importSettings import SettingsClass
from mirp.imageClass import ImageClass
from mirp.roiClass import RoiClass
from typing import Union, List


def randomise_roi_contours(roi_list, img_obj, settings: SettingsClass):
    """Use SLIC to randomise the roi based on supervoxels"""

    # Check whether randomisation should take place
    if not settings.perturbation.randomise_roi:
        return roi_list

    from scipy.ndimage import binary_closing

    new_roi_list = []

    # Iterate over roi objects
    for roi_ind in np.arange(0, len(roi_list)):

        # Resect image to speed up segmentation process
        res_img_obj, res_roi_obj = crop_image(img_obj=img_obj, roi_obj=roi_list[roi_ind], boundary=25.0, z_only=False)

        # Check if the roi is empty. If so, add the number of required empty rois
        if res_roi_obj.is_empty():
            for ii in np.arange(settings.perturbation.roi_random_rep):
                repl_roi = roi_list[roi_ind].copy()
                repl_roi.name += "_svx_" + str(ii)  # Adapt roi name
                repl_roi.svx_randomisation_id = ii + 1  # Update randomisation id
                new_roi_list.append(repl_roi)

            # Go on to the next roi in the roi list
            continue

        # Get supervoxels
        img_segments = get_supervoxels(img_obj=res_img_obj, roi_obj=res_roi_obj, settings=settings)

        # Determine overlap of supervoxels with contour
        overlap_indices, overlap_fract, overlap_size = get_supervoxel_overlap(roi_obj=res_roi_obj, img_segments=img_segments)

        # Set the highest overlap to 1.0 to ensure selection of at least 1 supervoxel
        overlap_fract[np.argmax(overlap_fract)] = 1.0

        # Include supervoxels with 90% coverage and exclude those with less then 20% coverage
        overlap_fract[overlap_fract >= 0.90] = 1.0
        overlap_fract[overlap_fract < 0.20] = 0.0

        # Determine grid indices of the resected grid with respect to the original image grid
        grid_origin = img_obj.to_voxel_coordinates(x=res_img_obj.origin)
        grid_origin = grid_origin.astype(int)

        # Iteratively create randomised regions of interest
        for ii in np.arange(settings.perturbation.roi_random_rep):

            # Draw random numbers between 0.0 and 1.0
            random_incl = np.random.random(size=len(overlap_fract))

            # Select those segments where the random number is less than the overlap fraction - i.e. the fraction is the
            # probability of selecting the supervoxel
            incl_segments = overlap_indices[np.less(random_incl, overlap_fract)]

            # Replace randomised contour in original roi voxel space
            roi_vox = np.zeros(shape=roi_list[roi_ind].roi.size, dtype=bool)
            roi_vox[grid_origin[0]: grid_origin[0] + res_roi_obj.roi.size[0],
                    grid_origin[1]: grid_origin[1] + res_roi_obj.roi.size[1],
                    grid_origin[2]: grid_origin[2] + res_roi_obj.roi.size[2], ] = \
                np.reshape(np.in1d(np.ravel(img_segments), incl_segments),  res_roi_obj.roi.size)

            # Apply binary closing to close gaps
            roi_vox = binary_closing(input=roi_vox)

            # Update voxels in original roi, adapt name and set randomisation id
            repl_roi = roi_list[roi_ind].copy()
            repl_roi.roi.set_voxel_grid(voxel_grid=roi_vox)  # Replace copied original contour with randomised contour
            repl_roi.name += "_svx_" + str(ii)             # Adapt roi name
            repl_roi.svx_randomisation_id = ii + 1         # Update randomisation id
            new_roi_list += [repl_roi]

    return new_roi_list


def adapt_roi_size(roi_list, settings: SettingsClass):
    """ Adapt roi size by growing or shrinking the roi """

    # Adapt roi size by shrinking or increasing the roi
    new_roi_list = []

    # Get the adaptation size and type. Rois with adapt_size > 0.0 are dilated. Rois with adapt_size < 0.0 are eroded.
    # The type determines whether the roi is grown/shrunk with by certain distance ("distance") or to a certain volume fraction ("fraction")
    adapt_size_list = settings.perturbation.roi_adapt_size
    adapt_type      = settings.perturbation.roi_adapt_type

    # Iterate over roi objects in the roi list and adaptation sizes
    for roi_obj in roi_list:
        for adapt_size in adapt_size_list:
            if adapt_size > 0.0 and adapt_type == "distance":
                new_roi_obj  = roi_obj.copy()
                new_roi_obj.dilate(by_slice=settings.general.by_slice, dist=adapt_size)

                # Update name and adaptation size
                new_roi_obj.name += "_grow" + str(adapt_size)
                new_roi_obj.adapt_size = adapt_size

                # Add to roi list
                new_roi_list += [new_roi_obj]

            elif adapt_size < 0.0 and adapt_type == "distance":
                new_roi_obj = roi_obj.copy()
                new_roi_obj.erode(by_slice=settings.general.by_slice, dist=adapt_size, eroded_vol_fract=settings.perturbation.max_volume_erosion)

                # Update name and adaptation size
                new_roi_obj.name += "_shrink" + str(np.abs(adapt_size))
                new_roi_obj.adapt_size = adapt_size

                # Add to roi list
                new_roi_list += [new_roi_obj]

            elif adapt_type == "fraction" and not adapt_size == 0.0:
                new_roi_obj = roi_obj.copy()
                new_roi_obj.adapt_volume(by_slice=settings.general.by_slice, vol_grow_fract=adapt_size)

                # Update name and adaptation size
                if adapt_size > 0:
                    new_roi_obj.name += "_grow" + str(adapt_size)
                else:
                    new_roi_obj.name += "_shrink" + str(np.abs(adapt_size))
                new_roi_obj.adapt_size = adapt_size

                # Add to roi list
                new_roi_list += [new_roi_obj]

            else:
                new_roi_list += [roi_obj]

    # Check for non-updated rois
    roi_names = extract_roi_names(new_roi_list)
    uniq_roi_names, uniq_index, uniq_counts = np.unique(np.asarray(roi_names), return_index=True, return_counts=True)
    if np.size(uniq_index) != len(roi_names):
        uniq_roi_list = [new_roi_list[ii] for ii in uniq_index]
    else:
        uniq_roi_list = new_roi_list

    # Return expanded roi list
    return uniq_roi_list
