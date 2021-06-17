import logging
from copy import deepcopy

from mirp.importSettings import SettingsClass
from mirp.imageClass import ImageClass

import numpy as np
import pandas as pd


def saturate_image(img_obj, intensity_range, fill_value):

    # Sature image
    img_obj.saturate(intensity_range=intensity_range, fill_value=fill_value)

    return img_obj


def normalise_image(img_obj, norm_method, intensity_range=None, saturation_range=None, mask=None):

    if intensity_range is None:
        intensity_range = [np.nan, np.nan]

    if saturation_range is None:
        saturation_range = [np.nan, np.nan]

    # Normalise intensities
    img_obj.normalise_intensities(norm_method=norm_method,
                                  intensity_range=intensity_range,
                                  saturation_range=saturation_range,
                                  mask=mask)

    return img_obj


def resegmentise(img_obj, roi_list, settings):
    # Resegmentises segmentation map based on selected method

    if roi_list is not None:

        for ii in np.arange(0, len(roi_list)):

            # Generate intensity and morphology masks
            roi_list[ii].generate_masks()

            # Skip if no resegmentation method is used
            if settings.roi_resegment.method is None: continue

            # Re-segment image
            roi_list[ii].resegmentise_mask(img_obj=img_obj, by_slice=settings.general.by_slice, method=settings.roi_resegment.method, settings=settings)

            # Set the roi as the union of the intensity and morphological maps
            roi_list[ii].update_roi()

    return roi_list


def interpolate_image(img_obj, settings):
    # Interpolates an image set to a new spacing
    img_obj.interpolate(by_slice=settings.general.by_slice, settings=settings)

    return img_obj


def interpolate_roi(roi_list, img_obj, settings):
    # Interpolates roi to a new spacing
    for ii in np.arange(0, len(roi_list)):
        roi_list[ii].interpolate(img_obj=img_obj, settings=settings)

    return roi_list


def estimate_image_noise(img_obj, settings, method="chang"):

    # TODO Implement as method for imageClass
    import scipy.ndimage as ndi

    # Skip if the image is missing
    if img_obj.is_missing:
        return -1.0

    if method == "rank":
        """ Estimate image noise level using the method by Rank, Lendl and Unbehauen, Estimation of image noise variance,
        IEEE Proc. Vis. Image Signal Process (1999) 146:80-84"""

        ################################################################################################################
        # Step 1: filter with a cascading difference filter to suppress original image volume
        ################################################################################################################

        diff_filter = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])

        # Filter voxel volume
        filt_vox = ndi.convolve1d(img_obj.get_voxel_grid(), weights=diff_filter, axis=1)
        filt_vox = ndi.convolve1d(filt_vox, weights=diff_filter, axis=2)

        del diff_filter

        ################################################################################################################
        # Step 2: compute histogram of local standard deviation and calculate histogram
        ################################################################################################################

        # Calculate local means
        local_means = ndi.uniform_filter(filt_vox, size=[1, 3, 3])

        # Calculate local sum of squares
        sum_filter = np.array([1.0, 1.0, 1.0])
        local_sum_square = ndi.convolve1d(np.power(filt_vox, 2.0), weights=sum_filter, axis=1)
        local_sum_square = ndi.convolve1d(local_sum_square, weights=sum_filter, axis=2)

        # Calculate local variance
        local_variance = 1.0 / 8.0 * (local_sum_square - 9.0 * np.power(local_means, 2.0))

        del local_means, filt_vox, local_sum_square, sum_filter

        ################################################################################################################
        # Step 3: calculate median noise - this differs from the original
        ################################################################################################################

        # Set local variances below 0 (due to floating point rounding) to 0
        local_variance = np.ravel(local_variance)
        local_variance[local_variance < 0.0] = 0.0

        # Select robust range (within IQR)
        local_variance = local_variance[np.logical_and(local_variance >= np.percentile(local_variance, 25),
                                                       local_variance <= np.percentile(local_variance, 75))]

        # Calculate Gaussian noise
        est_noise = np.sqrt(np.mean(local_variance))

        del local_variance

    elif method == "ikeda":
        """ Estimate image noise level using a method by Ikeda, Makino, Imai et al., A method for estimating noise variance of CT image,
                Comp Med Imaging Graph (2010) 34:642-650"""

        ################################################################################################################
        # Step 1: filter with a cascading difference filter to suppress original image volume
        ################################################################################################################

        diff_filter = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])

        # Filter voxel volume
        filt_vox = ndi.convolve1d(img_obj.get_voxel_grid(), weights=diff_filter, axis=1)
        filt_vox = ndi.convolve1d(filt_vox, weights=diff_filter, axis=2)

        ################################################################################################################
        # Step 2: calculate median noise
        ################################################################################################################

        est_noise = np.median(np.abs(filt_vox)) / 0.6754

        del filt_vox, diff_filter

    elif method == "chang":
        """ Noise estimation based on wavelets used in Chang, Yu and Vetterli, Adaptive wavelet thresholding for image
        denoising and compression. IEEE Trans Image Proc (2000) 9:1532-1546"""

        ################################################################################################################
        # Step 1: calculate HH subband of the wavelet transformation
        ################################################################################################################

        import pywt

        # Generate digital wavelet filter
        hi_filt = np.array(pywt.Wavelet("coif1").dec_hi)

        # Calculate HH subband image
        filt_vox = ndi.convolve1d(img_obj.get_voxel_grid(), weights=hi_filt, axis=1)
        filt_vox = ndi.convolve1d(filt_vox, weights=hi_filt, axis=2)

        ################################################################################################################
        # Step 2: calculate median noise
        ################################################################################################################

        est_noise = np.median(np.abs(filt_vox)) / 0.6754

        del filt_vox

    elif method == "immerkaer":
        """ Noise estimation based on laplacian filtering, described in Immerkaer, Fast noise variance estimation.
        Comput Vis Image Underst (1995) 64:300-302"""

        ################################################################################################################
        # Step 1: construct filter and filter voxel volume
        ################################################################################################################

        # Create filter
        noise_filt = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], ndmin=3)

        # Apply filter
        filt_vox = ndi.convolve(img_obj.get_voxel_grid(), weights=noise_filt)

        ################################################################################################################
        # Step 2: calculate noise level
        ################################################################################################################

        est_noise = np.sqrt(np.mean(np.power(filt_vox, 2.0))) / 36.0

        del filt_vox

    elif method == "zwanenburg":
        """ Noise estimation based on blob detection for weighting immerkaer filtering """

        ################################################################################################################
        # Step 1: construct laplacian filter and filter voxel volume
        ################################################################################################################

        # Create filter
        noise_filt = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], ndmin=3)

        # Apply filter
        filt_vox = ndi.convolve(img_obj.get_voxel_grid(), weights=noise_filt)
        filt_vox = np.power(filt_vox, 2.0)

        ################################################################################################################
        # Step 2: construct blob weighting
        ################################################################################################################

        # Spacing for gaussian
        gauss_filt_spacing = np.full(shape=(3), fill_value=np.min(img_obj.spacing))
        gauss_filt_spacing = np.divide(gauss_filt_spacing, img_obj.spacing)

        # Difference of gaussians
        weight_vox = ndi.gaussian_filter(img_obj.get_voxel_grid(), sigma=1.0 * gauss_filt_spacing) - ndi.gaussian_filter(img_obj.get_voxel_grid(), sigma=4.0 * gauss_filt_spacing)

        # Smooth edge detection
        weight_vox = ndi.gaussian_filter(np.abs(weight_vox), sigma=2.0*gauss_filt_spacing)

        # Convert to weighting scale
        weight_vox = 1.0 - weight_vox / np.max(weight_vox)

        # Decrease weight of vedge voxels
        weight_vox = np.power(weight_vox, 2.0)

        ################################################################################################################
        # Step 3: estimate noise level
        ################################################################################################################

        est_noise = np.sum(np.multiply(filt_vox, weight_vox)) / (36.0 * np.sum(weight_vox))
        est_noise = np.sqrt(est_noise)

    else:
        raise ValueError("The provided noise estimation method is not implemented. Use one of \"chang\" (default), \"rank\", \"ikeda\", \"immerkaer\" or \"zwanenburg\".")

    return est_noise


def get_supervoxels(img_obj, roi_obj, settings):
    """Extracts supervoxels from an image"""

    from skimage.segmentation import slic
    import copy

    # Check if image and/or roi exist, and skip otherwise
    if img_obj.is_missing or roi_obj.roi is None:
        return None

    # Get image object grid
    img_voxel_grid = copy.deepcopy(img_obj.get_voxel_grid())

    # Get grey level thresholds
    g_range = settings.roi_resegment.g_thresh
    if g_range[0] == np.nan:
        np.min(img_obj[roi_obj.roi.get_voxel_grid()])
    if g_range[1] == np.nan:
        np.max(img_obj[roi_obj.roi.get_voxel_grid()])

    # Add 10% range outside of the grey level range
    exp_range = 0.1 * (g_range[1] - g_range[0])
    g_range = np.array([g_range[0] - exp_range, g_range[1] + exp_range])

    # Apply threshold
    img_voxel_grid[img_voxel_grid < g_range[0]] = g_range[0]
    img_voxel_grid[img_voxel_grid > g_range[1]] = g_range[1]

    # Slic constants - sigma
    sigma = 1.0 * np.min(img_obj.spacing)

    # Slic constants - number of segments
    min_n_voxels = np.max([20.0, 500.0 / np.prod(img_obj.spacing)])
    n_segments = int(np.prod(img_obj.size) / min_n_voxels)

    # Convert to float with range [0.0, 1.0]
    img_voxel_grid -= g_range[0]
    img_voxel_grid *= 1.0 / (g_range[1]-g_range[0])

    if img_voxel_grid.dtype not in ["float", "float64"]:
        img_voxel_grid = img_voxel_grid.astype(np.float)

    # Create a slic segmentation of the image stack
    img_segments = slic(image=img_voxel_grid, n_segments=n_segments, sigma=sigma, spacing=img_obj.spacing,
                        compactness=0.05, multichannel=False, convert2lab=False, enforce_connectivity=True)
    img_segments += 1

    # Release img_voxel_grid
    del img_voxel_grid

    return img_segments


def get_supervoxel_overlap(roi_obj, img_segments, mask=None):
    """Determines overlap of supervoxels with other the region of interest"""

    # Return None in case image segments and/or ROI are missing
    if img_segments is None or roi_obj.roi is None:
        return None, None, None

    # Check segments overlapping with the current contour
    if mask == "morphological" and roi_obj.roi_morphology is not None:
        overlap_segment_labels, overlap_size = np.unique(np.multiply(img_segments, roi_obj.roi_morphology.get_voxel_grid()), return_counts=True)
    elif mask == "intensity" and roi_obj.roi_intensity is not None:
        overlap_segment_labels, overlap_size = np.unique(np.multiply(img_segments, roi_obj.roi_intensity.get_voxel_grid()), return_counts=True)
    else:
        overlap_segment_labels, overlap_size = np.unique(np.multiply(img_segments, roi_obj.roi.get_voxel_grid()), return_counts=True)

    # Find super voxels with non-zero overlap with the roi
    overlap_size           = overlap_size[overlap_segment_labels > 0]
    overlap_segment_labels = overlap_segment_labels[overlap_segment_labels > 0]

    # Check the actual size of the segments overlapping with the current contour
    segment_size = list(map(lambda x: np.sum([img_segments == x]), overlap_segment_labels))

    # Calculate the fraction of overlap
    overlap_frac = overlap_size / segment_size

    return overlap_segment_labels, overlap_frac, overlap_size


def transform_images(img_obj, roi_list, settings, compute_features=False, extract_images=False, file_path=None):
    """
    Performs image transformations and calculates features.
    :param img_obj: image object
    :param roi_list: list of region of interest objects
    :param settings: configuration settings
    :param compute_features: flag to enable feature computation
    :param extract_images: flag to enable image exports
    :param file_path: path for image exports
    :return: list of features computed in the transformed image
    """

    # Empty list for storing features
    feat_list = []

    # Check if image transformation is required
    if not settings.img_transform.perform_img_transform:
        return feat_list

    # Get spatial filters to apply
    spatial_filter = settings.img_transform.spatial_filters

    # Iterate over spatial filters
    for curr_filter in spatial_filter:

        if curr_filter == "wavelet":
            # Wavelet filters
            from mirp.imageFilters.waveletFilter import WaveletFilter

            filter_obj = WaveletFilter(settings=settings)
            feat_list += filter_obj.apply_transformation(img_obj=img_obj, roi_list=roi_list, settings=settings,
                                                         compute_features=compute_features, extract_images=extract_images,
                                                         file_path=file_path)

        elif curr_filter == "laplacian_of_gaussian":
            # Laplacian of Gaussian filters
            from mirp.imageFilters.laplacianOfGaussian import LaplacianOfGaussianFilter

            filter_obj = LaplacianOfGaussianFilter(settings=settings)
            feat_list += filter_obj.apply_transformation(img_obj=img_obj, roi_list=roi_list, settings=settings,
                                                         compute_features=compute_features, extract_images=extract_images,
                                                         file_path=file_path)

        elif curr_filter == "laws":
            # Laws' kernels
            from mirp.imageFilters.lawsFilter import LawsFilter

            filter_obj = LawsFilter(settings=settings)
            feat_list += filter_obj.apply_transformation(img_obj=img_obj, roi_list=roi_list, settings=settings,
                                                         compute_features=compute_features, extract_images=extract_images,
                                                         file_path=file_path)
        elif curr_filter == "gabor":
            # Gabor kernels
            from mirp.imageFilters.gaborFilter import GaborFilter

            filter_obj = GaborFilter(settings=settings)
            feat_list += filter_obj.apply_transformation(img_obj=img_obj,
                                                         roi_list=roi_list,
                                                         settings=settings,
                                                         compute_features=compute_features,
                                                         extract_images=extract_images,
                                                         file_path=file_path)

        elif curr_filter == "mean":
            # Mean / uniform filter
            from mirp.imageFilters.meanFilter import MeanFilter

            filter_obj = MeanFilter(settings=settings)
            feat_list += filter_obj.apply_transformation(img_obj=img_obj, roi_list=roi_list, settings=settings,
                                                         compute_features=compute_features, extract_images=extract_images,
                                                         file_path=file_path)

        else:
            raise ValueError(f"{curr_filter} is not implemented as a spatial filter. Please use one of wavelet, laplacian_of_gaussian, mean or laws.")

    return feat_list


def crop_image(img_obj, roi_list=None, roi_obj=None, boundary=0.0, z_only=False):
    """ The function is used to slice a subsection of the image so that further processing is facilitated in terms of
     memory and computational requirements. """

    ####################################################################################################################
    # Initial steps
    ####################################################################################################################

    # Temporarily parse roi_obj to list, if roi_obj is provided and not roi_list. This is done for easier code maintenance.
    if roi_list is None:
        roi_list = [roi_obj]
        return_roi_obj = True
    else:
        return_roi_obj = False

    ####################################################################################################################
    # Determine region of interest bounding box
    ####################################################################################################################
    roi_ext_x = [];  roi_ext_y = []; roi_ext_z = []

    # Determine extent of all rois
    for roi_obj in roi_list:

        # Skip if the ROI is missing
        if roi_obj.roi is None:
            continue

        z_ind, y_ind, x_ind = np.where(roi_obj.roi.get_voxel_grid() > 0.0)

        # Skip if the ROI is empty
        if len(z_ind) == 0 or len(y_ind) == 0 or len(x_ind) == 0:
            continue

        roi_ext_z += [np.min(z_ind), np.max(z_ind)]
        roi_ext_y += [np.min(y_ind), np.max(y_ind)]
        roi_ext_x += [np.min(x_ind), np.max(x_ind)]

    # Check if the combined ROIs are empty
    if not (len(roi_ext_z) == 0 or len(roi_ext_y) == 0 or len(roi_ext_x) == 0):

        # Express boundary in voxels.
        boundary = np.ceil(boundary / img_obj.spacing).astype(np.int)

        # Concatenate extents for rois and add boundary to generate map extent
        ind_ext_z = np.array([np.min(roi_ext_z) - boundary[0], np.max(roi_ext_z) + boundary[0]])
        ind_ext_y = np.array([np.min(roi_ext_y) - boundary[1], np.max(roi_ext_y) + boundary[1]])
        ind_ext_x = np.array([np.min(roi_ext_x) - boundary[2], np.max(roi_ext_x) + boundary[2]])

        ####################################################################################################################
        # Resect image based on roi extent
        ####################################################################################################################

        img_res = img_obj.copy()
        img_res.crop(ind_ext_z=ind_ext_z, ind_ext_y=ind_ext_y, ind_ext_x=ind_ext_x, z_only=z_only)

        ####################################################################################################################
        # Resect rois based on roi extent
        ####################################################################################################################

        # Copy roi objects before resection
        roi_res_list = [roi_res_obj.copy() for roi_res_obj in roi_list]

        # Resect in place
        [roi_res_obj.crop(ind_ext_z=ind_ext_z, ind_ext_y=ind_ext_y, ind_ext_x=ind_ext_x, z_only=z_only) for roi_res_obj in roi_res_list]

    else:
        # This happens if all rois are empty - only copies of the original image object and the roi are returned
        img_res = img_obj.copy()
        roi_res_list = [roi_res_obj.copy() for roi_res_obj in roi_list]

    ####################################################################################################################
    # Return to calling function
    ####################################################################################################################

    if return_roi_obj:
        return img_res, roi_res_list[0]
    else:
        return img_res, roi_res_list


def crop_image_to_size(img_obj, crop_size, roi_list=None, roi_obj=None):

    ####################################################################################################################
    # Initial steps
    ####################################################################################################################

    # Temporarily parse roi_obj to list, if roi_obj is provided and not roi_list. This is done for easier code maintenance.
    if roi_list is None:
        roi_list = [roi_obj]
        return_roi_obj = True
    else:
        return_roi_obj = False

    # Make a local copy of crop size before any alterations are made.
    crop_size = deepcopy(crop_size)

    # Determine whether cropping is only done in-plane or volumetrically.
    if len(crop_size) < 3:
        xy_only = True
        if len(crop_size) == 1:
            crop_size = [np.nan, crop_size[0], crop_size[0]]
        else:
            crop_size = [np.nan] + crop_size
    else:
        xy_only = False

    # Skip processing if all crop sizes are NaN.
    if not np.all(np.isnan(crop_size)):

        ####################################################################################################################
        # Determine geometric center
        ####################################################################################################################
        roi_m_x = 0; roi_m_y = 0; roi_m_z = 0; roi_n = 0

        # Determine geometric center of all rois
        for roi_obj in roi_list:

            # Skip if the ROI is missing
            if roi_obj.roi is None:
                continue

            # Find mask index coordinates
            z_ind, y_ind, x_ind = np.where(roi_obj.roi.get_voxel_grid() > 0.0)

            # Skip if the ROI is empty
            if len(z_ind) == 0 or len(y_ind) == 0 or len(x_ind) == 0:
                continue

            # Sum over all positions
            roi_m_x += np.sum(x_ind)
            roi_m_y += np.sum(y_ind)
            roi_m_z += np.sum(z_ind)
            roi_n   += len(x_ind)

        # Check if the combined ROIs are empty
        if not (roi_n == 0):

            # Calculate the mean roi center
            roi_m_x = roi_m_x / roi_n
            roi_m_y = roi_m_y / roi_n
            roi_m_z = roi_m_z / roi_n

            ####################################################################################################################
            # Resect image based on roi center
            ####################################################################################################################

            img_crop = img_obj.copy()
            img_crop.crop_to_size(center=np.array([roi_m_z, roi_m_y, roi_m_x]), crop_size=crop_size, xy_only=xy_only)

            ####################################################################################################################
            # Resect rois based on roi extent
            ####################################################################################################################

            # Copy roi objects before resection
            roi_crop_list = [roi_crop_obj.copy() for roi_crop_obj in roi_list]

            # Resect in place
            [roi_crop_obj.crop_to_size(center=np.array([roi_m_z, roi_m_y, roi_m_x]), crop_size=crop_size, xy_only=xy_only) for roi_crop_obj in roi_crop_list]

        else:
            # This happens if all rois are empty - only copies of the original image object and the roi are returned
            img_crop = img_obj.copy()
            roi_crop_list = [roi_crop_obj.copy() for roi_crop_obj in roi_list]

    else:
        # This happens if cropping is not required - only copies of the original image object and the roi are returned
        img_crop = img_obj.copy()
        roi_crop_list = [roi_crop_obj.copy() for roi_crop_obj in roi_list]

    ####################################################################################################################
    # Return to calling function
    ####################################################################################################################

    if return_roi_obj:
        return img_crop, roi_crop_list[0]
    else:
        return img_crop, roi_crop_list


def discretise_image_intensities(img_obj, roi_obj, discr_method="none", bin_width=None, bin_number=None):

    # Check if the roi intensity mask has been generated
    if roi_obj.roi_intensity is None:
        roi_obj.generate_masks()

    # Copy roi_obj and img_obj
    img_discr = img_obj.copy()
    roi_discr = roi_obj.copy()

    # Assign a None type to img_g if image object is missing or intensity mask could not be generated
    if img_obj.is_missing or roi_obj.roi_intensity is None:
        img_g = None
    else:
        # Only select voxel intensities in the roi
        img_g = np.unique(img_discr.get_voxel_grid()[roi_discr.roi_intensity.get_voxel_grid()])

    # Normal discretisation procedures with non-empty roi
    if discr_method == "none":
        if img_g is None:
            roi_discr.g_range = [np.nan, np.nan]
        elif len(img_g) > 0:
            roi_discr.g_range = [np.min(img_g), np.max(img_g)]
        else:
            # In case of empty roi
            roi_discr.g_range = [np.nan, np.nan]

    if discr_method == "fixed_bin_number":
        if img_g is None:
            roi_discr.g_range = [1.0, bin_number]

        elif len(img_g) > 0:
            # Set minimum and maximum intensity
            min_g = np.min(img_g)
            max_g = np.max(img_g)

            # Bin voxels. In the general case the minimum and maximum grey level are different. In the case they are the same,
            # all voxels are assigned the mean bin number.
            if max_g > min_g:
                img_vox = np.floor(bin_number * 1.0 * (img_discr.get_voxel_grid() - min_g) / (max_g - min_g)) + 1.0
            else:
                img_vox = np.zeros(shape=img_discr.size, dtype=np.float32) + np.ceil(bin_number/2.0)
            img_vox[img_vox <= 0.0] = 1.0
            img_vox[img_vox >= bin_number * 1.0] = bin_number * 1.0

            # Store to return image and roi
            img_discr.set_voxel_grid(voxel_grid=img_vox)
            roi_discr.g_range = [1.0, bin_number]

        else:
            # In case of empty roi
            img_vox = np.zeros(shape=img_discr.size, dtype=np.float32) + np.ceil(bin_number / 2.0)

            # Store to return image and roi
            img_discr.set_voxel_grid(voxel_grid=img_vox)
            roi_discr.g_range = [1.0, bin_number]

        # Update image discretisation settings
        img_discr.discretised = True
        img_discr.discretisation_algorithm = "fbn"
        img_discr.discretisation_settings = [bin_number]

    if discr_method == "fixed_bin_size":

        # Set minimum intensity
        if np.isnan(roi_obj.g_range[0]):
            if img_obj.modality == "CT":
                min_g = -1000.0
            elif img_obj.modality == "PT":
                min_g = 0.0
            elif len(img_g) > 0:
                min_g = np.min(img_g)
            else:
                raise ValueError("Minimum intensity for FBS discretisation could not be set.")
        else:
            min_g = roi_obj.g_range[0]

        # Discretise intensity levels
        if img_g is None:
            roi_discr.g_range = [1.0, 1.0]

        elif len(img_g) > 0:
            # Bin voxels
            img_vox = np.floor((img_discr.get_voxel_grid() - min_g) / (bin_width * 1.0)) + 1.0

            # Set voxels with grey level lower than 0.0 to 1.0. This may occur with non-roi voxels and voxels with the minimum intensity
            img_vox[img_vox <= 0.0] = 1.0

            # Determine number of bins
            n_bins = np.max(np.ravel(img_vox)[np.ravel(roi_discr.roi_intensity.get_voxel_grid())])

            # Limit to maximum number of bins
            img_vox[img_vox >= n_bins * 1.0] = n_bins * 1.0

            # Store to return image and roi
            img_discr.set_voxel_grid(voxel_grid=img_vox)
            roi_discr.g_range = [1.0, n_bins]

        else:
            # In case of an empty roi, set img to ones
            img_vox = np.ones(shape=img_discr.size, dtype=np.float32)

            # Store to return image and roi
            img_discr.set_voxel_grid(voxel_grid=img_vox)
            roi_discr.g_range = [1.0, 1.0]

        # Update image discretisation settings
        img_discr.discretised = True
        img_discr.discretisation_algorithm = "fbs"
        img_discr.discretisation_settings = [bin_width]

    return img_discr, roi_discr


def interpolate_to_new_grid(orig_dim,
                            orig_spacing,
                            orig_vox,
                            sample_dim=None,
                            sample_spacing=None,
                            grid_origin=None,
                            translation=np.array([0.0, 0.0, 0.0]), order=1, mode="nearest", align_to_center=True, processor="scipy"):
    """
    Resamples input grid and returns the output grid.
    :param orig_dim: dimensions of the input grid
    :param orig_origin: origin (in world coordinates) of the input grid
    :param orig_spacing: spacing (in world measures) of the input grid
    :param orig_vox: input grid
    :param sample_dim: desired output size (determined within the function if None)
    :param sample_origin: desired output origin (in world coordinates; determined within the function if None)
    :param sample_spacing: desired sample spacing (in world measures; should be provided if sample_dim or sample_origin is None)
    :param translation: a translation vector that is used to shift the interpolation grid (in voxel measures)
    :param order: interpolation spline order (0=nnb, 1=linear, 2=order 2 spline, 3=cubic splice, max 5).
    :param mode: describes how to handle extrapolation beyond input grid.
    :param align_to_center: whether the input and output grids should be aligned by their centers (True) or their origins (False)
    :param processor: which function to use for interpolation: "scipy" for scipy.ndimage.map_coordinates and "sitk" for SimpleITK.ResampleImageFilter
    :return:
    """

    # Check if sample spacing is provided
    if sample_dim is None and sample_spacing is None:
        logging.error("Sample spacing is required for interpolation, but not provided.")

    # If no sample spacing is provided, assume original spacing. Note that for most purposes sample spacing should be provided
    if sample_spacing is None:
        sample_spacing = orig_spacing

    # Set sample spacing and orig_spacing to float
    sample_spacing = sample_spacing.astype(np.float)
    orig_spacing = orig_spacing.astype(np.float)

    # If no sample dimensions are provided, assume that the user wants to sample the original grid
    if sample_dim is None:
        sample_dim = np.ceil(np.multiply(orig_dim, orig_spacing / sample_spacing))

    # Set grid spacing (i.e. a fractional spacing in input voxel dimensions)
    grid_spacing = sample_spacing / orig_spacing

    # Set grid origin, if not provided previously
    if grid_origin is None:
        if align_to_center:
            grid_origin = 0.5 * (np.array(orig_dim) - 1.0) - 0.5 * (np.array(sample_dim) - 1.0) * grid_spacing

        else:
            grid_origin = np.array([0.0, 0.0, 0.0])

        # Update with translation vector
        grid_origin += translation * grid_spacing

    if processor == "scipy":
        import scipy.ndimage as ndi

        # Convert sample_spacing and sample_origin to normalised original spacing (where voxel distance is 1 in each direction)
        # This is required for the use of ndi.map_coordinates, which uses the original grid as reference.

        # Generate interpolation map grid
        map_z, map_y, map_x = np.mgrid[:sample_dim[0], :sample_dim[1], :sample_dim[2]]

        # Transform map to normalised original space
        map_z = map_z * grid_spacing[0] + grid_origin[0]
        map_z = map_z.astype(np.float32)
        map_y = map_y * grid_spacing[1] + grid_origin[1]
        map_y = map_y.astype(np.float32)
        map_x = map_x * grid_spacing[2] + grid_origin[2]
        map_x = map_x.astype(np.float32)

        # Interpolate orig_vox on interpolation grid
        map_vox = ndi.map_coordinates(input=orig_vox.astype(np.float32),
                                      coordinates=np.array([map_z, map_y, map_x], dtype=np.float32),
                                      order=order,
                                      mode=mode)

    elif processor == "sitk":
        import SimpleITK as sitk

        # Convert input voxel grid to sitk image. Note that SimpleITK expects x,y,z ordering, while we use z,y,
        # x ordering. Hence origins, spacings and sizes are inverted for both input image (sitk_orig_img) and
        # ResampleImageFilter objects.
        sitk_orig_img = sitk.GetImageFromArray(orig_vox.astype(np.float32), isVector=False)
        sitk_orig_img.SetOrigin(np.array([0.0, 0.0, 0.0]))
        sitk_orig_img.SetSpacing(np.array([1.0, 1.0, 1.0]))

        interpolator = sitk.ResampleImageFilter()

        # Set interpolator algorithm; SimpleITK has more interpolators, but for now use the older scheme for scipy.
        if order == 0:
            interpolator.SetInterpolator(sitk.sitkNearestNeighbor)
        elif order == 1:
            interpolator.SetInterpolator(sitk.sitkLinear)
        elif order == 2:
            interpolator.SetInterpolator(sitk.sitkBSplineResamplerOrder2)
        elif order == 3:
            interpolator.SetInterpolator(sitk.sitkBSpline)

        # Set output origin and output spacing
        interpolator.SetOutputOrigin(grid_origin[::-1])
        interpolator.SetOutputSpacing(grid_spacing[::-1])
        interpolator.SetSize(sample_dim[::-1].astype(int).tolist())

        map_vox = sitk.GetArrayFromImage(interpolator.Execute(sitk_orig_img))
    else:
        raise ValueError("The selected processor should be one of \"scipy\" or \"sitk\"")

    # Return interpolated grid and spatial coordinates
    return sample_dim, sample_spacing, map_vox, grid_origin


def gaussian_preprocess_filter(orig_vox, orig_spacing, sample_spacing=None, param_beta=0.93, mode="nearest", by_slice=False):

    import scipy.ndimage

    # If no sample spacing is provided, assume original spacing. Note that for most purposes sample spacing should be provided
    if sample_spacing is None:
        sample_spacing = orig_spacing

    # Set sample spacing and orig_spacing to float
    sample_spacing = sample_spacing.astype(np.float)
    orig_spacing = orig_spacing.astype(np.float)

    # Calculate the zoom factors
    map_spacing = sample_spacing / orig_spacing

    # Only apply to down-sampling (map_spacing > 1.0)
    # map_spacing[map_spacing<=1.0] = 0.0

    # Don't filter along slices if calculations are to occur within the slice only
    if by_slice: map_spacing[0] = 0.0

    # Calculate sigma
    sigma = np.sqrt(-8 * np.power(map_spacing, 2.0) * np.log(param_beta))

    # Apply filter
    new_vox = scipy.ndimage.gaussian_filter(input=orig_vox.astype(np.float32), sigma=sigma, order=0, mode=mode)

    return new_vox


def divide_tumour_regions(roi_list, settings):

    # Create new list for storing roi boundaries and bulk
    new_roi_list = []

    # Get the boundary size.
    boundary_size_list = settings.vol_adapt.roi_boundary_size

    # Skip processing when no objects are requested
    if boundary_size_list == [0.0]:
        return roi_list

    # Iterate over rois
    for roi_obj in roi_list:
        # Store original
        new_roi_list += [roi_obj]

        for boundary_size in boundary_size_list:

            # Skip if the boundary has no size
            if boundary_size == 0.0: continue

            # Get copy for roi bulk and set names
            bulk_roi_obj = roi_obj.copy()
            bulk_roi_obj.name += "_bulk" + str(boundary_size)
            bulk_roi_obj.adapt_size = boundary_size

            # Get copy of roi boundary and set names
            boundary_roi_obj = roi_obj.copy()
            boundary_roi_obj.name += "_boundary" + str(boundary_size)
            boundary_roi_obj.adapt_size = boundary_size

            # Remove boundary from the roi to generate the bulk
            bulk_roi_obj.erode(by_slice=settings.general.by_slice, dist=-boundary_size,
                               eroded_vol_fract=settings.vol_adapt.bulk_min_vol_fract)

            # Get roi boundary if roi exists
            if roi_obj.roi is not None:
                boundary_roi_obj.roi.set_voxel_grid(voxel_grid=np.logical_xor(roi_obj.roi.get_voxel_grid(), bulk_roi_obj.roi.get_voxel_grid()))

            # Check whether the bulk and boundary roi object are empty or not
            if not bulk_roi_obj.is_empty() and not boundary_roi_obj.is_empty():

                # Store bulk and boundary rois to list when both are not empty
                new_roi_list += [bulk_roi_obj, boundary_roi_obj]

    # Return rois
    return new_roi_list


# def selectHeterogeneousSuperVoxels(img_obj, roi_list, settings, file_str):
#     """Selects a roi of heterogeneous supervoxels"""
#
#     from utilities import world_to_index
#     import copy
#     import re
#
#     # Check if
#     if settings.vol_adapt.heterogeneous_svx_count <= 0.0:
#         return roi_list
#
#     new_roi_list = []
#
#     # Adapt the settings file to extract the run length matrix
#     svx_settings = copy.deepcopy(settings)
#
#     # Parse required feature families and add
#     svx_settings.feature_extr.families = np.unique(
#         [re.split(pattern="_", string=x)[0] for x in svx_settings.vol_adapt.heterogeneity_features]).tolist()
#
#     # Iterate over roi objects
#     for roi_ind in np.arange(0, len(roi_list)):
#
#         # Resect image to speed up segmentation process
#         res_img_obj, res_roi_obj = crop_image(img_obj=img_obj, roi_obj=roi_list[roi_ind], boundary=25.0, z_only=False)
#
#         # Get supervoxels
#         img_segments = getSuperVoxels(img_obj=res_img_obj, roi_obj=res_roi_obj, settings=settings)
#
#         # Determine overlap of supervoxels with contour
#         overlap_indices, overlap_fract, overlap_size = getSuperVoxelOverlap(roi_obj=res_roi_obj, img_segments=img_segments, mask="intensity")
#
#         # Create a data frame
#         df_svx = pd.DataFrame(data={"svx_index": overlap_indices,
#                                     "overlap": overlap_fract,
#                                     "volume": overlap_size * np.prod(res_img_obj.spacing)})
#
#         # Set the highest overlap to 1.0 to ensure selection of at least 1 supervoxel
#         df_svx.loc[np.argmax(df_svx.overlap), "overlap"] = 1.0
#
#         # Select all supervoxels with 67% or more overlap
#         df_svx = df_svx.loc[df_svx.overlap >= 0.67, :]
#         # selected_svx_indices = overlap_indices[overlap_fract >= 0.67]
#
#         # Iterate over selected indices and add to list
#         svx_roi_list = []
#         for ii in np.arange(0, df_svx.shape[0]):
#
#             # Make local of the roi and replace the intensity grid
#             svx_roi = res_roi_obj.copy()
#             svx_roi.roi_intensity.set_voxel_grid(voxel_grid=img_segments == df_svx.svx_index.values[ii])
#             svx_roi.name += "_svx" + str(df_svx.svx_index.values[ii])
#
#             # Add to roi list
#             svx_roi_list += [svx_roi]
#
#         # Calculate rlm features and extract
#         df_feat = calculateFeatures(img_obj=res_img_obj, roi_list=svx_roi_list, settings=svx_settings)
#
#         # Save the features used for heterogeneity
#         file_name = file_str + "_" + res_roi_obj.name + "_heterogeneity.csv"
#         pd.concat([df_feat.filter(regex="|".join(svx_settings.vol_adapt.heterogeneity_features)).reset_index(drop=True),
#                    df_svx.reset_index(drop=True)], axis=1).to_csv(path_or_buf=file_name, index=False, sep=";", decimal=".")
#
#         # Iterate over features
#         for ii in np.arange(len(svx_settings.vol_adapt.heterogeneity_features)):
#
#             # Select feature columns
#             df_svx["ranks"] = df_feat.filter(regex=svx_settings.vol_adapt.heterogeneity_features[ii]).rank(axis=0, ascending=svx_settings.vol_adapt.heterogen_low_values[ii]).sum(axis=1).rank().values
#             heterogeneous_svx_indices = df_svx.svx_index.values[df_svx.ranks <= settings.vol_adapt.heterogeneous_svx_count]
#
#             # Determine grid indices of the resected grid with respect to the original image grid
#             grid_origin = world_to_index(coord=res_img_obj.origin, origin=img_obj.origin, spacing=img_obj.spacing)
#             grid_origin = grid_origin.astype(np.int)
#
#             # Replace randomised contour in original roi voxel space
#             roi_vox = np.zeros(shape=roi_list[roi_ind].roi.size, dtype=np.bool)
#             roi_vox[grid_origin[0]: grid_origin[0] + res_roi_obj.roi.size[0],
#                     grid_origin[1]: grid_origin[1] + res_roi_obj.roi.size[1],
#                     grid_origin[2]: grid_origin[2] + res_roi_obj.roi.size[2], ] = \
#                 np.reshape(np.in1d(np.ravel(img_segments), heterogeneous_svx_indices), res_roi_obj.roi.size)
#
#             # Update voxels in original roi, and adapt name
#             heter_roi = roi_list[roi_ind].copy()
#             heter_roi.roi.set_voxel_grid(voxel_grid=roi_vox)  # Replace copied original contour with randomised contour
#             heter_roi.name += "_heterogeneous_" + svx_settings.vol_adapt.heterogeneity_features[ii]   # Adapt roi name
#
#             new_roi_list += [heter_roi]
#
#             del roi_vox
#         #
#         # # Find columns containing rlm_sre features and select the most heterogeneous voxels
#         # df_feat.filter(regex="|".join(svx_settings.vol_adapt.heterogeneity_features))
#         # ranks = df_rlm.filter(regex="rlm_sre").rank(axis=0, ascending=False).sum(axis=1).rank()
#         # heterogeneous_svx_indices = selected_svx_indices[ranks.values <= settings.vol_adapt.heterogeneous_svx_count]
#         #
#         # # Determine grid indices of the resected grid with respect to the original image grid
#         # grid_origin = worldToIndex(coord=res_img_obj.origin, origin=img_obj.origin, spacing=img_obj.spacing)
#         # grid_origin = grid_origin.astype(np.int)
#         #
#         # # Replace randomised contour in original roi voxel space
#         # roi_vox = np.zeros(shape=roi_list[roi_ind].roi.size, dtype=np.bool)
#         # roi_vox[grid_origin[0]: grid_origin[0] + res_roi_obj.roi.size[0],
#         #         grid_origin[1]: grid_origin[1] + res_roi_obj.roi.size[1],
#         #         grid_origin[2]: grid_origin[2] + res_roi_obj.roi.size[2], ] = \
#         #     np.reshape(np.in1d(np.ravel(img_segments), heterogeneous_svx_indices), res_roi_obj.roi.size)
#         #
#         # # Update voxels in original roi, and adapt name
#         # heter_roi = roi_list[roi_ind].copy()
#         # heter_roi.roi.setVoxelGrid(voxel_grid=roi_vox)  # Replace copied original contour with randomised contour
#         # heter_roi.name += "_heterogeneous"              # Adapt roi name
#         #
#         # new_roi_list += [heter_roi]
#
#         del res_img_obj, res_roi_obj
#
#     # Re-apply resegmentisation
#     new_roi_list = resegmentise(img_obj=img_obj, roi_list=new_roi_list, settings=settings)
#
#     return new_roi_list


# def selectNonEmptyRegions(roi_list):
#     """Shrink roi list to exclude empty regions of interest"""
#
#     # Initialise roi list
#     new_roi_list = []
#
#     for roi_ind in np.arange(0, len(roi_list)):
#
#         # Check if roi or one of its masks is empty
#         if roi_list[roi_ind].is_empty():
#             continue
#         else:
#             new_roi_list += [roi_list[roi_ind]]
#
#     return new_roi_list


def calculate_features(img_obj, roi_list, settings, append_str=""):
    """
    Calculate image features from the provided data
    :param img_obj:
    :param roi_list:
    :param settings:
    :param append_str:
    :return:
    """

    from mirp.featureSets.localIntensity import get_local_intensity_features
    from mirp.featureSets.statistics import get_intensity_statistics_features
    from mirp.featureSets.intensityVolumeHistogram import get_intensity_volume_histogram_features
    from mirp.featureSets.volumeMorphology import get_volumetric_morphological_features

    feat_list = []

    for roi_ind in np.arange(0, len(roi_list)):

        roi_feat_list = []

        ################################################################################################################
        # Local mapping features
        ################################################################################################################

        if np.any([np.in1d(["li", "loc.int", "loc_int", "local_int", "local_intensity", "all"], settings.feature_extr.families)]):
            # Cut roi and image with 10 mm boundary
            img_cut, roi_cut = crop_image(img_obj=img_obj, roi_obj=roi_list[roi_ind], boundary=10.0)

            # Decode roi voxel grid
            roi_cut.decode_voxel_grid()

            # Calculate local intensities
            roi_feat_list += [get_local_intensity_features(img_obj=img_cut, roi_obj=roi_cut)]

            # Clean up
            del img_cut, roi_cut

        ################################################################################################################
        # ROI features without discretisation
        ################################################################################################################

        # Cut roi and image to image
        img_cut, roi_cut = crop_image(img_obj=img_obj, roi_obj=roi_list[roi_ind], boundary=0.0)

        # Decode roi voxel grid
        roi_cut.decode_voxel_grid()

        # Extract statistical features
        if np.any([np.in1d(["st", "stat", "stats", "statistics", "statistical", "all"], settings.feature_extr.families)]):
            roi_feat_list += [get_intensity_statistics_features(img_obj=img_cut, roi_obj=roi_cut)]

        # Calculate intensity volume histogram features
        if np.any([np.in1d(["ivh", "int_vol_hist", "intensity_volume_histogram", "all"], settings.feature_extr.families)]):
            roi_feat_list += [get_intensity_volume_histogram_features(img_obj=img_cut, roi_obj=roi_cut, settings=settings)]

        # Calculate morphological features
        if np.any([np.in1d(["mrp", "morph", "morphology", "morphological", "all"], settings.feature_extr.families)]):
            roi_feat_list += [get_volumetric_morphological_features(img_obj=img_cut, roi_obj=roi_cut, settings=settings)]

        ################################################################################################################
        # ROI features with discretisation
        ################################################################################################################

        for discr_method in settings.feature_extr.discr_method:

            # Skip discretisation when we are working from a transformed image without fixed_bin_number discretisation
            if not img_obj.spat_transform == "base" and discr_method not in ["fixed_bin_number"]:
                continue

            if discr_method in ["fixed_bin_size"]:
                for bin_width in settings.feature_extr.discr_bin_width:
                    roi_feat_list += [compute_discretised_features(img_obj=img_cut, roi_obj=roi_cut, settings=settings,
                                                                   discr_method=discr_method, bin_width=bin_width,
                                                                   bin_number=None)]
            if discr_method in ["fixed_bin_number"]:
                for bin_number in settings.feature_extr.discr_n_bins:
                    roi_feat_list += [compute_discretised_features(img_obj=img_cut, roi_obj=roi_cut, settings=settings,
                                                                   discr_method=discr_method, bin_width=None,
                                                                   bin_number=bin_number)]
            if discr_method in ["none"]:
                roi_feat_list += [compute_discretised_features(img_obj=img_cut, roi_obj=roi_cut, settings=settings,
                                                               discr_method=discr_method, bin_width=None,
                                                               bin_number=None)]

        ################################################################################################################
        # Concatenate and parse feature tables for the ROI
        ################################################################################################################

        # Concatenate
        df_roi_feat = pd.concat(roi_feat_list, axis=1)

        df_roi_feat.columns = append_str + df_roi_feat.columns.values

        feat_list += [df_roi_feat]

    ####################################################################################################################
    # Concatenate and parse feature tables for the complete analysis
    ####################################################################################################################

    # Concatenate feature data frames
    if len(feat_list) > 0:
        df_feat = pd.concat(feat_list, axis=0)

        return df_feat

    else:
        return None


def compute_discretised_features(img_obj, roi_obj, settings, discr_method="none", bin_width=None, bin_number=None):
    """Function to process and calculate discretised image features"""

    from mirp.featureSets.intensityHistogram import get_intensity_histogram_features
    from mirp.featureSets.cooccurrenceMatrix import get_cm_features
    from mirp.featureSets.runLengthMatrix import get_rlm_features
    from mirp.featureSets.sizeZoneMatrix import get_szm_features
    from mirp.featureSets.distanceZoneMatrix import get_dzm_features
    from mirp.featureSets.neighbourhoodGreyToneDifferenceMatrix import get_ngtdm_features
    from mirp.featureSets.neighbouringGreyLevelDifferenceMatrix import get_ngldm_features

    # Apply image discretisation
    img_discr, roi_discr = discretise_image_intensities(img_obj=img_obj, roi_obj=roi_obj, discr_method=discr_method,
                                                        bin_width=bin_width, bin_number=bin_number)

    # Decode roi object
    roi_discr.decode_voxel_grid()

    # Initiate empty feature list
    feat_list = []

    # Intensity histogram
    if np.any([np.in1d(["ih", "int_hist", "int_histogram", "intensity_histogram", "all"], settings.feature_extr.families)]):
        feat_list += [get_intensity_histogram_features(img_obj=img_discr, roi_obj=roi_discr)]

    # Grey level cooccurrence matrix
    if np.any([np.in1d(["cm", "glcm", "grey_level_cooccurrence_matrix", "cooccurrence_matrix", "all"], settings.feature_extr.families)]):
        feat_list += [get_cm_features(img_obj=img_discr, roi_obj=roi_discr, settings=settings)]

    # Grey level run length matrix
    if np.any([np.in1d(["rlm", "glrlm", "grey_level_run_length_matrix", "run_length_matrix", "all"], settings.feature_extr.families)]):
        feat_list += [get_rlm_features(img_obj=img_discr, roi_obj=roi_discr, settings=settings)]

    # Grey level size zone matrix
    if np.any([np.in1d(["szm", "glszm", "grey_level_size_zone_matrix", "size_zone_matrix", "all"], settings.feature_extr.families)]):
        feat_list += [get_szm_features(img_obj=img_discr, roi_obj=roi_discr, settings=settings)]

    # Grey level distance zone matrix
    if np.any([np.in1d(["dzm", "gldzm", "grey_level_distance_zone_matrix", "distance_zone_matrix", "all"], settings.feature_extr.families)]):
        feat_list += [get_dzm_features(img_obj=img_discr, roi_obj=roi_discr, settings=settings)]

    # Neighbourhood grey tone difference matrix
    if np.any([np.in1d(["tdm", "ngtdm", "neighbourhood_grey_tone_difference_matrix", "grey_tone_difference_matrix", "all"], settings.feature_extr.families)]):
        feat_list += [get_ngtdm_features(img_obj=img_discr, roi_obj=roi_discr, settings=settings)]

    # Neighbouring grey level dependence matrix
    if np.any([np.in1d(["ldm", "ngldm", "neighbouring_grey_level_dependence_matrix", "grey_level_dependence_matrix", "all"], settings.feature_extr.families)]):
        feat_list += [get_ngldm_features(img_obj=img_discr, roi_obj=roi_discr, settings=settings)]

    # Check if any features were added to the feature list; otherwise return to main function
    if len(feat_list) == 0:
        return None

    # Concatenate list of feature tables
    df_feat = pd.concat(feat_list, axis=1)

    # Parse name
    parse_str = ""

    # Add discretisation method to string
    if discr_method == "fixed_bin_size": parse_str += "_fbs"
    if discr_method == "fixed_bin_number": parse_str += "_fbn"

    # Add bin witdth/ bin number to string
    if bin_width is not None: parse_str += "_w" + str(bin_width)
    if bin_number is not None: parse_str += "_n" + str(int(bin_number))

    df_feat.columns += parse_str

    return df_feat


def create_tissue_mask(img_obj: ImageClass, settings: SettingsClass):

    if settings.post_process.tissue_mask_type == "none":
        # The entire image is the tissue mask.
        mask = np.ones(img_obj.size, dtype=np.uint8)

    elif settings.post_process.tissue_mask_type == "range":
        # The intensity range provided forms the mask range.
        tissue_range = deepcopy(settings.post_process.tissue_mask_range)
        if np.isnan(tissue_range[1]): tissue_range[1] = 0.0
        if np.isnan(tissue_range[2]): tissue_range[2] = np.max(img_obj.get_voxel_grid())

        voxel_grid = img_obj.get_voxel_grid()
        mask = np.logical_and(voxel_grid >= tissue_range[1], voxel_grid <= tissue_range[2])

    elif settings.post_process.tissue_mask_type == "relative_range":
        # The relative intensity range provided forms the mask range. This means that we need to convert the relative
        # range to the range present in the image.
        tissue_range = deepcopy(settings.post_process.tissue_mask_range)
        if np.isnan(tissue_range[0]): tissue_range[0] = 0.0
        if np.isnan(tissue_range[1]): tissue_range[1] = 1.0

        voxel_grid = img_obj.get_voxel_grid()
        intensity_range = [np.min(voxel_grid), np.max(voxel_grid)]

        # Convert relative range to the image intensities
        tissue_range = [intensity_range[0] + tissue_range[0] * (intensity_range[1] - intensity_range[0]),
                        intensity_range[0] + tissue_range[1] * (intensity_range[1] - intensity_range[0])]

        mask = np.logical_and(voxel_grid >= tissue_range[0], voxel_grid <= tissue_range[1])
    else:
        raise ValueError(f"The tissue_mask_type configuration parameter is expected to be one of none, range, "
                         f"or relative_range. Encountered: {settings.post_process.tissue_mask_type}")

    return mask


def bias_field_correction(img_obj: ImageClass, settings: SettingsClass, mask=None):
    import itk

    if not settings.post_process.bias_field_correction:
        return img_obj

    if img_obj.modality != "MR":
        return img_obj

    if mask is None:
        mask = np.ones(img_obj.size, dtype=np.uint8)

    # Create ITK input masks
    input_image = itk.GetImageFromArray(img_obj.get_voxel_grid())
    input_image.SetSpacing(img_obj.spacing[::-1])
    input_mask = itk.GetImageFromArray(mask.astype(np.uint8))
    input_mask.SetSpacing(img_obj.spacing[::-1])

    # Start N4 bias correction
    corrector = itk.N4BiasFieldCorrectionImageFilter.New(input_image, input_mask)
    corrector.SetNumberOfFittingLevels(settings.post_process.n_fitting_levels)
    corrector.SetMaximumNumberOfIterations(settings.post_process.n_max_iterations)
    corrector.SetConvergenceThreshold(settings.post_process.convergence_threshold)
    output_image = corrector.GetOutput()

    # Save bias-corrected image.
    img_obj.set_voxel_grid(voxel_grid=itk.GetArrayFromImage(output_image).astype(dtype=np.float32))

    return img_obj
