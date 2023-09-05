import warnings
from typing import Union, List, Tuple, Optional, Any
from copy import deepcopy

from mirp.settings.settingsClass import SettingsClass, FeatureExtractionSettingsClass
from mirp.imageClass import ImageClass
from mirp.roiClass import RoiClass

from mirp.images.genericImage import GenericImage
from mirp.images.maskImage import MaskImage
from mirp.masks.baseMask import BaseMask
from mirp.importData.utilities import flatten_list

import numpy as np
import pandas as pd


def _standard_checks(
        image: GenericImage,
        masks: Optional[Union[BaseMask, MaskImage, List[BaseMask]]]
):
    if masks is None:
        return image, None, None
    if isinstance(masks, list) and len(masks) == 0:
        return image, None, None

    # Determine the return format.
    return_list = False
    if isinstance(masks, list):
        return_list = True
    else:
        masks = [masks]

    if not isinstance(image, GenericImage):
        raise TypeError(
            f"The image argument is expected to be a GenericImage object, or inherit from it. Found: {type(image)}")

    if not all(isinstance(mask, BaseMask) or isinstance(mask, MaskImage) for mask in masks):
        raise TypeError(
            f"The masks argument is expected to be a BaseMask or MaskImage object, or a list thereof.")

    return image, masks, return_list


def set_intensity_range(
        image: GenericImage,
        mask: Optional[MaskImage] = None,
        intensity_range: Optional[Tuple[Any]] = None
) -> Tuple[float]:
    if intensity_range is not None and not np.any(np.isnan(intensity_range)):
        return intensity_range

    if mask is None or mask.is_empty() or mask.is_empty_mask():
        mask_data = np.ones(image.image_dimension, dtype=bool)
    else:
        mask_data = mask.get_voxel_grid()

    # Make intensity range mutable.
    if intensity_range is None:
        intensity_range = [np.nan, np.nan]
    else:
        intensity_range = list(intensity_range)

    if np.isnan(intensity_range[0]):
        intensity_range[0] = np.min(image.get_voxel_grid()[mask_data])
    if np.isnan(intensity_range[1]):
        intensity_range[1] = np.max(image.get_voxel_grid()[mask_data])

    return tuple(intensity_range)


def extend_intensity_range(
        intensity_range: Tuple[Any],
        extend_fraction=0.1
) -> Optional[Tuple[Any]]:
    if intensity_range is None or np.any(np.isnan(intensity_range)):
        return intensity_range

    if extend_fraction <= 0.0:
        return intensity_range

    # Add 10% range outside of the grey level range
    extension = 0.1 * (intensity_range[1] - intensity_range[0])
    intensity_range = list(intensity_range)
    intensity_range[0] -= extension
    intensity_range[1] += extension

    return tuple(intensity_range)


def crop(
        image: GenericImage,
        masks: Union[BaseMask, MaskImage, List[BaseMask]],
        boundary: float = 0.0,
        xy_only: bool = False,
        z_only: bool = False,
        in_place: bool = False,
        by_slice: bool = False
) -> Tuple[GenericImage, Optional[Union[BaseMask, MaskImage, List[BaseMask]]]]:
    """ The function is used to slice a subsection of the image so that further processing is facilitated in terms of
     memory and computational requirements. """

    image, masks, return_list = _standard_checks(image=image, masks=masks)
    if return_list is None:
        return image, None

    bounds_z: Optional[List[int]] = None
    bounds_y: Optional[List[int]] = None
    bounds_x: Optional[List[int]] = None

    for mask in masks:
        current_bounds_z, current_bounds_y, current_bounds_x = mask.get_bounding_box()
        if current_bounds_z is None or current_bounds_y is None or current_bounds_x is None:
            continue

        if bounds_z is None:
            bounds_z = list(current_bounds_z)
        else:
            bounds_z = [min(bounds_z[0], current_bounds_z[0]), max(bounds_z[1], current_bounds_z[1])]

        if bounds_y is None:
            bounds_y = list(current_bounds_y)
        else:
            bounds_y = [min(bounds_y[0], current_bounds_y[0]), max(bounds_y[1], current_bounds_y[1])]

        if bounds_x is None:
            bounds_x = list(current_bounds_x)
        else:
            bounds_x = [min(bounds_x[0], current_bounds_x[0]), max(bounds_x[1], current_bounds_x[1])]

    # Check if bounds were determined.
    if bounds_x is None or bounds_y is None or bounds_z is None:
        if return_list:
            return image, masks
        else:
            return image, masks[0]

    # Compute boundary and add to bounding box.
    boundary = np.ceil(boundary / np.array(image.image_spacing)).astype(int)

    if not by_slice:
        bounds_z = [bounds_z[0] - boundary[0], bounds_z[1] + boundary[0]]
    bounds_y = [bounds_y[0] - boundary[1], bounds_y[1] + boundary[1]]
    bounds_x = [bounds_x[0] - boundary[2], bounds_x[1] + boundary[2]]

    if not in_place:
        image = image.copy()
        masks = [mask.copy() for mask in masks]

    # Crop images and masks.
    image.crop(
        ind_ext_z=bounds_z,
        ind_ext_y=bounds_y,
        ind_ext_x=bounds_x,
        xy_only=xy_only,
        z_only=z_only
    )

    for mask in masks:
        mask.crop(
            ind_ext_z=bounds_z,
            ind_ext_y=bounds_y,
            ind_ext_x=bounds_x,
            xy_only=xy_only,
            z_only=z_only
        )

    if return_list:
        return image, masks
    else:
        return image, masks[0]


def crop_image_to_size(
        image: GenericImage,
        masks: Union[BaseMask, MaskImage, List[BaseMask]],
        crop_size: List[float],
        crop_center: Optional[List[float]] = None,
        xy_only: bool = False,
        z_only: bool = False,
        in_place: bool = False,
        by_slice: bool = False
) -> Tuple[GenericImage, Optional[Union[BaseMask, MaskImage, List[BaseMask]]]]:

    image, masks, return_list = _standard_checks(image=image, masks=masks)
    if return_list is None:
        return image, None

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


def resegmentise_mask(
        image: GenericImage,
        masks: Optional[Union[BaseMask, List[BaseMask]]],
        resegmentation_method: Optional[Union[str, List[str]]] = None,
        intensity_range: Optional[Tuple[Any, Any]] = None,
        sigma: Optional[float] = None
):
    # Resegmentises mask based on the selected method.
    image, masks, return_list = _standard_checks(image, masks)
    if return_list is None:
        return masks

    masks: List[BaseMask] = masks

    for mask in masks:
        mask.resegmentise_mask(
            image=image,
            resegmentation_method=resegmentation_method,
            intensity_range=intensity_range,
            sigma=sigma
        )

    if return_list:
        return masks
    else:
        return masks[0]


def split_masks(
        masks: Optional[Union[BaseMask, List[BaseMask]]],
        boundary_sizes: Optional[List[float]] = None,
        max_erosion: Optional[float] = 0.8,
        by_slice: bool = False
):
    if boundary_sizes is None or len(boundary_sizes) == 0 or \
            all(boundary_size == 0.0 for boundary_size in boundary_sizes):
        return masks

    if masks is None:
        return None

    # Determine the return format.
    if not isinstance(masks, list):
        masks = [masks]

    masks: List[BaseMask] = masks
    new_masks = []

    # Iterate over masks.
    for mask in masks:
        # Store original.
        new_masks += [mask]

        for boundary_size in boundary_sizes:
            if boundary_size == 0.0:
                continue

            bulk_mask = mask.copy()
            bulk_mask.roi_name += "_bulk_" + str(boundary_size)

            boundary_mask = mask.copy()
            boundary_mask.roi_name += "_boundary_" + str(boundary_size)

            bulk_mask.erode(
                by_slice=by_slice,
                distance=boundary_size,
                max_eroded_volume_fraction=max_erosion
            )

            boundary_mask.roi.set_voxel_grid(voxel_grid=np.logical_xor(
                mask.roi.get_voxel_grid(), bulk_mask.roi.get_voxel_grid()))

            if bulk_mask.roi.is_empty_mask() or boundary_mask.roi.is_empty_mask():
                continue

            new_masks += [bulk_mask, boundary_mask]

    return new_masks


def randomise_mask(
        image: GenericImage,
        masks: Union[BaseMask, MaskImage, List[BaseMask]],
        boundary: float = 25.0,
        repetitions: int = 1,
        by_slice: bool = False
):
    image, masks, return_list = _standard_checks(image=image, masks=masks)
    if return_list is None:
        return None

    new_masks = []
    for mask in masks:
        if isinstance(mask, MaskImage):
            randomised_masks = mask.randomise_mask(
                image=image,
                boundary=boundary,
                repetitions=repetitions,
                by_slice=by_slice
            )

            for randomised_mask in randomised_masks:
                if randomised_mask is None:
                    continue
                new_masks += [randomised_mask]

        elif isinstance(mask, BaseMask):
            randomised_masks = mask.roi.randomise_mask(
                image=image,
                boundary=boundary,
                repetitions=repetitions,
                intensity_range=mask.intensity_range,
                by_slice=by_slice
            )

            for randomised_mask in randomised_masks:
                if randomised_mask is None:
                    continue
                new_mask = mask.copy(drop_image=True)
                new_mask.roi = randomised_mask
                new_masks += [new_mask]
        else:
            raise TypeError("The masks attribute is expected to be MaskImage and BaseMask")

    new_masks = flatten_list(new_masks)
    if len(new_masks) == 0:
        return None

    if not return_list and repetitions == 1:
        return new_masks[0]
    else:
        return new_masks


def alter_mask(
        masks: Union[BaseMask, MaskImage, List[BaseMask]],
        alteration_size: Optional[List[float]] = None,
        alteration_method: Optional[str] = None,
        max_erosion: Optional[float] = 0.8,
        by_slice: bool = False
):
    """ Adapt roi size by growing or shrinking the roi """

    if alteration_size is None or alteration_method is None:
        return masks

    # Determine the return format.
    return_list = False
    if isinstance(masks, list):
        return_list = True
    else:
        masks = [masks]

    new_masks = []

    for mask in masks:
        for current_adapt_size in alteration_size:
            new_mask = mask.copy()
            if alteration_method == "distance" and current_adapt_size < 0.0:
                if isinstance(new_mask, BaseMask):
                    new_mask.roi.erode(
                        by_slice=by_slice,
                        max_eroded_volume_fraction=max_erosion,
                        distance=current_adapt_size
                    )
                elif isinstance(new_mask, MaskImage):
                    new_mask.erode(
                        by_slice=by_slice,
                        max_eroded_volume_fraction=max_erosion,
                        distance=current_adapt_size
                    )

            elif alteration_method == "distance" and current_adapt_size > 0.0:
                if isinstance(new_mask, BaseMask):
                    new_mask.roi.dilate(
                        by_slice=by_slice,
                        distance=current_adapt_size
                    )
                elif isinstance(new_mask, MaskImage):
                    new_mask.dilate(
                        by_slice=by_slice,
                        distance=current_adapt_size
                    )

            elif alteration_method == "fraction":
                if isinstance(new_mask, BaseMask):
                    new_mask.roi.fractional_volume_change(
                        by_slice=by_slice,
                        fractional_change=current_adapt_size
                    )
                elif isinstance(new_mask, MaskImage):
                    new_mask.fractional_volume_change(
                        by_slice=by_slice,
                        fractional_change=current_adapt_size
                    )

            if new_mask is not None:
                new_masks += [new_mask]

    new_masks = flatten_list(new_masks)
    if len(new_masks) == 0:
        return None

    if not return_list and len(new_masks) == 1:
        return new_masks[0]
    else:
        return new_masks


def add_noise(
        image: GenericImage,
        noise_level: Optional[float] = None,
        noise_estimation_method: str = "chang",
        repetitions: Optional[int] = None,
        repetition_id: Optional[int] = None
):
    if (repetitions is None and repetition_id is None) or repetitions == 0:
        return image

    if noise_level is None:
        noise_level = image.estimate_noise(method=noise_estimation_method)

    if noise_level is None:
        return image

    if repetition_id is not None:
        image.add_noise(noise_level=noise_level, noise_iteration_id=repetition_id)

    else:
        new_images = []
        for ii in range(repetitions):
            new_image = image.copy()
            new_image.add_noise(noise_level=noise_level, noise_iteration_id=ii)

            new_images += [new_image]

        return new_images


def discretise_image(
        image: GenericImage,
        mask: Optional[BaseMask],
        discretisation_method: Optional[str] = "none",
        intensity_range: Optional[Tuple[Any, Any]] = None,
        bin_width: Optional[int] = None,
        bin_number: Optional[int] = None,
        in_place: bool = False
):
    if image.is_empty():
        return None, None

    if not in_place:
        image = image.copy()
        mask = mask.copy()

    if mask is None:
        mask_data = np.ones(image.image_dimension, dtype=bool)
    elif mask.roi_intensity is None or mask.roi_intensity.is_empty() or mask.roi_intensity.is_empty_mask():
        return None, None
    else:
        mask_data = mask.roi_intensity.get_voxel_grid()
        intensity_range = mask.intensity_range

    if discretisation_method is None or discretisation_method == "none":
        levels = np.unique(image.get_voxel_grid()[mask_data])

        # Check if voxels are discretised.
        if not np.all(np.fmod(levels, 1.0) == 0.0):
            raise ValueError(f"The 'none' transformation method can only be used for data that resemble bins.")

        if not np.min(levels) >= 1.0:
            raise ValueError(
                f"The 'none' transformation method requires integer (i.e. 1.0, 2.0, etc.) values with a minimum value "
                f"of 1."
            )

        discretisation_method = "none"
        discretised_voxels = image.get_voxel_grid()
        bin_number = np.max(levels)

    elif discretisation_method == "fixed_bin_number":
        min_intensity = np.min(image.get_voxel_grid()[mask_data])
        max_intensity = np.max(image.get_voxel_grid()[mask_data])

        if min_intensity < max_intensity:
            discretised_voxels = np.floor(
                bin_number * 1.0 * (image.get_voxel_grid() - min_intensity) / (max_intensity - min_intensity)) + 1.0
        else:
            discretised_voxels = np.zeros(image.image_dimension, dtype=float) + np.ceil(bin_number / 2.0)

        discretised_voxels[discretised_voxels < 1.0] = 1.0
        discretised_voxels[discretised_voxels >= bin_number * 1.0] = bin_number * 1.0

        # Set number of bins.
        image.discretisation_bin_number = bin_number

    elif discretisation_method == "fixed_bin_size":
        if intensity_range is None or np.isnan(intensity_range[0]):
            min_intensity = image.get_default_lowest_intensity()

            if min_intensity is None:
                raise ValueError("Discretisation using the Fixed Bin Size method requires that the lower bound of the intensity range is set.")
            else:
                warnings.warn(
                    f"No lower bound of the intensity range was set for discretisation using the Fixed Bin "
                    f"Size method. A default value {min_intensity} was used."
                )
        else:
            min_intensity = intensity_range[0]

        # Bin voxels.
        discretised_voxels = np.floor((image.get_voxel_grid() - min_intensity) / (bin_width * 1.0)) + 1.0

        # Set voxels with grey level lower than 0.0 to 1.0. This may occur with non-roi voxels and voxels with the
        # minimum intensity.
        discretised_voxels[discretised_voxels <= 1.0] = 1.0

        # Determine number of bins
        bin_number = np.max(discretised_voxels[mask_data])

        # Limit to maximum number of bins
        discretised_voxels[discretised_voxels >= bin_number * 1.0] = bin_number * 1.0

        # Set bin width.
        image.discretisation_bin_width = bin_width

    elif discretisation_method == "fixed_bin_size_pyradiomics":
        # PyRadiomics (up to at least version 3.1.0) used a version of fixed bin size that is not IBSI-compliant.
        min_intensity = np.min(image.get_voxel_grid()[mask_data])
        min_intensity -= min_intensity % bin_width

        # Bin voxels.
        discretised_voxels = np.floor((image.get_voxel_grid() - min_intensity) / (bin_width * 1.0)) + 1.0

        # Set voxels with grey level lower than 0.0 to 1.0. This may occur with non-roi voxels and voxels with the
        # minimum intensity.
        discretised_voxels[discretised_voxels <= 1.0] = 1.0

        # Determine number of bins
        bin_number = np.max(discretised_voxels[mask_data])

        # Limit to maximum number of bins
        discretised_voxels[discretised_voxels >= bin_number * 1.0] = bin_number * 1.0

        # Set bin width.
        image.discretisation_bin_width = bin_width

    else:
        raise ValueError(f"The discretisation_method argument was not recognised. Found: {discretisation_method}")

    # Update voxel grid.
    image.set_voxel_grid(discretised_voxels)
    image.discretisation_method = discretisation_method
    mask.intensity_range = tuple([1.0, bin_number])

    return image, mask


def saturate_image(
        image: GenericImage,
        intensity_range: Optional[Tuple[Any, Any]],
        fill_value: Optional[Tuple[float, float]],
        in_place: bool = True
):
    if in_place:
        image = image.copy()

    # Saturate image
    image.saturate(intensity_range=intensity_range, fill_value=fill_value)

    return image


def normalise_image(
        image: GenericImage,
        normalisation_method: Optional[str] = None,
        intensity_range: Optional[Tuple[Any, Any]] = None,
        saturation_range: Optional[Tuple[Any, Any]] = None,
        mask: Optional[np.ndarray] = None,
        in_place: bool = True
):
    if intensity_range is None:
        intensity_range = [np.nan, np.nan]

    if saturation_range is None:
        saturation_range = [np.nan, np.nan]

    if in_place:
        image = image.copy()

    image.normalise_intensities(
        normalisation_method=normalisation_method,
        intensity_range=intensity_range,
        saturation_range=saturation_range,
        mask=mask
    )

    return image


def create_tissue_mask(
        image: GenericImage,
        mask_type: Optional[str] = None,
        mask_intensity_range: Optional[Tuple[Any, Any]] = None
) -> np.ndarray:

    if mask_type is None or mask_type == "none":
        # The entire image is the tissue mask.
        mask = np.ones(image.image_dimension, dtype=np.uint8)

    elif mask_type == "range":
        if mask_intensity_range is None:
            mask_intensity_range = [np.nan, np.nan]
        else:
            mask_intensity_range = list(mask_intensity_range)

        # The intensity range provided forms the mask range.
        if np.isnan(mask_intensity_range[0]):
            mask_intensity_range[1] = np.min(image.get_voxel_grid())
        if np.isnan(mask_intensity_range[1]):
            mask_intensity_range[2] = np.max(image.get_voxel_grid())

        voxel_grid = image.get_voxel_grid()
        mask = np.logical_and(voxel_grid >= mask_intensity_range[0], voxel_grid <= mask_intensity_range[1])

    elif mask_type == "relative_range":
        # The relative intensity range provided forms the mask range. This means that we need to convert the relative
        # range to the range present in the image.
        if mask_intensity_range is None:
            mask_intensity_range = [np.nan, np.nan]
        else:
            mask_intensity_range = list(mask_intensity_range)

        if np.isnan(mask_intensity_range[0]):
            mask_intensity_range[0] = 0.0
        if np.isnan(mask_intensity_range[1]):
            mask_intensity_range[1] = 1.0

        voxel_grid = image.get_voxel_grid()
        intensity_range = [np.min(voxel_grid), np.max(voxel_grid)]

        # Convert relative range to the image intensities
        tissue_range = [
            intensity_range[0] + mask_intensity_range[0] * (intensity_range[1] - intensity_range[0]),
            intensity_range[0] + mask_intensity_range[1] * (intensity_range[1] - intensity_range[0])
        ]

        mask = np.logical_and(voxel_grid >= tissue_range[0], voxel_grid <= tissue_range[1])
    else:
        raise ValueError(f"The tissue_mask_type configuration parameter is expected to be one of none, range, "
                         f"or relative_range. Encountered: {mask_type}")

    return mask


def saturate_image_deprecated(img_obj, intensity_range, fill_value):

    # Sature image
    img_obj.saturate(intensity_range=intensity_range, fill_value=fill_value)

    return img_obj


def normalise_image_deprecated(img_obj, norm_method, intensity_range=None, saturation_range=None, mask=None):

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


def resegmentise_deprecated(img_obj: ImageClass,
                            roi_list: List[RoiClass],
                            settings: SettingsClass):
    # Resegmentises segmentation map based on selected method

    if roi_list is not None:

        for ii in np.arange(0, len(roi_list)):

            # Generate intensity and morphology masks
            roi_list[ii].generate_masks()

            # Skip if no resegmentation method is used
            if "none" in settings.roi_resegment.resegmentation_method:
                continue

            # Resegment image
            roi_list[ii].resegmentise_mask(img_obj=img_obj,
                                           by_slice=settings.general.by_slice,
                                           method=settings.roi_resegment.resegmentation_method,
                                           settings=settings)

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


def select_largest_slice(roi_list):

    # Select the largest slice.
    for ii in np.arange(0, len(roi_list)):
        roi_list[ii].select_largest_slice()

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





def get_supervoxels_deprecated(
        img_obj: ImageClass,
        roi_obj: RoiClass,
        settings: SettingsClass):
    """Extracts supervoxels from an image"""

    from skimage.segmentation import slic
    import copy

    # Check if image and/or roi exist, and skip otherwise
    if img_obj.is_missing or roi_obj.roi is None:
        return None

    # Get image object grid
    img_voxel_grid = copy.deepcopy(img_obj.get_voxel_grid())

    # Get grey level thresholds
    g_range = settings.roi_resegment.intensity_range
    if np.isnan(g_range[0]):
        np.min(img_obj.get_voxel_grid()[roi_obj.roi.get_voxel_grid()])
    if np.isnan(g_range[1]):
        np.max(img_obj.get_voxel_grid()[roi_obj.roi.get_voxel_grid()])

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
        img_voxel_grid = img_voxel_grid.astype(float)

    # Create a slic segmentation of the image stack
    img_segments = slic(
        image=img_voxel_grid,
        n_segments=n_segments,
        sigma=sigma,
        spacing=img_obj.spacing,
        compactness=0.05,
        convert2lab=False,
        enforce_connectivity=True,
        channel_axis=None)
    img_segments += 1

    # Release img_voxel_grid
    del img_voxel_grid

    return img_segments


def get_supervoxel_overlap_deprecated(roi_obj, img_segments, mask=None):
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
    overlap_size = overlap_size[overlap_segment_labels > 0]
    overlap_segment_labels = overlap_segment_labels[overlap_segment_labels > 0]

    # Check the actual size of the segments overlapping with the current contour
    segment_size = list(map(lambda x: np.sum([img_segments == x]), overlap_segment_labels))

    # Calculate the fraction of overlap
    overlap_frac = overlap_size / segment_size

    return overlap_segment_labels, overlap_frac, overlap_size


def transform_images_default(
        img_obj: ImageClass,
        roi_list: List[RoiClass],
        settings: SettingsClass,
        compute_features: bool = False,
        extract_images: bool = False,
        file_path: Union[None, str] = None
):
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
    feature_list = []
    response_map_list = []

    # Check if image transformation is required
    if settings.img_transform.spatial_filters is None:
        return feature_list, response_map_list

    # Get spatial filters to apply
    spatial_filter = settings.img_transform.spatial_filters

    # Iterate over spatial filters
    for curr_filter in spatial_filter:

        filter_obj = None

        if settings.img_transform.has_separable_wavelet_filter(x=curr_filter):
            # Separable wavelet filters
            from mirp.imageFilters.separableWaveletFilter import SeparableWaveletFilter
            filter_obj = SeparableWaveletFilter(settings=settings, name=curr_filter)

        elif settings.img_transform.has_nonseparable_wavelet_filter(x=curr_filter):
            # Non-separable wavelet filters
            from mirp.imageFilters.nonseparableWaveletFilter import NonseparableWaveletFilter
            filter_obj = NonseparableWaveletFilter(settings=settings, name=curr_filter)

        elif settings.img_transform.has_gaussian_filter(x=curr_filter):
            # Gaussian filters
            from mirp.imageFilters.gaussian import GaussianFilter
            filter_obj = GaussianFilter(settings=settings, name=curr_filter)

        elif settings.img_transform.has_laplacian_of_gaussian_filter(x=curr_filter):
            # Laplacian of Gaussian filters
            from mirp.imageFilters.laplacianOfGaussian import LaplacianOfGaussianFilter
            filter_obj = LaplacianOfGaussianFilter(settings=settings, name=curr_filter)

        elif settings.img_transform.has_laws_filter(x=curr_filter):
            # Laws' kernels
            from mirp.imageFilters.lawsFilter import LawsFilter
            filter_obj = LawsFilter(settings=settings, name=curr_filter)

        elif settings.img_transform.has_gabor_filter(x=curr_filter):
            # Gabor kernels
            from mirp.imageFilters.gaborFilter import GaborFilter
            filter_obj = GaborFilter(settings=settings, name=curr_filter)

        elif settings.img_transform.has_mean_filter(x=curr_filter):
            # Mean / uniform filter
            from mirp.imageFilters.meanFilter import MeanFilter
            filter_obj = MeanFilter(settings=settings, name=curr_filter)

        else:
            raise ValueError(
                f"{curr_filter} is not implemented as a spatial filter. Please use one of ",
                ", ".join(settings.img_transform.get_available_image_filters())
            )

        current_feature_list, current_response_map_list = filter_obj.apply_transformation(
            img_obj=img_obj,
            roi_list=roi_list,
            settings=settings,
            compute_features=compute_features,
            extract_images=extract_images,
            file_path=file_path)

        feature_list += current_feature_list
        response_map_list += current_response_map_list

    return feature_list, response_map_list


def crop_image_deprecated(img_obj, roi_list=None, roi_obj=None, boundary=0.0, z_only=False):
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
        boundary = np.ceil(boundary / img_obj.spacing).astype(int)

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


def crop_image_to_size_deprecated(img_obj, crop_size, roi_list=None, roi_obj=None):

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


def discretise_image_intensities(img_obj: ImageClass,
                                 roi_obj: RoiClass,
                                 discr_method: str = "none",
                                 bin_width: Union[None, int] = None,
                                 bin_number: Union[None, int] = None):

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


def gaussian_preprocess_filter(orig_vox, orig_spacing, sample_spacing=None, param_beta=0.93, mode="nearest", by_slice=False):

    import scipy.ndimage

    # If no sample spacing is provided, assume original spacing. Note that for most purposes sample spacing should be provided
    if sample_spacing is None:
        sample_spacing = orig_spacing

    # Set sample spacing and orig_spacing to float
    sample_spacing = sample_spacing.astype(float)
    orig_spacing = orig_spacing.astype(float)

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


def divide_tumour_regions_deprecated(roi_list: List[RoiClass], settings: SettingsClass):

    # Create new list for storing roi boundaries and bulk
    new_roi_list = []

    # Get the boundary size.
    boundary_size_list = settings.perturbation.roi_boundary_size

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
            bulk_roi_obj.erode(by_slice=settings.general.by_slice,
                               dist=-boundary_size,
                               eroded_vol_fract=settings.perturbation.max_bulk_volume_erosion)

            # Get roi boundary if roi exists
            if roi_obj.roi is not None:
                boundary_roi_obj.roi.set_voxel_grid(voxel_grid=np.logical_xor(roi_obj.roi.get_voxel_grid(),
                                                                              bulk_roi_obj.roi.get_voxel_grid()))

            # Check whether the bulk and boundary roi object are empty or not
            if not bulk_roi_obj.is_empty() and not boundary_roi_obj.is_empty():

                # Store bulk and boundary rois to list when both are not empty
                new_roi_list += [bulk_roi_obj, boundary_roi_obj]

    # Return rois
    return new_roi_list


def calculate_features(img_obj: ImageClass,
                       roi_list: List[RoiClass],
                       settings: Union[SettingsClass, FeatureExtractionSettingsClass],
                       append_str: Union[None, str] = ""):
    """
    Calculate image features from the provided data
    :param img_obj:
    :param roi_list:
    :param settings:
    :param append_str:
    :return:
    """

    from mirp.featureSets.localIntensity import get_local_intensity_features_deprecated
    from mirp.featureSets.statistics import get_intensity_statistics_features_deprecated
    from mirp.featureSets.intensityVolumeHistogram import get_intensity_volume_histogram_features_deprecated
    from mirp.featureSets.volumeMorphology import get_volumetric_morphological_features_deprecated

    feat_list = []

    # Update settings.
    if isinstance(settings, SettingsClass):
        settings = settings.feature_extr

    # Skip if no feature families are specified.
    if not settings.has_any_feature_family():
        return None

    for roi_obj in roi_list:

        roi_feat_list = []

        ################################################################################################################
        # Local mapping features
        ################################################################################################################

        if settings.has_local_intensity_family():
            # Cut roi and image with 10 mm boundary
            img_cut, roi_cut = crop_image_deprecated(img_obj=img_obj,
                                                     roi_obj=roi_obj,
                                                     boundary=10.0)

            # Decode roi voxel grid
            roi_cut.decode_voxel_grid()

            # Calculate local intensities
            roi_feat_list += [get_local_intensity_features_deprecated(img_obj=img_cut,
                                                                      roi_obj=roi_cut)]

            # Clean up
            del img_cut, roi_cut

        ################################################################################################################
        # ROI features without discretisation
        ################################################################################################################

        # Cut roi and image to image
        img_cut, roi_cut = crop_image_deprecated(img_obj=img_obj,
                                                 roi_obj=roi_obj,
                                                 boundary=0.0)

        # Decode roi voxel grid
        roi_cut.decode_voxel_grid()

        # Extract statistical features
        if settings.has_stats_family():
            roi_feat_list += [get_intensity_statistics_features_deprecated(img_obj=img_cut,
                                                                           roi_obj=roi_cut)]

        # Calculate intensity volume histogram features
        if settings.has_ivh_family():
            roi_feat_list += [get_intensity_volume_histogram_features_deprecated(img_obj=img_cut,
                                                                                 roi_obj=roi_cut,
                                                                                 settings=settings)]

        # Calculate morphological features
        if settings.has_morphology_family():
            roi_feat_list += [get_volumetric_morphological_features_deprecated(img_obj=img_cut,
                                                                               roi_obj=roi_cut,
                                                                               settings=settings)]

        ################################################################################################################
        # ROI features with discretisation
        ################################################################################################################

        if settings.has_discretised_family():
            for discretisation_method in settings.discretisation_method:

                if discretisation_method in ["fixed_bin_size"]:
                    for bin_width in settings.discretisation_bin_width:
                        roi_feat_list += [compute_discretised_features_deprecated(img_obj=img_cut,
                                                                                  roi_obj=roi_cut,
                                                                                  settings=settings,
                                                                                  discretisation_method=discretisation_method,
                                                                                  bin_width=bin_width,
                                                                                  bin_number=None)]
                if discretisation_method in ["fixed_bin_number"]:
                    for bin_number in settings.discretisation_n_bins:
                        roi_feat_list += [compute_discretised_features_deprecated(img_obj=img_cut,
                                                                                  roi_obj=roi_cut,
                                                                                  settings=settings,
                                                                                  discretisation_method=discretisation_method,
                                                                                  bin_width=None,
                                                                                  bin_number=bin_number)]
                if discretisation_method in ["none"]:
                    roi_feat_list += [compute_discretised_features_deprecated(img_obj=img_cut, roi_obj=roi_cut,
                                                                              settings=settings,
                                                                              discretisation_method=discretisation_method,
                                                                              bin_width=None,
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


def compute_discretised_features_deprecated(img_obj: ImageClass,
                                            roi_obj: RoiClass,
                                            settings: FeatureExtractionSettingsClass,
                                            discretisation_method: str = "none",
                                            bin_width: Union[None, int] = None,
                                            bin_number: Union[None, int] = None):
    """Function to process and calculate discretised image features"""

    from mirp.featureSets.intensityHistogram import get_intensity_histogram_features_deprecated
    from mirp.featureSets.cooccurrenceMatrix import get_cm_features_deprecated
    from mirp.featureSets.runLengthMatrix import get_rlm_features_deprecated
    from mirp.featureSets.sizeZoneMatrix import get_szm_features_deprecated
    from mirp.featureSets.distanceZoneMatrix import get_dzm_features_deprecated
    from mirp.featureSets.neighbourhoodGreyToneDifferenceMatrix import get_ngtdm_features_deprecated
    from mirp.featureSets.neighbouringGreyLevelDifferenceMatrix import get_ngldm_features_deprecated

    # Apply image discretisation
    img_discr, roi_discr = discretise_image_intensities(img_obj=img_obj,
                                                        roi_obj=roi_obj,
                                                        discr_method=discretisation_method,
                                                        bin_width=bin_width,
                                                        bin_number=bin_number)

    # Decode roi object
    roi_discr.decode_voxel_grid()

    # Initiate empty feature list
    feat_list = []

    # Intensity histogram
    if settings.has_ih_family():
        feat_list += [get_intensity_histogram_features_deprecated(img_obj=img_discr,
                                                                  roi_obj=roi_discr)]

    # Grey level cooccurrence matrix
    if settings.has_glcm_family():
        feat_list += [get_cm_features_deprecated(img_obj=img_discr,
                                                 roi_obj=roi_discr,
                                                 settings=settings)]

    # Grey level run length matrix
    if settings.has_glrlm_family():
        feat_list += [get_rlm_features_deprecated(img_obj=img_discr,
                                                  roi_obj=roi_discr,
                                                  settings=settings)]

    # Grey level size zone matrix
    if settings.has_glszm_family():
        feat_list += [get_szm_features_deprecated(img_obj=img_discr,
                                                  roi_obj=roi_discr,
                                                  settings=settings)]

    # Grey level distance zone matrix
    if settings.has_gldzm_family():
        feat_list += [get_dzm_features_deprecated(img_obj=img_discr,
                                                  roi_obj=roi_discr,
                                                  settings=settings)]

    # Neighbourhood grey tone difference matrix
    if settings.has_ngtdm_family():
        feat_list += [get_ngtdm_features_deprecated(img_obj=img_discr,
                                                    roi_obj=roi_discr,
                                                    settings=settings)]

    # Neighbouring grey level dependence matrix
    if settings.has_ngldm_family():
        feat_list += [get_ngldm_features_deprecated(img_obj=img_discr,
                                                    roi_obj=roi_discr,
                                                    settings=settings)]

    # Check if any features were added to the feature list; otherwise return to main function
    if len(feat_list) == 0:
        return None

    # Concatenate list of feature tables
    df_feat = pd.concat(feat_list, axis=1)

    # Parse name
    parse_str = ""

    # Add discretisation method to string
    if discretisation_method == "fixed_bin_size":
        parse_str += "_fbs"
    if discretisation_method == "fixed_bin_number":
        parse_str += "_fbn"

    # Add bin witdth/ bin number to string
    if bin_width is not None:
        parse_str += "_w" + str(bin_width)
    if bin_number is not None:
        parse_str += "_n" + str(int(bin_number))

    df_feat.columns += parse_str

    return df_feat


def create_tissue_mask_deprecated(img_obj: ImageClass, settings: SettingsClass):

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


def bias_field_correction_deprecated(img_obj: ImageClass, settings: SettingsClass, mask=None):
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
