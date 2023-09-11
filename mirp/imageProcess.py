import warnings
from typing import Union, List, Tuple, Optional, Any

from mirp.images.genericImage import GenericImage
from mirp.images.maskImage import MaskImage
from mirp.masks.baseMask import BaseMask
from mirp.importData.utilities import flatten_list

import numpy as np


def _standard_checks(
        image: GenericImage,
        masks: Optional[Union[BaseMask, MaskImage, List[BaseMask]]]
) -> Tuple[GenericImage, Optional[Union[List[BaseMask], List[MaskImage]]], Optional[bool]]:
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

    # Add 10% range outside the grey level range
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
        in_place: bool = False
) -> Tuple[GenericImage, Union[None, BaseMask, MaskImage, List[BaseMask]]]:

    image, masks, return_list = _standard_checks(image=image, masks=masks)
    if return_list is None:
        return image, None

    # Set crop_center
    if crop_center is None:
        crop_centers = [mask.get_center_position() for mask in masks]
        if len(crop_centers) == 1:
            crop_center = crop_centers[0]
        else:
            crop_center = np.mean(np.stack(crop_centers, axis=1), axis=1)

    # Convert to list
    crop_center = list(crop_center)
    crop_center = [x if not np.isnan(x) else None for x in crop_center]

    if not in_place:
        image = image.copy()
        masks = [mask.copy() for mask in masks]

    image.crop_to_size(center=crop_center, crop_size=crop_size)
    for mask in masks:
        mask.crop_to_size(center=crop_center, crop_size=crop_size)

    if return_list:
        return image, masks
    else:
        return image, masks[0]


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
                raise ValueError(
                    "Discretisation using the Fixed Bin Size method requires that the lower bound of the "
                    "intensity range is set."
                )
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
        mask_intensity_range: Optional[Tuple[float, ...]] = None
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
            mask_intensity_range[0] = np.min(image.get_voxel_grid())
        if np.isnan(mask_intensity_range[1]):
            mask_intensity_range[1] = np.max(image.get_voxel_grid())

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


def gaussian_preprocess_filter(
        orig_vox,
        orig_spacing,
        sample_spacing=None,
        param_beta=0.98,
        mode="nearest",
        by_slice=False
):

    from scipy.ndimage import gaussian_filter

    # If no sample spacing is provided, assume original spacing. Note that for most purposes sample
    # spacing should be provided.
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
    if by_slice:
        map_spacing[0] = 0.0

    # Calculate sigma
    sigma = np.sqrt(-8 * np.power(map_spacing, 2.0) * np.log(param_beta))

    # Apply filter
    new_vox = gaussian_filter(
        input=orig_vox.astype(np.float32),
        sigma=sigma,
        order=0,
        mode=mode
    )

    return new_vox
