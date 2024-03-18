from typing import Any

import numpy as np

from mirp._images.generic_image import GenericImage
from mirp._images.mask_image import MaskImage
from mirp._masks.baseMask import BaseMask


def standard_image_process_checks(
        image: GenericImage,
        masks: None | BaseMask | MaskImage | list[BaseMask]
) -> tuple[GenericImage, None | list[BaseMask] | list[MaskImage], None | bool]:
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
        mask: None | MaskImage = None,
        intensity_range: tuple[Any, Any] | None = None
) -> tuple[float, float]:
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
        intensity_range: tuple[Any, Any],
        extend_fraction=0.1
) -> None | tuple[Any, Any]:
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
