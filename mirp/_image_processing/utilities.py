from typing import Any, Generator

import numpy as np

from mirp._images.generic_image import GenericImage
from mirp._images.mask_image import MaskImage
from mirp._masks.base_mask import BaseMask


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
) -> tuple[float, ...]:
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
) -> None | tuple[float, ...]:
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


def _coord_to_index(z, y, x, dims: tuple[int, ...]):
    # Translate coordinates to indices
    index = x + y * dims[2] + z * dims[2] * dims[1]

    # Mark invalid transitions
    index[np.logical_or(x < 0, x >= dims[2])] = -99999
    index[np.logical_or(y < 0, y >= dims[1])] = -99999
    index[np.logical_or(z < 0, z >= dims[0])] = -99999

    return index


def _index_to_coord(index, dims: tuple[int, ...]):
    z = index // (dims[2] * dims[1])
    index -= z * (dims[2] * dims[1])
    y = index // (dims[2])
    x = index - y * dims[2]

    return z, y, x


def lookup_neighbour_voxel_value(
        voxels: np.ndarray,
        dims: tuple[int, ...],
        lookup_vector: tuple[int, ...]
):
    # voxels are a flat np.ndarray.
    z, y, x = _index_to_coord(index=np.arange(len(voxels)), dims=dims)
    neighbour_index = _coord_to_index(
        z = z + lookup_vector[0],
        y = y + lookup_vector[1],
        x = x + lookup_vector[2],
        dims = dims
    )

    mask = neighbour_index > 0
    return mask, voxels[neighbour_index[mask]]


def generate_neighbour_direction(
        d: float = 1.8,
        spacing: None | tuple[float, ...] = None,
        metric: str = "euclidian",
        keep_centre: bool = False,
        complete: bool = False,
        dim3: bool = True
) -> Generator[tuple[int, ...], None, None]:
    from mirp._features.utilities import rep

    if spacing is None:
        spacing = tuple([1.0, 1.0, 1.0])

    # Convert to numpy array.
    spacing = np.array(spacing)

    # Set footprint size (in voxel units).
    footprint_size = int(np.ceil(np.max(d / spacing)))

    # Base transition vector
    trans = np.arange(-footprint_size, footprint_size + 1)
    n = np.size(trans)

    # Build transition array [z,y,x]
    nbrs = np.array([
        rep(x=trans, each=1, times=n * n),
        rep(x=trans, each=n, times=n),
        rep(x=trans, each=n * n, times=1)
    ], dtype=int)

    # Initiate maintenance index
    index = np.zeros(nbrs.shape[1], dtype=bool)

    # Remove neighbours more than distance d from the center.
    if metric.lower() in ["manhattan", "l1", "l_1"]:
        # Manhattan distance
        distance = np.sum(np.multiply(np.abs(nbrs), np.expand_dims(spacing, axis=1)), axis=0)
    elif metric.lower() in ["euclidian", "l2", "l_2"]:
        # Euclidian distance
        distance = np.sqrt(np.sum(np.power(np.multiply(nbrs, np.expand_dims(spacing, axis=1)), 2.0), axis=0))
    elif metric in ["chebyshev", "linf", "l_inf"]:
        # Chebyshev distance
        distance = np.max(np.multiply(np.abs(nbrs), np.expand_dims(spacing, axis=1)), axis=0)
    else:
        raise ValueError(f"Did not recognize distance metric: {metric}")

    index = np.logical_or(index, distance <= d)

    # Check if centre voxel [0,0,0] should be maintained; False indicates removal
    if not keep_centre:
        index = np.logical_and(index, distance > 0.0)

    # Check if a complete neighbourhood should be returned
    # False indicates that only half of the vectors are returned
    if not complete:
        index[np.arange(0, stop=len(index) // 2 + 1)] = False

    # Check if neighbourhood should be 3D or 2D
    if not dim3:
        index[nbrs[0, :] != 0] = False

    for ii, flag in enumerate(index):
        if flag:
            yield tuple(nbrs[:, ii].flatten())