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


def generate_neighbour_direction(self) -> Generator[tuple[int, ...], None, None]:
    from mirp._features.utilities import rep

    if self.separate_slices:
        m = 8
        nbrs = np.array([
            np.zeros(m, int),
            np.round(self.d * np.sin(2 * np.pi * np.arange(m, dtype=float) / m)),
            np.round(self.d * np.cos(2 * np.pi * np.arange(m, dtype=float) / m))
        ], dtype = int)

        # Remove duplicates
        _, indices = np.unique(nbrs, return_index=True, axis=1)
        nbrs = nbrs[:, indices.sort()].squeeze()

        # Compute distance to eliminate
        neighbour_distance = np.sqrt(np.sum(np.multiply(nbrs, nbrs), axis=0))
        index = neighbour_distance > 0.0

        for ii, flag in enumerate(index):
            if flag:
                yield tuple(nbrs[:, ii].flatten())

    else:
        # Base transition vector
        trans = np.arange(start=-np.ceil(self.d + 1.0), stop=np.ceil(self.d + 1.0) + 1)
        n = np.size(trans)

        # Build transition array [z,y,x]
        nbrs = np.array([
            rep(x=trans, each=n * n, times=1),
            rep(x=trans, each=n, times=n, use_inversion=True),
            rep(x=trans, each=1, times=n * n, use_inversion=True)
        ], dtype=int)

        # Filter neighbours based on distance. That is, all voxels that fall within distance d and d-1.0 (a single
        # rim of voxels), and excluding the central voxel.
        neighbour_distance = np.sqrt(np.sum(np.multiply(nbrs, nbrs), axis = 0))
        index = np.logical_and(neighbour_distance <= self.d, neighbour_distance > self.d - 1.0)
        index = np.logical_and(index, neighbour_distance > 0.0)

        for ii, flag in enumerate(index):
            if flag:
                yield tuple(nbrs[:, ii].flatten())