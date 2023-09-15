from typing import Optional, Tuple

import numpy as np

from mirp.images.genericImage import GenericImage


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
