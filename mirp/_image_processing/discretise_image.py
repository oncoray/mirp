import warnings
from typing import Any

import numpy as np

from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask


def discretise_image(
        image: GenericImage,
        mask: None | BaseMask,
        discretisation_method: None | str = "none",
        intensity_range: None | tuple[Any, Any] = None,
        bin_width: None | int = None,
        bin_number: None | int = None,
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
