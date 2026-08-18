from keyword import kwlist

import numpy as np
from typing import Generator

from mirp._data_import.generic_file import ImageFile
from mirp._data_import.utilities import flatten_list
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask


def read_image(
        image: ImageFile | GenericImage,
        to_numpy=False,
        **kwargs
) -> list[np.ndarray] | list[GenericImage]:
    image_out = [x.promote() for x in image.to_object(**kwargs)]

    if to_numpy:
        image_out = [x.get_voxel_grid() for x in image_out]

    return image_out


def read_image_and_masks(
        image: ImageFile | GenericImage,
        to_numpy=False,
        **kwargs
) -> tuple[list[np.ndarray] | list[GenericImage], list[np.ndarray] | list[BaseMask]]:
    # Read image from file.
    image_out = [x.promote() for x in image.to_object(**kwargs)]

    mask_list = []
    if image.associated_masks is not None:
        mask_list = image.associated_masks

    # Read masks from file.
    if mask_list is not None:
        mask_list = [list(mask.to_object(image=image_out[0], **kwargs)) for mask in mask_list]
        mask_list = flatten_list(mask_list)

    # Remove None entries.
    mask_list = [mask for mask in mask_list if mask is not None]

    if to_numpy:
        image_out = [x.get_voxel_grid() for x in image_out]
        mask_list = [mask.roi.get_voxel_grid() for mask in mask_list]

    return image_out, mask_list


def read_image_and_masks_generator(
        image: ImageFile | GenericImage,
        to_numpy: bool = False,
        **kwargs
) -> Generator[tuple[np.ndarray | GenericImage, list[np.ndarray] | list[BaseMask]], None, None]:

    for image_out in image.to_object(**kwargs):
        image_out = image_out.promote()

        mask_list = []
        if image.associated_masks is not None:
            mask_list = image.associated_masks

        # Read masks from file.
        if mask_list is not None:
            mask_list = [list(mask.to_object(image=image_out, **kwargs)) for mask in mask_list]
            mask_list = flatten_list(mask_list)

        # Remove None entries.
        mask_list = [mask for mask in mask_list if mask is not None]

        if to_numpy:
            image_out = image_out.get_voxel_grid()
            mask_list = [mask.roi.get_voxel_grid() for mask in mask_list]

        yield image_out, mask_list
