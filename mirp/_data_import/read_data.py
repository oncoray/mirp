import numpy as np

from mirp._data_import.generic_file import ImageFile
from mirp._data_import.utilities import flatten_list
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask


def read_image(
        image: ImageFile | GenericImage,
        to_numpy=False,
        **kwargs
) -> np.ndarray | GenericImage:
    image = image.to_object(**kwargs).promote()

    if to_numpy:
        image = image.get_voxel_grid()

    return image


def read_image_and_masks(
        image: ImageFile | GenericImage,
        to_numpy=False,
        **kwargs
) -> tuple[np.ndarray | GenericImage, list[np.ndarray] | list[BaseMask]]:
    # Read image from file.
    image_out = image.to_object(**kwargs).promote()

    mask_list = []
    if image.associated_masks is not None:
        mask_list = image.associated_masks

    # Read masks from file.
    if mask_list is not None:
        mask_list = [mask.to_object(image=image, **kwargs) for mask in mask_list]
        mask_list = flatten_list(mask_list)

    # Remove None entries.
    mask_list = [mask for mask in mask_list if mask is not None]

    if to_numpy:
        image_out = image_out.get_voxel_grid()
        mask_list = [mask.roi.get_voxel_grid() for mask in mask_list]

    return image_out, mask_list
