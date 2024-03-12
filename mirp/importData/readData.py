import numpy as np

from mirp.importData.imageGenericFile import ImageFile
from mirp.importData.utilities import flatten_list
from mirp.images.genericImage import GenericImage
from mirp.masks.baseMask import BaseMask


def read_image(
        image: ImageFile,
        to_numpy=False
) -> np.ndarray | GenericImage:
    image = image.to_object().promote()

    if to_numpy:
        image = image.get_voxel_grid()

    return image


def read_image_and_masks(
        image: ImageFile,
        to_numpy=False
) -> tuple[np.ndarray | GenericImage, list[np.ndarray] | list[BaseMask]]:
    mask_list = []
    if image.associated_masks is not None:
        mask_list = image.associated_masks

    # Read masks from file.
    if mask_list is not None:
        mask_list = [mask.to_object(image=image) for mask in mask_list]
        mask_list = flatten_list(mask_list)

    # Remove None entries.
    mask_list = [mask for mask in mask_list if mask is not None]

    # Read image from file.
    image = image.to_object().promote()

    if to_numpy:
        image = image.get_voxel_grid()
        mask_list = [mask.roi.get_voxel_grid() for mask in mask_list]

    return image, mask_list
