import numpy as np

from mirp._image_processing.utilities import standard_image_process_checks
from mirp._images.genericImage import GenericImage
from mirp._images.maskImage import MaskImage
from mirp._masks.baseMask import BaseMask


def crop(
        image: GenericImage,
        masks: BaseMask | MaskImage | list[BaseMask],
        boundary: float = 0.0,
        xy_only: bool = False,
        z_only: bool = False,
        in_place: bool = False,
        by_slice: bool = False
) -> tuple[GenericImage, None | BaseMask | MaskImage | list[BaseMask]]:
    """ The function is used to slice a subsection of the image so that further processing is facilitated in terms of
     memory and computational requirements. """

    image, masks, return_list = standard_image_process_checks(image=image, masks=masks)
    if return_list is None:
        return image, None

    bounds_z: None | list[int] = None
    bounds_y: None | list[int] = None
    bounds_x: None | list[int] = None

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

    # Crop _images and _masks.
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
        masks: BaseMask | MaskImage | list[BaseMask],
        crop_size: list[float],
        crop_center: None | list[float] = None,
        in_place: bool = False
) -> tuple[GenericImage, None | BaseMask | MaskImage | list[BaseMask]]:

    image, masks, return_list = standard_image_process_checks(image=image, masks=masks)
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
