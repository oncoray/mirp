from typing import Any

from mirp._image_processing.utilities import standard_image_process_checks
from mirp._images.genericImage import GenericImage
from mirp._masks.baseMask import BaseMask


def resegmentise_mask(
        image: GenericImage,
        masks: None | BaseMask | list[BaseMask],
        resegmentation_method: None | str | list[str] = None,
        intensity_range: None | tuple[Any, Any] = None,
        sigma: None | float = None
):
    # Resegmentises mask based on the selected method.
    image, masks, return_list = standard_image_process_checks(image, masks)
    if return_list is None:
        return masks

    masks: list[BaseMask] = masks

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
