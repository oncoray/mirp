from typing import Optional, Union, List, Tuple, Any

from mirp.imageProcess.utilities import standard_image_process_checks
from mirp.images.genericImage import GenericImage
from mirp.masks.baseMask import BaseMask


def resegmentise_mask(
        image: GenericImage,
        masks: Optional[Union[BaseMask, List[BaseMask]]],
        resegmentation_method: Optional[Union[str, List[str]]] = None,
        intensity_range: Optional[Tuple[Any, Any]] = None,
        sigma: Optional[float] = None
):
    # Resegmentises mask based on the selected method.
    image, masks, return_list = standard_image_process_checks(image, masks)
    if return_list is None:
        return masks

    masks: List[BaseMask] = masks

    for mask in masks:
        mask.resegmentise_mask(
            image=image,
            intensity_range=intensity_range,
            sigma=sigma
        )

    if return_list:
        return masks
    else:
        return masks[0]
