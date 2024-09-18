from mirp._images.mask_image import MaskImage
from mirp._data_import.utilities import flatten_list
from mirp._masks.base_mask import BaseMask


def alter_mask(
        masks: BaseMask | MaskImage | list[BaseMask],
        alteration_size: None | list[float] = None,
        alteration_method: None | str = None,
        max_erosion: None | float = 0.8
):
    """ Adapt roi size by growing or shrinking the roi """

    if alteration_size is None or alteration_method is None:
        return masks

    # Determine the return format.
    return_list = False
    if isinstance(masks, list):
        return_list = True
    else:
        masks = [masks]

    new_masks = []

    for mask in masks:
        for current_adapt_size in alteration_size:
            new_mask = mask.copy()
            if alteration_method == "distance" and current_adapt_size < 0.0:
                if isinstance(new_mask, BaseMask):
                    new_mask.roi.erode(
                        max_eroded_volume_fraction=max_erosion,
                        distance=current_adapt_size
                    )
                elif isinstance(new_mask, MaskImage):
                    new_mask.erode(
                        max_eroded_volume_fraction=max_erosion,
                        distance=current_adapt_size
                    )

            elif alteration_method == "distance" and current_adapt_size > 0.0:
                if isinstance(new_mask, BaseMask):
                    new_mask.roi.dilate(
                        distance=current_adapt_size
                    )
                elif isinstance(new_mask, MaskImage):
                    new_mask.dilate(
                        distance=current_adapt_size
                    )

            elif alteration_method == "fraction":
                if isinstance(new_mask, BaseMask):
                    new_mask.roi.fractional_volume_change(
                        fractional_change=current_adapt_size
                    )
                elif isinstance(new_mask, MaskImage):
                    new_mask.fractional_volume_change(
                        fractional_change=current_adapt_size
                    )

            if new_mask is not None:
                new_masks += [new_mask]

    new_masks = flatten_list(new_masks)
    if len(new_masks) == 0:
        return None

    if not return_list and len(new_masks) == 1:
        return new_masks[0]
    else:
        return new_masks
