from mirp._image_processing.utilities import standard_image_process_checks
from mirp._images.generic_image import GenericImage
from mirp._images.mask_image import MaskImage
from mirp._data_import.utilities import flatten_list
from mirp._masks.baseMask import BaseMask


def randomise_mask(
        image: GenericImage,
        masks: BaseMask | MaskImage | list[BaseMask],
        boundary: float = 25.0,
        repetitions: int = 1,
        by_slice: bool = False
):
    image, masks, return_list = standard_image_process_checks(image=image, masks=masks)
    if return_list is None:
        return None

    new_masks = []
    for mask in masks:
        if isinstance(mask, MaskImage):
            randomised_masks = mask.randomise_mask(
                image=image,
                boundary=boundary,
                repetitions=repetitions,
                by_slice=by_slice
            )

            for randomised_mask in randomised_masks:
                if randomised_mask is None:
                    continue
                new_masks += [randomised_mask]

        elif isinstance(mask, BaseMask):
            randomised_masks = mask.roi.randomise_mask(
                image=image,
                boundary=boundary,
                repetitions=repetitions,
                intensity_range=mask.intensity_range,
                by_slice=by_slice
            )

            for randomised_mask in randomised_masks:
                if randomised_mask is None:
                    continue
                new_mask = mask.copy(drop_image=True)
                new_mask.roi = randomised_mask
                new_masks += [new_mask]
        else:
            raise TypeError("The _masks attribute is expected to be MaskImage and BaseMask")

    new_masks = flatten_list(new_masks)
    if len(new_masks) == 0:
        return None

    if not return_list and repetitions == 1:
        return new_masks[0]
    else:
        return new_masks
