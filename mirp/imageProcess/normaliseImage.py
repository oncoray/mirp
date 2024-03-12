from typing import Any
import numpy as np

from mirp.images.genericImage import GenericImage


def normalise_image(
        image: GenericImage,
        normalisation_method: None | str = None,
        intensity_range: None | tuple[Any, Any] = None,
        saturation_range: None | tuple[Any, Any] = None,
        mask: None | np.ndarray = None,
        in_place: bool = True
):
    if intensity_range is None:
        intensity_range = [np.nan, np.nan]

    if saturation_range is None:
        saturation_range = [np.nan, np.nan]

    if in_place:
        image = image.copy()

    image.normalise_intensities(
        normalisation_method=normalisation_method,
        intensity_range=intensity_range,
        saturation_range=saturation_range,
        mask=mask
    )

    return image
