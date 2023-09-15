from typing import Optional, Tuple, Any

from mirp.images.genericImage import GenericImage


def saturate_image(
        image: GenericImage,
        intensity_range: Optional[Tuple[Any, Any]],
        fill_value: Optional[Tuple[float, float]],
        in_place: bool = True
):
    if in_place:
        image = image.copy()

    # Saturate image
    image.saturate(intensity_range=intensity_range, fill_value=fill_value)

    return image
