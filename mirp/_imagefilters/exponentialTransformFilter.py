import copy
import numpy as np

from mirp._images.genericImage import GenericImage
from mirp._images.transformedImage import ExponentialTransformedImage
from mirp._imagefilters.genericFilter import GenericFilter
from mirp.settings.settingsGeneric import SettingsClass


class ExponentialTransformFilter(GenericFilter):

    def __init__(self, settings: SettingsClass, name: str):

        super().__init__(
            settings=settings,
            name=name
        )

    def generate_object(self):
        yield copy.deepcopy(self)

    def transform(self, image: GenericImage) -> ExponentialTransformedImage:
        # Create placeholder response map.
        response_map = ExponentialTransformedImage(
            image_data=None,
            template=image
        )

        if image.is_empty():
            return response_map

        image_data = image.get_voxel_grid()
        max_value = np.max(np.abs(image_data))

        # Prevent issues with alpha values that are not strictly positive.
        if not np.isfinite(max_value) or max_value == 0.0:
            max_value = 1.0

        alpha = np.log(max_value) / max_value

        # Prevent issues with alpha values that are not strictly positive.
        if not np.isfinite(alpha) or alpha == 0.0:
            alpha = 1.0

        response_map.set_voxel_grid(
            voxel_grid=np.exp(image_data * alpha)
        )

        return response_map
