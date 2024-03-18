import copy
import numpy as np

from mirp._images.genericImage import GenericImage
from mirp._images.transformedImage import LogarithmTransformedImage
from mirp._imagefilters.genericFilter import GenericFilter
from mirp.settings.settingsGeneric import SettingsClass


class LogarithmTransformFilter(GenericFilter):

    def __init__(self, settings: SettingsClass, name: str):

        super().__init__(
            settings=settings,
            name=name
        )

    def generate_object(self):
        yield copy.deepcopy(self)

    def transform(self, image: GenericImage) -> LogarithmTransformedImage:
        # Create placeholder response map.
        response_map = LogarithmTransformedImage(
            image_data=None,
            template=image
        )

        if image.is_empty():
            return response_map

        image_data = image.get_voxel_grid()
        max_value_original = np.max(np.abs(image_data))
        image_data = np.sign(image_data) * np.log1p(np.abs(image_data))
        max_value_new = np.max(np.abs(image_data))

        # Prevent issues with alpha values that are not strictly positive.
        if not np.isfinite(max_value_new) or max_value_new == 0.0:
            max_value_new = 1.0
        if not np.isfinite(max_value_original):
            max_value_original = 1.0

        response_map.set_voxel_grid(
            voxel_grid=image_data * max_value_original / max_value_new
        )

        return response_map
