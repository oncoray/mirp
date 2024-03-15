import copy
import numpy as np

from mirp.images.genericImage import GenericImage
from mirp.images.transformedImage import SquareTransformedImage
from mirp.imageFilters.genericFilter import GenericFilter
from mirp.settings.settingsGeneric import SettingsClass


class SquareTransformFilter(GenericFilter):

    def __init__(self, settings: SettingsClass, name: str):

        super().__init__(
            settings=settings,
            name=name
        )

    def generate_object(self):
        yield copy.deepcopy(self)

    def transform(self, image: GenericImage) -> SquareTransformedImage:
        # Create placeholder response map.
        response_map = SquareTransformedImage(
            image_data=None,
            template=image
        )

        if image.is_empty():
            return response_map

        image_data = image.get_voxel_grid()
        alpha = 1.0 / np.sqrt(np.max(np.abs(image_data)))

        response_map.set_voxel_grid(
            voxel_grid=np.power(alpha * image_data, 2.0)
        )

        return response_map
