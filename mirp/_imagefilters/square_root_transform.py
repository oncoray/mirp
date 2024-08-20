import copy
import numpy as np

from mirp._images.generic_image import GenericImage
from mirp._images.transformed_image import SquareRootTransformedImage
from mirp._imagefilters.generic import GenericFilter
from mirp.settings.generic import SettingsClass


class SquareRootTransformFilter(GenericFilter):

    def __init__(self, image: GenericImage, settings: SettingsClass, name: str):

        super().__init__(image=image, settings=settings, name=name)

        # Square root transform filters are not IBSI-compliant.
        self.ibsi_compliant: bool = False

    def generate_object(self):
        yield copy.deepcopy(self)

    def transform(self, image: GenericImage) -> SquareRootTransformedImage:
        # Create placeholder response map.
        response_map = SquareRootTransformedImage(
            image_data=None,
            template=image
        )
        response_map.ibsi_compliant = self.ibsi_compliant and image.ibsi_compliant

        if image.is_empty():
            return response_map

        image_data = image.get_voxel_grid()
        alpha = np.max(np.abs(image_data))

        # Prevent issues with alpha values that are not strictly positive.
        if not np.isfinite(alpha) or alpha == 0.0:
            alpha = 1.0

        response_map.set_voxel_grid(
            voxel_grid=np.sign(image_data) * np.sqrt(np.abs(image_data) * alpha)
        )

        return response_map
