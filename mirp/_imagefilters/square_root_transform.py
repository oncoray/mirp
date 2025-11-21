import copy
from typing import Any

import numpy as np
import pandas as pd

from mirp._images.generic_image import GenericImage
from mirp._imagefilters.generic import GenericFilter
from mirp._images.transformed_image import TransformedImage
from mirp.settings.generic import SettingsClass


class SquareRootTransformedImage(TransformedImage):
    def __init__(
            self,
            template: None | GenericImage = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Update image parameters using the template.
        if isinstance(template, GenericImage):
            self.update_from_template(template=template)

    def get_file_name_descriptor(self) -> list[str]:
        descriptors = super().get_file_name_descriptor()
        descriptors += ["sqrt"]

        return descriptors

    def get_export_attributes(self) -> dict[str, Any]:
        parent_attributes = super().get_export_attributes()
        attributes = [("filter_type", "square_root_transformation")]
        parent_attributes.update(dict(attributes))

        return parent_attributes

    def parse_feature_names(self, x: None | pd.DataFrame) -> pd.DataFrame:
        x = super().parse_feature_names(x=x)
        feature_name_prefix = ["sqrt"]

        if len(feature_name_prefix) > 0:
            feature_name_prefix = "_".join(feature_name_prefix)
            feature_name_prefix += "_"
            x.columns = feature_name_prefix + x.columns

        return x



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
