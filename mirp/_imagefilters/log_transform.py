import copy
from typing import Any

import numpy as np
import pandas as pd

from mirp._images.generic_image import GenericImage
from mirp._imagefilters.generic import GenericFilter
from mirp._images.transformed_image import TransformedImage
from mirp.settings.generic import SettingsClass


class LogarithmTransformedImage(TransformedImage):
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
        descriptors += ["lgrthm"]

        return descriptors

    def get_export_attributes(self) -> dict[str, Any]:
        parent_attributes = super().get_export_attributes()
        attributes = [("filter_type", "logarithm_transformation")]
        parent_attributes.update(dict(attributes))

        return parent_attributes

    def parse_feature_names(self, x: None | pd.DataFrame) -> pd.DataFrame:
        x = super().parse_feature_names(x=x)
        feature_name_prefix = ["lgrthm"]

        if len(feature_name_prefix) > 0:
            feature_name_prefix = "_".join(feature_name_prefix)
            feature_name_prefix += "_"
            x.columns = feature_name_prefix + x.columns

        return x


class LogarithmTransformFilter(GenericFilter):

    def __init__(self, image: GenericImage, settings: SettingsClass, name: str):

        super().__init__(image=image, settings=settings, name=name)

        # Logarithmic transform filters are not IBSI-compliant.
        self.ibsi_compliant: bool = False

    def generate_object(self):
        yield copy.deepcopy(self)

    def transform(self, image: GenericImage) -> LogarithmTransformedImage:
        # Create placeholder response map.
        response_map = LogarithmTransformedImage(
            image_data=None,
            template=image
        )
        response_map.ibsi_compliant = self.ibsi_compliant and image.ibsi_compliant

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
