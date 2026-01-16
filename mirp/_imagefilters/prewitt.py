from typing import Any

import numpy as np
import pandas as pd
import copy

from mirp._images.generic_image import GenericImage
from mirp._images.transformed_image import TransformedImage
from mirp.settings.generic import SettingsClass
from mirp._imagefilters.generic import GenericFilter
from mirp._imagefilters.utilities import FilterSet2D, FilterSet3D, SeparableFilterSet


class PrewittTransformedImage(TransformedImage):
    def __init__(
            self,
            boundary_condition: None | str = None,
            template: None | GenericImage = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Filter parameters
        self.boundary_condition = boundary_condition

        # Update image parameters using the template.
        if isinstance(template, GenericImage):
            self.update_from_template(template=template)

    def get_file_name_descriptor(self) -> list[str]:
        descriptors = super().get_file_name_descriptor()

        descriptors += ["prewitt"]

        return descriptors

    def get_export_attributes(self) -> dict[str, Any]:
        parent_attributes = super().get_export_attributes()

        attributes = [
            ("filter_type", "prewitt"),
            ("boundary_condition", self.boundary_condition)
        ]

        parent_attributes.update(dict(attributes))

        return parent_attributes

    def parse_feature_names(self, x: None | pd.DataFrame) -> pd.DataFrame:
        x = super().parse_feature_names(x=x)

        feature_name_prefix = ["prewitt"]

        if len(feature_name_prefix) > 0:
            feature_name_prefix = "_".join(feature_name_prefix)
            feature_name_prefix += "_"
            x.columns = feature_name_prefix + x.columns

        return x


class PrewittFilter(GenericFilter):

    def __init__(self, image: GenericImage, settings: SettingsClass, name: str):

        super().__init__(image=image, settings=settings, name=name)
        self.ibsi_compliant = False

        # Set boundary condition
        self.mode = settings.img_transform.prewitt_boundary_condition

    def _not_isotropic_warning_message(self):
        return f"The Prewitt filter requires isotropic voxel spacing."

    def generate_object(self):
        yield copy.deepcopy(self)

    def transform(self, image: GenericImage) -> PrewittTransformedImage:
        # Create placeholder separable wavelet response map.
        response_map = PrewittTransformedImage(
            image_data=None,
            boundary_condition=self.mode,
            template=image
        )
        response_map.ibsi_compliant = self.ibsi_compliant and image.ibsi_compliant

        if image.is_empty():
            return response_map

        # Check that the voxel spacing is isotropic.
        self.check_isotropic_image(image=image)

        # Initialise voxel grid.
        response_voxel_grid = np.zeros(image.image_dimension, np.float32)

        # Get filter list (can be 2D or 3D filters).
        filter_set_list = self.get_filter_set()

        for ii, filter_set in enumerate(filter_set_list):
            # Convolve and compute response map.
            response_voxel_grid += np.power(
                filter_set.convolve(voxel_grid=image.get_voxel_grid(), mode=self.mode),
                2.0
            )

        # To get to the magnitude, take the square root.
        response_voxel_grid = np.sqrt(response_voxel_grid)

        # Set voxel grid.
        response_map.set_voxel_grid(voxel_grid=response_voxel_grid)

        return response_map

    def get_filter_set(self) -> list[SeparableFilterSet]:
        filter_set = [
            SeparableFilterSet(filter_x=np.array([1.0, 0.0, -1.0])),
            SeparableFilterSet(filter_y=np.array([1.0, 0.0, -1.0]))
        ]

        if not self.separate_slices:
            filter_set += [SeparableFilterSet(filter_z=np.array([1.0, 0.0, -1.0]))]

        return filter_set
