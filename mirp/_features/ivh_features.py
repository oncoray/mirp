from functools import cache
from typing import Generator

import numpy as np
import pandas as pd
import scipy.ndimage as ndi


from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.base_feature import Feature
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class DataIntensityVolumeHistogram(object):

    def __init__(self):
        # Raw intensity data
        self.data: pd.DataFrame | None = None

        # Number of voxels
        self.n_voxels: int | None = None

    def compute(self, image: GenericImage, mask: BaseMask):
        # Skip processing if input image and/or roi are missing
        if image is None:
            raise ValueError(
                "image cannot be None, but may not have been provided in the calling function."
            )
        if mask is None:
            raise ValueError(
                "mask cannot be None, but may not have been provided in the calling function."
            )

        # Check if data actually exists
        if image.is_empty() or mask.roi_intensity.is_empty_mask():
            return

        # Set number of voxels.
        self.n_voxels = np.sum(mask.roi_intensity.get_voxel_grid())




class FeatureIntensityVolumeHistogram(Feature):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def clear_cache(self):
        super().clear_cache()
        self._get_data.cache_clear()

    @staticmethod
    @cache
    def _get_data(
            image: GenericImage,
            mask: BaseMask
    ) -> DataIntensityVolumeHistogram:
        data = DataIntensityVolumeHistogram()
        data.compute(image=image, mask=mask)

        return data

    def compute(self, image: GenericImage, mask: BaseMask):
        # Get data.
        data = self._get_data(image=image, mask=mask)

        # Compute feature value.
        if data.is_empty():
            self.value = np.nan
        else:
            self.value = self._compute(data=data)

    @staticmethod
    def _compute(data: DataIntensityVolumeHistogram):
        raise NotImplementedError("Implement _compute for feature-specific computation.")

    def create_table_name(self):
        table_elements = self._get_base_table_name_element()
        self.table_name = "_".join(table_elements)


def get_intensity_volume_histogram_class_dict() -> dict[str, FeatureIntensityVolumeHistogram]:
    class_dict = {
    }

    return class_dict


def generate_local_intensity_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str] = None
) -> Generator[FeatureIntensityVolumeHistogram, None, None]:
    class_dict = get_intensity_volume_histogram_class_dict()
    ivh_features = list(class_dict.keys())

    # Populate features if available.
    if features is None and settings.has_ivh_family():
        features = ivh_features

    # Terminate early if no features are set, and none are required.
    if features is None:
        return

    # Select only ivh features, and return if none are present.
    features = [feature for feature in features if feature in ivh_features]
    if len(features) == 0:
        return

    # Set default percentiles.
    percentiles = [10.0, 90.0]

    for feature in features:
        if feature == "stat_p":
            for percentile in percentiles:
                yield class_dict[feature](
                    percentile=percentile
                )
        else:
            yield class_dict[feature]()