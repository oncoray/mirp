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

        # Maximum intensity
        self.max_intensity: float | int | None = None

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

        # Set maximum intensity.
        self.max_intensity = ...


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

    def _compute(self, data: DataIntensityVolumeHistogram):
        raise NotImplementedError("Implement _compute for feature-specific computation.")

    def create_table_name(self):
        table_elements = self._get_base_table_name_element()
        self.table_name = "_".join(table_elements)


class FeatureIVHVolumeFractionAtIntensity(FeatureIntensityVolumeHistogram):
    def __init__(self, percentile: float, **kwargs):
        # This class allows for passing percentile values.
        super().__init__(**kwargs)

        # Set percentile
        self.percentile = percentile

        # Set IBSI identifier. Only the 10th and 90th percentile are explicitly identified.
        ibsi_id = ""
        if percentile == 10.0:
            ibsi_id = "NK6P"
        elif percentile == 90.0:
            ibsi_id = "4279"
        self.ibsi_id = ibsi_id

        if percentile.is_integer():
            self.name = f"IVH - volume fraction at {int(percentile)}% intensity"
            self.abbr_name = f"ivh_v{int(percentile)}"
        else:
            self.name = f"IVH - volume fraction at {percentile}% intensity"
            self.abbr_name = f"ivh_v{percentile}"

        self.ibsi_compliant = True

    def _compute(self, data: DataIntensityVolumeHistogram):
        return data.data.loc[data.data.gamma >= self.percentile / 100.0, :].nu.max()


class FeatureIVHIntensityAtVolumeFraction(FeatureIntensityVolumeHistogram):
    def __init__(self, percentile: float, **kwargs):
        # This class allows for passing percentile values.
        super().__init__(**kwargs)

        # Set percentile
        self.percentile = percentile

        # Set IBSI identifier. Only the 10th and 90th percentile are explicitly identified.
        ibsi_id = ""
        if percentile == 10.0:
            ibsi_id = "PWN1"
        elif percentile == 90.0:
            ibsi_id = "BOHI"
        self.ibsi_id = ibsi_id

        if percentile.is_integer():
            self.name = f"IVH - intensity at {int(percentile)}% volume"
            self.abbr_name = f"ivh_i{int(percentile)}"
        else:
            self.name = f"IVH - intensity at {percentile}% volume"
            self.abbr_name = f"ivh_i{percentile}"

        self.ibsi_compliant = True

    def _compute(self, data: DataIntensityVolumeHistogram):
        x = data.data.loc[data.data.nu <= self.percentile / 100.0, :].g.min()
        if np.isnan(x):
            x = data.max_intensity

        return x


class FeatureIVHVolumeFractionDifference(FeatureIntensityVolumeHistogram):
    def __init__(self, percentile: tuple[float, float], **kwargs):
        # This class allows for passing percentile values.
        super().__init__(**kwargs)

        # Set percentile
        self.percentile = percentile

        # Set IBSI identifier. Only the 10th and 90th percentile are explicitly identified.
        ibsi_id = ""
        if percentile[0] == 10.0 and percentile[1] == 90.0:
            ibsi_id = "WITY"
        self.ibsi_id = ibsi_id

        if all(x.is_integer() for x in percentile):
            self.name = f"IVH - difference in volume fraction between {int(percentile[0])}% and {int(percentile[1])}% intensity"
            self.abbr_name = f"ivh_diff_v{int(percentile[0])}_v{int(percentile[1])}"
        else:
            self.name = f"IVH - difference in volume fraction between {percentile[0]}% and {percentile[1]}% intensity"
            self.abbr_name = f"ivh_diff_v{percentile[0]}_v{percentile[1]}"

        self.ibsi_compliant = True

    def _compute(self, data: DataIntensityVolumeHistogram):
        return (
                data.data.loc[data.data.gamma >= self.percentile[0] / 100.0, :].nu.max()
                - data.data.loc[data.data.gamma >= self.percentile[1] / 100.0, :].nu.max()
        )


class FeatureIVHIntensityDifference(FeatureIntensityVolumeHistogram):
    def __init__(self, percentile: tuple[float, float], **kwargs):
        # This class allows for passing percentile values.
        super().__init__(**kwargs)

        # Set percentile
        self.percentile = percentile

        # Set IBSI identifier. Only the 10th and 90th percentile are explicitly identified.
        ibsi_id = ""
        if percentile[0] == 10.0 and percentile[1] == 90.0:
            ibsi_id = "JXJA"
        self.ibsi_id = ibsi_id

        if all(x.is_integer() for x in percentile):
            self.name = f"IVH - difference in intensity between {int(percentile[0])}% and {int(percentile[1])}% volume"
            self.abbr_name = f"ivh_diff_i{int(percentile[0])}_i{int(percentile[1])}"
        else:
            self.name = f"IVH - difference in intensity between {percentile[0]}% and {percentile[1]}% volume"
            self.abbr_name = f"ivh_diff_i{percentile[0]}_i{percentile[1]}"

        self.ibsi_compliant = True

    def _compute(self, data: DataIntensityVolumeHistogram):
        x_0 = data.data.loc[data.data.nu <= self.percentile[0] / 100.0, :].g.min()
        if np.isnan(x_0):
            x_0 = data.max_intensity

        x_1 = data.data.loc[data.data.nu <= self.percentile[1] / 100.0, :].g.min()
        if np.isnan(x_1):
            x_1 = data.max_intensity

        return x_0 - x_1


class FeatureIVHAreaUnderCurve(FeatureIntensityVolumeHistogram):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IVH - area under IVH curve"
        self.abbr_name = "ivh_auc"
        self.ibsi_id = "9CMM"
        self.ibsi_compliant = False

    def _compute(self, data: DataIntensityVolumeHistogram):
        return np.trapz(y=data.data.nu, x=data.data.gamma)


def get_intensity_volume_histogram_class_dict() -> dict[str, FeatureIntensityVolumeHistogram]:
    class_dict = {
        "ivh_v": FeatureIVHVolumeFractionAtIntensity,
        "ivh_i": FeatureIVHIntensityAtVolumeFraction,
        "ivh_diff_v": FeatureIVHVolumeFractionDifference,
        "ivh_diff_i": FeatureIVHIntensityDifference,
        "ivh_auc": FeatureIVHAreaUnderCurve
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
    intensity_percentiles = [10.0, 25.0, 50.0, 75.0, 90.0]
    volume_percentiles = [10.0, 25.0, 50.0, 75.0, 90.0]
    intensity_difference_range = [(10.0, 90.0), (25.0, 75.0)]
    volume_difference_range = [(10.0, 90.0), (25.0, 75.0)]

    for feature in features:
        if feature == "ivh_v":
            for percentile in volume_percentiles:
                yield class_dict[feature](
                    percentile=percentile
                )
        elif feature == "ivh_i":
            for percentile in intensity_percentiles:
                yield class_dict[feature](
                    percentile=percentile
                )
        elif feature == "ivh_diff_v":
            for percentile in volume_difference_range:
                yield class_dict[feature](
                    percentile=percentile
                )
        elif feature == "ivh_diff_i":
            for percentile in intensity_difference_range:
                yield class_dict[feature](
                    percentile=percentile
                )
        else:
            yield class_dict[feature]()
