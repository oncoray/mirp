from functools import lru_cache
from typing import Generator

import numpy as np
import pandas as pd


from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.histogram import HistogramDerivedFeature
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class DataIntensityVolumeHistogram(object):

    def __init__(self):
        # Raw intensity data
        self.data: pd.DataFrame | None = None

        # Number of voxels
        self.n_voxels: int | None = None

        # Maximum intensity
        self.next_max_intensity: float | int | None = None

    def compute(
            self,
            image: GenericImage,
            mask: BaseMask,
            discretisation_method: str,
            bin_number: int | None = None,
            bin_size: float | None = None
    ):
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

        # Convert image volume to table.
        data = mask.as_pandas_dataframe(image=image, intensity_mask=True)
        data = data[data.roi_int_mask == True]

        # Set number of voxels.
        self.n_voxels = len(data)

        # Set intensity range
        if mask.intensity_range is None:
            intensity_range = np.array([np.nan, np.nan])
        else:
            intensity_range = np.array(mask.intensity_range).astype(float)

        if np.isnan(intensity_range[0]):
            intensity_range[0] = float(np.min(data.g))
        if np.isnan(intensity_range[1]):
            intensity_range[1] = float(np.max(data.g))

        # Determine the discretisation method.
        if discretisation_method == "none":
            discretisation_method = image.get_default_ivh_discretisation_method()

        # Set bin_number, if required.
        if discretisation_method == "fixed_bin_number" and bin_number is None:
            bin_number = image.get_default_ivh_bin_number()
            if bin_number is None:
                raise ValueError("IVH bin number should be provided.")

        # Set bin_size, if required.
        if discretisation_method == "fixed_bin_size":
            bin_size = image.get_default_ivh_bin_size()
            if bin_size is None:
                raise ValueError("IVH bin size should be provided.")

        if discretisation_method == "none":
            # No transformation.

            # Set number of bins. The number of bins is equal to the number of grey levels present.
            bin_number = intensity_range[1] - intensity_range[0] + 1.0

            # Create histogram by grouping by intensity level and counting bin size
            data = data.groupby(by="g").size().reset_index(name="n")

            # Append empty grey levels to histogram
            levels = np.arange(start=intensity_range[0], stop=intensity_range[1] + 1)
            missing_levels = levels[np.logical_not(np.isin(levels, data.g))]
            n_missing = len(missing_levels)
            if n_missing > 0:
                data = pd.concat([
                    data,
                    pd.DataFrame({"g": missing_levels, "n": np.zeros(n_missing)})
                ], ignore_index=True)

            # Set maximum intensity.
            self.next_max_intensity = intensity_range[1] + 1.0

        elif discretisation_method == "fixed_bin_size":
            # Fixed bin size/width calculations

            # Get the number of bins
            bin_number = np.ceil((intensity_range[1] - intensity_range[0]) / bin_size) + 1.0

            # Bin voxels
            data.g = np.floor((data.g - intensity_range[0]) / (bin_size * 1.0)) + 1.0

            # Set voxels with grey level lower than 0.0 to 1.0. This may occur with non-mask voxels
            # and voxels with the minimum intensity
            data.loc[data["g"] <= 0.0, "g"] = 1.0

            # Create histogram by grouping by intensity level and counting bin size
            data = data.groupby(by="g").size().reset_index(name="n")

            # Append empty grey levels to histogram
            levels = np.arange(start=1, stop=int(bin_number) + 1)
            missing_levels = levels[np.logical_not(np.isin(levels, data.g))]
            n_missing = len(missing_levels)
            if n_missing > 0:
                data = pd.concat([
                    data,
                    pd.DataFrame({"g": missing_levels, "n": np.zeros(n_missing)})
                ], ignore_index=True)

            # Replace g by the bin centers
            data.loc[:, "g"] = intensity_range[0] + (data["g"] - 0.5) * bin_size * 1.0

            # Update intensity range
            intensity_range[0] = np.min(data.g)
            intensity_range[1] = np.max(data.g)

            # Set maximum intensity.
            self.next_max_intensity = intensity_range[1] + bin_size

        elif discretisation_method == "fixed_bin_number":
            # Calculation for all other image types

            data.loc[:, "g"] = np.floor(
                bin_number * (data["g"] - intensity_range[0]) / (intensity_range[1] - intensity_range[0])) + 1.0

            # Update values at the boundaries
            data.loc[data["g"] <= 0.0, "g"] = 1.0
            data.loc[data["g"] >= bin_number * 1.0, "g"] = bin_number * 1.0

            # Create histogram by grouping by intensity level and counting bin size
            data = data.groupby(by="g").size().reset_index(name="n")

            # Append empty grey levels to histogram
            levels = np.arange(start=1, stop=bin_number + 1)
            missing_levels = levels[np.logical_not(np.isin(levels, data.g))]
            n_missing = len(missing_levels)
            if n_missing > 0:
                data = pd.concat([
                    data,
                    pd.DataFrame({"g": missing_levels, "n": np.zeros(n_missing)})
                ], ignore_index=True)

            # Update grey level range
            intensity_range[0] = 1.0
            intensity_range[1] = bin_number

            # Set maximum intensity.
            self.next_max_intensity = bin_number + 1.0

        else:
            raise ValueError(f"{discretisation_method} is not a valid IVH discretisation method.")

        # Order histogram table by increasing grey level
        data = data.sort_values(by="g")

        # Calculate intensity fraction
        data["gamma"] = (data.g - intensity_range[0]) / (intensity_range[1] - intensity_range[0])

        # Calculate volume fraction
        data["nu"] = 1.0 - (np.cumsum(np.append([0], data.n))[0:int(bin_number)]) * 1.0 / (self.n_voxels * 1.0)

        # Set data.
        self.data = data

    def is_empty(self):
        return self.data is None or len(self.data) == 0


class FeatureIntensityVolumeHistogram(HistogramDerivedFeature):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def clear_cache(self):
        super().clear_cache()
        self._get_data.cache_clear()

    def _data_key(self):
        return super()._data_key().update({
            "class": "IVH"
        })

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_data(
            image: GenericImage,
            mask: BaseMask,
            discretisation_method: str,
            bin_number: int | None = None,
            bin_size: float | None = None
    ) -> DataIntensityVolumeHistogram:
        data = DataIntensityVolumeHistogram()
        data.compute(
            image=image,
            mask=mask,
            discretisation_method=discretisation_method,
            bin_number=bin_number,
            bin_size=bin_size
        )

        return data

    def compute(self, image: GenericImage, mask: BaseMask):
        # Get data.
        data = self._get_data(
            image=image,
            mask=mask,
            discretisation_method=self.discretisation_method,
            bin_number=self.bin_number,
            bin_size=self.bin_width
        )

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
            x = data.next_max_intensity

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
            x_0 = data.next_max_intensity

        x_1 = data.data.loc[data.data.nu <= self.percentile[1] / 100.0, :].g.min()
        if np.isnan(x_1):
            x_1 = data.next_max_intensity

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


def generate_ivh_features(
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

    discretisation_parameters = {
        "discretisation_method": settings.ivh_discretisation_method,
        "bin_width": settings.ivh_discretisation_bin_width,
        "bin_number": settings.ivh_discretisation_n_bins
    }

    for feature in features:
        if feature == "ivh_v":
            for percentile in volume_percentiles:
                yield class_dict[feature](
                    percentile=percentile,
                    **discretisation_parameters
                )
        elif feature == "ivh_i":
            for percentile in intensity_percentiles:
                yield class_dict[feature](
                    percentile=percentile,
                    **discretisation_parameters
                )
        elif feature == "ivh_diff_v":
            for percentile in volume_difference_range:
                yield class_dict[feature](
                    percentile=percentile,
                    **discretisation_parameters
                )
        elif feature == "ivh_diff_i":
            for percentile in intensity_difference_range:
                yield class_dict[feature](
                    percentile=percentile,
                    **discretisation_parameters
                )
        else:
            yield class_dict[feature](**discretisation_parameters)
