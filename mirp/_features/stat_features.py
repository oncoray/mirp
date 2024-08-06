from functools import lru_cache
from typing import Generator

import numpy as np

from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.base_feature import Feature
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class DataStatistics(object):

    def __init__(self):
        # Raw intensity data
        self.image: np.ndarray | None = None

        # Number of voxels
        self.n_voxels: int | None = None

        # Mean intensity
        self.mu: float | None = None

        # Intensity standard deviation
        self.sigma: float | None = None

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

        # Convert to dataframe and remove entries outside the mask.
        image = mask.as_pandas_dataframe(image=image, intensity_mask=True)
        image = image[image.roi_int_mask == True]
        if len(image) == 0:
            return

        # Set image data
        self.image = image.g.values

        # Number of voxels
        self.n_voxels = len(image)

        # Mean intensity
        self.mu = np.mean(self.image)

        # Standard deviation
        self.sigma = np.std(self.image, ddof=0)

    def is_empty(self):
        return self.image is None


class FeatureStat(Feature):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def clear_cache(self):
        super().clear_cache()
        self._get_data.cache_clear()

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_data(
            image: GenericImage,
            mask: BaseMask
    ) -> DataStatistics:
        data = DataStatistics()
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
    def _compute(data: DataStatistics):
        raise NotImplementedError("Implement _compute for feature-specific computation.")

    def create_table_name(self):
        table_elements = self._get_base_table_name_element()
        self.table_name = "_".join(table_elements)


class FeatureStatMean(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - mean"
        self.abbr_name = "stat_mean"
        self.ibsi_id = "Q4LE"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        return data.mu


class FeatureStatVariance(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - variance"
        self.abbr_name = "stat_var"
        self.ibsi_id = "ECT3"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        return data.sigma ** 2.0


class FeatureStatSkewness(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - skewness"
        self.abbr_name = "stat_skew"
        self.ibsi_id = "KE2A"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        if data.sigma == 0.0:
            return 0.0
        return np.sum((data.image - data.mu) ** 3.0) / (data.n_voxels * data.sigma ** 3.0)


class FeatureStatKurtosis(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - kurtosis"
        self.abbr_name = "stat_kurt"
        self.ibsi_id = "IPH6"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        if data.sigma == 0.0:
            return 0.0
        return np.sum((data.image - data.mu) ** 4.0) / (data.n_voxels * data.sigma ** 4.0) - 3.0


class FeatureStatMedian(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - median"
        self.abbr_name = "stat_median"
        self.ibsi_id = "Y12H"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        return np.median(data.image)


class FeatureStatMinimum(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - minimum"
        self.abbr_name = "stat_min"
        self.ibsi_id = "1GSF"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        return np.min(data.image)


class FeatureStatPercentile(FeatureStat):
    def __init__(self, percentile: float, **kwargs):
        # This class allows for passing percentile values.
        super().__init__(**kwargs)

        # Set percentile
        self.percentile = percentile

        # Set IBSI identifier. Only the 10th and 90th percentile are explicitly identified.
        ibsi_id = ""
        if percentile == 10.0:
            ibsi_id = "QG58"
        elif percentile == 90.0:
            ibsi_id = "8DWT"
        self.ibsi_id = ibsi_id

        if percentile.is_integer():
            self.name = f"Statistics - {int(percentile)}th percentile"
            self.abbr_name = f"stat_p{int(percentile)}"
        else:
            self.name = self.name = f"Statistics - {percentile}th percentile"
            self.abbr_name = f"stat_p{percentile}"

        self.ibsi_compliant = True

    def compute(self, image: GenericImage, mask: BaseMask):
        # Get data.
        data = self._get_data(image=image, mask=mask)

        # Compute feature value.
        if data.is_empty():
            self.value = np.nan
        else:
            self.value = np.percentile(data.image, q=self.percentile)


class FeatureStatMaximum(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - maximum"
        self.abbr_name = "stat_max"
        self.ibsi_id = "84IY"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        return np.max(data.image)


class FeatureStatInterQuartileRange(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - interquartile range"
        self.abbr_name = "stat_iqr"
        self.ibsi_id = "SALO"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        return np.percentile(data.image, q=75) - np.percentile(data.image, q=25)


class FeatureStatRange(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - range"
        self.abbr_name = "stat_range"
        self.ibsi_id = "2OJQ"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        return float(np.max(data.image) - np.min(data.image))


class FeatureStatMeanAbsoluteDeviation(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - mean absolute deviation"
        self.abbr_name = "stat_mad"
        self.ibsi_id = "4FUA"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        return np.mean(np.abs(data.image - data.mu))


class FeatureStatRobustMeanAbsoluteDeviation(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - robust mean absolute deviation"
        self.abbr_name = "stat_rmad"
        self.ibsi_id = "1128"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        selected_values = data.image[
            (data.image >= np.percentile(data.image, q=10))
            & (data.image <= np.percentile(data.image, q=90))
        ]
        return np.mean(np.abs(selected_values - np.mean(selected_values)))


class FeatureStatMedianAbsoluteDeviation(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - median absolute deviation"
        self.abbr_name = "stat_medad"
        self.ibsi_id = "N72L"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        return np.mean(np.abs(data.image - np.median(data.image)))


class FeatureStatCoefficientOfVariation(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - coefficient of variation"
        self.abbr_name = "stat_cov"
        self.ibsi_id = "7TET"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        if data.sigma == 0.0:
            return 0.0
        return data.sigma / data.mu


class FeatureStatQuartileCoefficientOfDispersion(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - quartile coefficient of dispersion"
        self.abbr_name = "stat_qcod"
        self.ibsi_id = "9S40"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        return (
                (np.percentile(data.image, q=75) - np.percentile(data.image, q=25))
                / (np.percentile(data.image, q=75) + np.percentile(data.image, q=25))
        )


class FeatureStatEnergy(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - energy"
        self.abbr_name = "stat_energy"
        self.ibsi_id = "N8CA"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        return np.sum(data.image ** 2.0)


class FeatureStatRootMeanSquare(FeatureStat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Statistics - "
        self.abbr_name = "stat_rms"
        self.ibsi_id = "5ZWQ"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataStatistics) -> float:
        return  np.sqrt(np.sum(data.image ** 2.0) / data.n_voxels)


def get_statistics_class_dict() -> dict[str, FeatureStat]:
    class_dict = {
        "stat_mean": FeatureStatMean,
        "stat_var": FeatureStatVariance,
        "stat_skew": FeatureStatSkewness,
        "stat_kurt": FeatureStatKurtosis,
        "stat_median": FeatureStatMedian,
        "stat_min": FeatureStatMinimum,
        "stat_p": FeatureStatPercentile,
        "stat_max": FeatureStatMaximum,
        "stat_iqr": FeatureStatInterQuartileRange,
        "stat_range": FeatureStatRange,
        "stat_mad": FeatureStatMeanAbsoluteDeviation,
        "stat_rmad": FeatureStatRobustMeanAbsoluteDeviation,
        "stat_medad": FeatureStatMedianAbsoluteDeviation,
        "stat_cov": FeatureStatCoefficientOfVariation,
        "stat_qcod": FeatureStatQuartileCoefficientOfDispersion,
        "stat_energy": FeatureStatEnergy,
        "stat_rms": FeatureStatRootMeanSquare
    }

    return class_dict


def generate_stat_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str] = None
) -> Generator[FeatureStat, None, None]:
    class_dict = get_statistics_class_dict()
    stat_features = list(class_dict.keys())

    # Populate features if available.
    if features is None and settings.has_stats_family():
        features = stat_features

    # Terminate early if no features are set, and none are required.
    if features is None:
        return

    # Select only statistical features, and return if none are present.
    features = [feature for feature in features if feature in stat_features]
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
