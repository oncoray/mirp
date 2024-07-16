from functools import cache
from typing import Generator

import numpy as np
import pandas as pd

from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.histogram import get_discretisation_parameters, HistogramDerivedFeature
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class DataIH(object):

    def __init__(self):
        # Raw intensity data
        self.image: np.ndarray | None = None

        # Histogram data
        self.histogram: pd.DataFrame | None = None

        # Number of voxels
        self.n_voxels: int | None = None

        # Number of grey levels
        self.n_g: int | None = None

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

        # Number of grey levels
        self.n_g = mask.intensity_range[1] - mask.intensity_range[0] + 1

        # Define histogram
        histogram = image.groupby(by="g").size().reset_index(name="n")

        # Append empty grey levels to histogram
        levels = np.arange(start=0, stop=self.n_g) + 1
        missing_level = levels[np.logical_not(np.isin(levels, histogram.g))]
        n_missing = len(missing_level)
        if n_missing > 0:
            histogram = pd.concat([
                histogram,
                pd.DataFrame({
                    "g": missing_level,
                    "n": np.zeros(n_missing)
                })
            ], ignore_index=True)

        # Update histogram by sorting grey levels and adding bin probabilities
        histogram = histogram.sort_values(by="g")
        histogram["p"] = histogram.n / self.n_voxels

        # Add gradient data.
        if len(histogram) > 1:
            histogram["grad"] = np.gradient(histogram.n)
        else:
            histogram["grad"] = 0.0

        self.histogram = histogram

        # Mean intensity
        self.mu = np.sum(histogram.g * histogram.p)

        # Standard deviation
        self.sigma = np.sqrt(np.sum((histogram.g - self.mu) ** 2.0 * histogram.p))

    def is_empty(self):
        return self.image is None or self.histogram is None


class FeatureIH(HistogramDerivedFeature):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Perform close crop for intensity histograms.
        self.cropping_distance = 0.0

    def clear_cache(self):
        super().clear_cache()
        self._get_data.cache_clear()

    @staticmethod
    @cache
    def _get_data(
            image: GenericImage,
            mask: BaseMask
    ) -> DataIH:
        data = DataIH()
        data.compute(image=image, mask=mask)

        return data

    def compute(self, image: GenericImage, mask: BaseMask):
        # Discretise images.
        image, mask = self.discretise_image(
            image=image,
            mask=mask,
            discretisation_method=self.discretisation_method,
            bin_width=self.bin_width,
            bin_number=self.bin_number,
            cropping_distance=self.cropping_distance
        )

        # Get data.
        data = self._get_data(image=image, mask=mask)

        # Compute feature value.
        if data.is_empty():
            self.value = np.nan
        else:
            self.value = self._compute(data=data)

    @staticmethod
    def _compute(data: DataIH):
        raise NotImplementedError("Implement _compute for feature-specific computation.")

    def create_table_name(self):
        table_elements = (
                self._get_base_table_name_element()
                + self._get_discretisation_table_name_element()
        )
        self.table_name = "_".join(table_elements)


class FeatureIHMean(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - mean"
        self.abbr_name = "ih_mean"
        self.ibsi_id = "X6K6"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return data.mu


class FeatureIHVariance(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - variance"
        self.abbr_name = "ih_var"
        self.ibsi_id = "CH89"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return data.sigma ** 2.0


class FeatureIHSkewness(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - skewness"
        self.abbr_name = "ih_skew"
        self.ibsi_id = "88K1"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        if data.sigma == 0.0:
            return 0.0
        return np.sum((data.histogram.g - data.mu) ** 3.0 * data.histogram.p) / (data.sigma ** 3.0)


class FeatureIHKurtosis(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - kurtosis"
        self.abbr_name = "ih_kurt"
        self.ibsi_id = "C3I7"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        if data.sigma == 0.0:
            return 0.0
        return np.sum((data.histogram.g - data.mu) ** 4.0 * data.histogram.p) / (data.sigma ** 4.0) - 3.0


class FeatureIHMedian(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - median"
        self.abbr_name = "ih_median"
        self.ibsi_id = "WIFQ"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return np.median(data.image)


class FeatureIHMinimum(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - minimum"
        self.abbr_name = "ih_min"
        self.ibsi_id = "1PR8"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return np.min(data.image)


class FeatureIHPercentile(FeatureIH):
    def __init__(self, percentile: float, **kwargs):
        # This class allows for passing percentile values.
        super().__init__(**kwargs)

        # Set percentile
        self.percentile = percentile

        # Set IBSI identifier. Only the 10th and 90th percentile are explicitly identified.
        ibsi_id = ""
        if percentile == 10.0:
            ibsi_id = "GPMT"
        elif percentile == 90.0:
            ibsi_id = "OZ0C"
        self.ibsi_id = ibsi_id

        if percentile.is_integer():
            self.name = f"IH - {int(percentile)}th percentile"
            self.abbr_name = "ih_p" + str(int(percentile))

        self.ibsi_compliant = True

    def compute(self, image: GenericImage, mask: BaseMask):
        # Discretise images.
        image, mask = self.discretise_image(
            image=image,
            mask=mask,
            discretisation_method=self.discretisation_method,
            bin_width=self.bin_width,
            bin_number=self.bin_number,
            cropping_distance=self.cropping_distance
        )

        # Get data.
        data = self._get_data(image=image, mask=mask)

        # Compute feature value.
        if data.is_empty():
            self.value = np.nan
        else:
            self.value = np.percentile(data.image, q=self.percentile)


class FeatureIHMaximum(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - maximum"
        self.abbr_name = "ih_max"
        self.ibsi_id = "3NCY"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return np.max(data.image)


class FeatureIHMode(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - mode"
        self.abbr_name = "ih_mode"
        self.ibsi_id = "AMMC"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        mode_g = data.histogram.loc[data.histogram.n == np.max(data.histogram.n)].g.values
        return mode_g[np.argmin(np.abs(mode_g - data.mu))]


class FeatureIHInterQuartileRange(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - interquartile range"
        self.abbr_name = "ih_iqr"
        self.ibsi_id = "WR0O"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return np.percentile(data.image, q=75) - np.percentile(data.image, q=25)


class FeatureIHRange(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - range"
        self.abbr_name = "ih_range"
        self.ibsi_id = "5Z3W"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return float(np.max(data.image) - np.min(data.image))


class FeatureIHMeanAbsoluteDeviation(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - mean absolute deviation"
        self.abbr_name = "ih_mad"
        self.ibsi_id = "D2ZX"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return np.mean(np.abs(data.image - data.mu))


class FeatureIHRobustMeanAbsoluteDeviation(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - robust mean absolute deviation"
        self.abbr_name = "ih_rmad"
        self.ibsi_id = "WRZB"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        selected_values = data.image[
            (data.image >= np.percentile(data.image, q=10))
            & (data.image <= np.percentile(data.image, q=90))
        ]
        return np.mean(np.abs(selected_values - np.mean(selected_values)))


class FeatureIHMedianAbsoluteDeviation(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - median absolute deviation"
        self.abbr_name = "ih_medad"
        self.ibsi_id = "4RNL"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return np.mean(np.abs(data.image - np.median(data.image)))


class FeatureIHCoefficientOfVariation(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - coefficient of variation"
        self.abbr_name = "ih_cov"
        self.ibsi_id = "CWYJ"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        if data.sigma == 0.0:
            return 0.0
        return data.sigma / data.mu


class FeatureIHQuartileCoefficientOfDispersion(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - quartile coefficient of dispersion"
        self.abbr_name = "ih_qcod"
        self.ibsi_id = "SLWD"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return (
                (np.percentile(data.image, q=75) - np.percentile(data.image, q=25))
                / (np.percentile(data.image, q=75) + np.percentile(data.image, q=25))
        )


class FeatureIHEntropy(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - entropy"
        self.abbr_name = "ih_entropy"
        self.ibsi_id = "TLU2"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return -np.sum(data.histogram.p[data.histogram.p > 0.0] * np.log2(data.histogram.p[data.histogram.p > 0.0]))


class FeatureIHUniformity(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - uniformity"
        self.abbr_name = "ih_uniformity"
        self.ibsi_id = "BJ5W"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return np.sum(data.histogram.p ** 2.0)


class FeatureIHMaximumHistogramGradient(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - maximum histogram gradient"
        self.abbr_name = "ih_max_grad"
        self.ibsi_id = "12CE"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return np.max(data.histogram.grad)


class FeatureIHMaximumHistogramGradientGreyLevel(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - maximum histogram gradient grey level"
        self.abbr_name = "ih_max_grad_g"
        self.ibsi_id = "8E6O"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return data.histogram.g[data.histogram.grad.idxmax()]


class FeatureIHMinimumHistogramGradient(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - minimum histogram gradient"
        self.abbr_name = "ih_min_grad"
        self.ibsi_id = "VQB3"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return np.min(data.histogram.grad)


class FeatureIHMinimumHistogramGradientGreyLevel(FeatureIH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IH - minimum histogram gradient grey level"
        self.abbr_name = "ih_min_grad_g"
        self.ibsi_id = "RHQZ"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataIH) -> float:
        return data.histogram.g[data.histogram.grad.idxmin()]


def get_ih_class_dict() -> dict[str, FeatureIH]:
    class_dict = {
        "ih_mean": FeatureIHMean,
        "ih_var": FeatureIHVariance,
        "ih_skew": FeatureIHSkewness,
        "ih_kurt": FeatureIHKurtosis,
        "ih_median": FeatureIHMedian,
        "ih_min": FeatureIHMinimum,
        "ih_p": FeatureIHPercentile,
        "ih_max": FeatureIHMaximum,
        "ih_mode": FeatureIHMode,
        "ih_iqr": FeatureIHInterQuartileRange,
        "ih_range": FeatureIHRange,
        "ih_mad": FeatureIHMeanAbsoluteDeviation,
        "ih_rmad": FeatureIHRobustMeanAbsoluteDeviation,
        "ih_medad": FeatureIHMedianAbsoluteDeviation,
        "ih_cov": FeatureIHCoefficientOfVariation,
        "ih_qcod": FeatureIHQuartileCoefficientOfDispersion,
        "ih_entropy": FeatureIHEntropy,
        "ih_uniformity": FeatureIHUniformity,
        "ih_max_grad": FeatureIHMaximumHistogramGradient,
        "ih_max_grad_g": FeatureIHMaximumHistogramGradientGreyLevel,
        "ih_min_grad": FeatureIHMinimumHistogramGradient,
        "ih_min_grad_g": FeatureIHMinimumHistogramGradientGreyLevel
    }

    return class_dict


def generate_ih_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str] = None
) -> Generator[FeatureIH, None, None]:
    class_dict = get_ih_class_dict()
    ih_features = list(class_dict.keys())

    # Populate features if available.
    if features is None and settings.has_gldzm_family():
        features = ih_features

    # Terminate early if no features are set, and none are required.
    if features is None:
        return

    # Select only DZM-features, and return if none are present.
    features = [feature for feature in features if feature in ih_features]
    if len(features) == 0:
        return

    # Set default percentiles.
    percentiles = [10.0, 90.0]

    # Features are parametrised by the choice of discretisation parameters and spatial methods.
    for discretisation_parameters in get_discretisation_parameters(
        settings=settings
    ):
        for feature in features:
            if feature == "ih_p":
                for percentile in percentiles:
                    yield class_dict[feature](
                        percentile=percentile,
                        **discretisation_parameters
                    )
            else:
                yield class_dict[feature](
                    **discretisation_parameters
                )
