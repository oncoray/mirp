import numpy as np
from mirp._features.histogram import HistogramDerivedFeature
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass

from typing import Generator


def get_feature_pooling_parameters(
        settings: FeatureExtractionSettingsClass,
        spatial_method: str
) -> Generator[str, None, None]:
    # For spatial methods that always yield one feature from one (merged) matrix, only use average.
    if spatial_method in ["2.5d", "3d", "2.5d_volume_merge", "3d_volume_merge"]:
        yield "average"
    else:
        for feature_pooling_method in settings.texture_feature_pooling_method:
            yield feature_pooling_method


class FeatureTexture(HistogramDerivedFeature):

    def __init__(
            self,
            spatial_method: str,
            feature_pooling_method: str = "average",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.spatial_method = spatial_method.lower()
        self.feature_pooling_method = feature_pooling_method

    def _get_spatial_table_name_element(self) -> list[str | None]:
        if self.feature_pooling_method == "average":
            fpm = "avg"
        elif self.feature_pooling_method == "min":
            fpm = "min"
        elif self.feature_pooling_method == "max":
            fpm = "max"
        elif self.feature_pooling_method == "range":
            fpm = "range"
        elif self.feature_pooling_method == "std":
            fpm = "std"
        elif self.feature_pooling_method == "var":
            fpm = "var"
        else:
            raise ValueError(f"Did not recognise feature pooling method: {self.feature_pooling_method}.")

        if self.spatial_method == "2d_average":
            table_elements = ["2d_" + fpm]
        elif self.spatial_method == "2d_slice_merge":
            table_elements = ["2d_s_mrg" + ("_" + fpm if fpm != "avg" else "")]
        elif self.spatial_method == "2.5d_direction_merge":
            table_elements = ["2.5d_d_mrg" + ("_" + fpm if fpm != "avg" else "")]
        elif self.spatial_method == "2.5d_volume_merge":
            table_elements = ["2.5d_v_mrg"]
        elif self.spatial_method == "3d_average":
            table_elements = ["3d_" + fpm]
        elif self.spatial_method == "3d_volume_merge":
            table_elements = ["3d_v_mrg"]
        elif self.spatial_method == "2d":
            table_elements = ["2d" + ("_" + fpm if fpm != "avg" else "")]
        elif self.spatial_method == "2.5d":
            table_elements = ["2.5d"]
        elif self.spatial_method == "3d":
            table_elements = ["3d"]
        else:
            raise ValueError(f"Did not recognise spatial method: {self.spatial_method}")

        return table_elements

    def update_ibsi_compliance(self):
        super().update_ibsi_compliance()

        if self.feature_pooling_method != "average":
            # Feature pooling methods other than average pooling are not IBSI-compliant. Note that average pooling
            # for spatial methods such as "3d_v_mrg" is simply an identity operation, because only a single matrix
            # is formed to compute a feature from.
            self.ibsi_compliant = False

    def pool_feature_values(self, x: list[float]) -> float:

        if np.all(np.isnan(x)):
            return np.nan
        elif self.feature_pooling_method == "average":
            return np.nanmean(x)

        elif self.feature_pooling_method == "min":
            return np.nanmin(x)

        elif self.feature_pooling_method == "max":
            return np.nanmax(x)

        elif self.feature_pooling_method == "range":
            return np.nanmax(x) - np.nanmin(x)

        elif self.feature_pooling_method == "std":
            value = np.nanstd(x)
            return value if np.isfinite(value) else np.nan

        elif self.feature_pooling_method == "var":
            value = np.nanvar(x)
            return value if np.isfinite(value) else np.nan

        else:
            raise ValueError(f"Did not recognise feature pooling method: {self.feature_pooling_method}.")