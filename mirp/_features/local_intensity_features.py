from functools import cache
from typing import Generator

import numpy as np
import pandas as pd
import scipy.ndimage as ndi


from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.base_feature import Feature
from mirp._features.utilities import rep
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class DataLocalIntensity(object):

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

        # Set data.
        if self.n_voxels > 300:
            self._filter_compute(image=image, mask=mask)
        else:
            self._direct_compute(image=image, mask=mask)

        if not self.is_empty():
            # Shrink to contain only voxels within the intensity mask.
            self.data = self.data.loc[self.data.in_roi == True, :]

    def _filter_compute(self, image: GenericImage, mask: BaseMask):
        # Determine distance
        distance = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * 10.0

        # Get maximal extension in cubic space
        base_ext = np.floor(distance / np.array(image.image_spacing))

        # Create displacement map
        df_base = pd.DataFrame({
            "x": rep(
                x=np.arange(-base_ext[2], base_ext[2] + 1),
                each=(2 * base_ext[0] + 1) * (2 * base_ext[1] + 1),
                times=1),
            "y": rep(
                x=np.arange(-base_ext[1], base_ext[1] + 1),
                each=2 * base_ext[0] + 1,
                times=2 * base_ext[2] + 1),
            "z": rep(
                x=np.arange(-base_ext[0], base_ext[0] + 1),
                each=1,
                times=(2 * base_ext[1] + 1) * (2 * base_ext[2] + 1))
        })

        # Calculate distances for displacement map
        df_base["dist"] = np.sqrt(np.sum(np.multiply(
            df_base.loc[:, ("z", "y", "x")].values, image.image_spacing
        ) ** 2.0, axis=1))

        # Identify elements in range
        df_base["set_weight"] = df_base.dist <= distance

        # Set weights for filter
        df_base["weight"] = np.zeros(len(df_base))
        df_base.loc[df_base.set_weight == True, "weight"] = 1.0 / np.sum(df_base.set_weight)

        # Update coordinates to start at 0
        df_base["x"] += base_ext[2]
        df_base["y"] += base_ext[1]
        df_base["z"] += base_ext[0]

        # Generate convolution filter
        conv_filter = np.zeros(shape=(
            int(np.max(df_base.z)) + 1,
            int(np.max(df_base.y)) + 1,
            int(np.max(df_base.x)) + 1)
        )
        conv_filter[df_base.z.astype(int), df_base.y.astype(int), df_base.x.astype(int)] = df_base.weight

        # Filter image using mean filter
        if image.get_default_lowest_intensity() is not None:
            # Use 0.0 constant for PET data
            img_avg = ndi.convolve(
                image.get_voxel_grid(),
                weights=conv_filter,
                mode="constant",
                cval=image.get_default_lowest_intensity()
            )
        else:
            img_avg = ndi.convolve(
                image.get_voxel_grid(),
                weights=conv_filter,
                mode="nearest"
            )

        # Construct data frame for comparison
        self.data = pd.DataFrame({
            "g": np.ravel(image.get_voxel_grid()),
            "g_loc": np.ravel(img_avg),
            "in_roi": np.ravel(mask.roi_intensity.get_voxel_grid())
        })

    def _direct_compute(self, image: GenericImage, mask: BaseMask):
        # Determine distance
        distance = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * 10.0

        # Construct data frame for comparison
        self.data = pd.DataFrame({
            "g": np.ravel(image.get_voxel_grid()),
            "g_loc": np.ravel(np.full(image.image_dimension, np.nan)),
            "in_roi": np.ravel(mask.roi_intensity.get_voxel_grid())
        })

        # Generate position matrix
        pos_mat = np.array(
            np.unravel_index(indices=np.arange(0, np.prod(image.image_dimension)), shape=image.image_dimension),
            dtype=np.float32
        ).transpose()

        # Iterate over voxels in the roi
        if np.sum(self.data.in_roi) > 1:
            for i in np.array(np.where(self.data.in_roi == True)).squeeze():
                # Determine distance from currently selected voxel
                vox_dist = np.sqrt(
                    np.sum(np.power(np.multiply(pos_mat - pos_mat[i, :], image.image_spacing), 2.0), axis=1))

                # Calculate mean grey level over all voxels within range
                self.data.loc[i, "g_loc"] = np.mean(self.data.g[vox_dist <= distance])

        else:
            i = np.where(self.data.in_roi == True)[0][0]

            # Determine distance from currently selected voxel
            vox_dist = np.sqrt(np.sum(np.power(np.multiply(pos_mat - pos_mat[i, :], image.image_spacing), 2.0), axis=1))

            # Calculate mean grey level over all voxels within range
            self.data.loc[i, "g_loc"] = np.mean(self.data.g[vox_dist <= distance])

    def is_empty(self):
        return self.data is None


class FeatureLocalIntensity(Feature):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def _data_key(self):
        return super()._data_key().update({
            "class": "local_intensity"
        })

    def clear_cache(self):
        super().clear_cache()
        self._get_data.cache_clear()

    @staticmethod
    @cache
    def _get_data(
            image: GenericImage,
            mask: BaseMask
    ) -> DataLocalIntensity:
        data = DataLocalIntensity()
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
    def _compute(data: DataLocalIntensity):
        raise NotImplementedError("Implement _compute for feature-specific computation.")

    def create_table_name(self):
        table_elements = self._get_base_table_name_element()
        self.table_name = "_".join(table_elements)


class FeatureLocalIntensityLocalPeak(FeatureLocalIntensity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Local intensity - local peak"
        self.abbr_name = "loc_peak_loc"
        self.ibsi_id = "VJGA"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataLocalIntensity) -> float:
        return np.max(data.data.loc[data.data.g == np.max(data.data.g), "g_loc"])


class FeatureLocalIntensityGlobalPeak(FeatureLocalIntensity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Local intensity - global peak"
        self.abbr_name = "loc_peak_glob"
        self.ibsi_id = "0F91"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: DataLocalIntensity) -> float:
        return np.max(data.data.g_loc)


def get_local_intensity_class_dict() -> dict[str, FeatureLocalIntensity]:
    class_dict = {
        "loc_peak_loc": FeatureLocalIntensityLocalPeak,
        "loc_peak_glob": FeatureLocalIntensityGlobalPeak
    }

    return class_dict


def generate_local_intensity_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str] = None
) -> Generator[FeatureLocalIntensity, None, None]:
    class_dict = get_local_intensity_class_dict()
    local_int_features = list(class_dict.keys())

    # Populate features if available.
    if features is None and settings.has_local_intensity_family():
        features = local_int_features

    # Terminate early if no features are set, and none are required.
    if features is None:
        return

    # Select only local intensity features, and return if none are present.
    features = [feature for feature in features if feature in local_int_features]
    if len(features) == 0:
        return

    for feature in features:
        yield class_dict[feature]()