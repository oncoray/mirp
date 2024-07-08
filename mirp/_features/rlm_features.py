from functools import cache
from typing import Generator

import numpy as np
import copy

import pandas as pd

from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.histogram import HistogramDerivedFeature, get_discretisation_parameters
from mirp._features.texture_matrix import DirectionalMatrix
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class MatrixRLM(DirectionalMatrix):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Placeholders for derivative values computed using set_values_from_matrix
        # Run-length matrix
        self.rij: pd.DataFrame | None = None

        # Marginal sum over grey levels
        self.ri: pd.DataFrame | None = None

        # Marginal sum over run lengths
        self.rj: pd.DataFrame | None = None

        # Number of runs
        self.n_s: int | None = None

    def compute(
            self,
            data: pd.DataFrame | None,
            image_dimension: tuple[int, int, int] | None = None,
            **kwargs
    ):
        # Check if the df_img actually exists
        if data is None:
            return

        # Check if the roi contains any masked voxels. If this is not the case, don't construct the GLRLM.
        if not np.any(data.roi_int_mask):
            return

        # Create local copies of the image table
        if self.spatial_method in ["3d_average", "3d_volume_merge"]:
            data = copy.deepcopy(data)

        elif self.spatial_method in ["2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge"]:
            data = copy.deepcopy(data[data.z == self.slice_id])
            data["index_id"] = np.arange(0, len(data))
            data["z"] = 0
            data = data.reset_index(drop=True)

        else:
            self._spatial_method_error()

        # Set grey level of voxels outside ROI to NaN
        data.loc[data.roi_int_mask == False, "g"] = np.nan

        # Set the number of voxels
        self.n_voxels = np.sum(data.roi_int_mask.values)

        # Determine update index number for direction
        if ((
                self.direction[2]
                + self.direction[1] * image_dimension[2]
                + self.direction[0] * image_dimension[2] * image_dimension[1]
        ) >= 0):
            direction = self.direction
        else:
            direction = tuple(-x for x in self.direction)

        # Step size
        index_update = (
                direction[2]
                + direction[1] * image_dimension[2]
                + direction[0] * image_dimension[2] * image_dimension[1]
        )

        # Generate information concerning segments
        n_segments = index_update  # Number of segments

        # Check if the number of segments is greater than one
        if n_segments == 0:
            return

        # Nominal segment length
        segment_length = (len(data) - 1) // index_update + 1

        # Initial segment length for transitions (nominal length - 1)
        trans_segment_length = np.tile([segment_length - 1], reps=n_segments)

        # Number of full segments
        full_len_trans = n_segments - n_segments * segment_length + len(data)

        # Update full segments
        trans_segment_length[0:full_len_trans] += 1

        # Create transition vector
        trans_vec = (
                np.tile(
                    np.arange(start=0, stop=len(data), step=index_update),
                    reps=index_update
                )
                + np.repeat(
                    np.arange(start=0, stop=n_segments),
                    repeats=segment_length
                )
            )
        trans_vec = trans_vec[trans_vec < len(data)]

        # Determine valid transitions
        to_index = self.coord_to_index(
            x=data.x.values + direction[2],
            y=data.y.values + direction[1],
            z=data.z.values + direction[0],
            dims=image_dimension
        )

        # Determine which transitions are valid
        end_ind = np.nonzero(to_index[trans_vec] < 0)[0]  # Find transitions that form an endpoint.

        # Get an interspersed array of intensities. Runs are broken up by np.nan
        intensities = np.insert(data.g.values[trans_vec], end_ind + 1, np.nan)

        # Determine run length start and end indices
        rle_end = np.array(np.append(np.where(intensities[1:] != intensities[:-1]), len(intensities) - 1))
        rle_start = np.cumsum(np.append(0, np.diff(np.append(-1, rle_end))))[:-1]

        # Generate matrix
        matrix = pd.DataFrame({
            "i": intensities[rle_start],
            "r": rle_end - rle_start + 1
        })
        matrix = matrix.loc[~np.isnan(matrix.i), :]
        matrix = matrix.groupby(by=["i", "j"]).size().reset_index(name="rij")

        # Add matrix to object
        self.matrix = matrix

    def set_values_from_matrix(self):
        if self.is_empty():
            return

        # Copy of matrix
        self.rij = copy.deepcopy(self.matrix)

        # Sum over grey levels
        self.ri = self.matrix.groupby(by="i")["rij"].sum().reset_index().rename(columns={"rij": "ri"})

        # Sum over run lengths
        self.rj = self.matrix.groupby(by="j")["rij"].sum().reset_index().rename(columns={"rij": "rj"})

        # Constant definitions
        self.n_s = np.sum(self.matrix.rij) * 1.0  # Number of runs

    @staticmethod
    def _get_grouping_columns():
        return ["i", "j"]


class FeatureRLM(HistogramDerivedFeature):

    def __init__(
            self,
            spatial_method: str,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.spatial_method = spatial_method.lower()

        # Perform close crop for RLM.
        self.cropping_distance = 0.0

    def get_matrix(
            self,
            image: GenericImage,
            mask: BaseMask,
    ) -> list[MatrixRLM]:
        # First discretise the image.
        image, mask = self.discretise_image(
            image=image,
            mask=mask,
            discretisation_method=self.discretisation_method,
            bin_width=self.bin_width,
            bin_number=self.bin_number,
            cropping_distance=self.cropping_distance
        )

        # Then get matrix or matrices
        matrix_list = self._get_matrix(
            image=image,
            mask=mask,
            spatial_method=self.spatial_method
        )

        return matrix_list

    @staticmethod
    @cache
    def _get_matrix(
            image: GenericImage,
            mask: BaseMask,
            spatial_method: str
    ) -> list[MatrixRLM]:
        # Represent image and mask as a dataframe.
        data = mask.as_pandas_dataframe(
            image=image,
            intensity_mask=True
        )

        # Instantiate a helper copy of the current class to be able to use class methods without tying the cache to the
        # instance of the original object from which this method is called.
        matrix_instance = MatrixRLM(
            spatial_method=spatial_method
        )

        # Compute the required matrices.
        matrix_list = [
            matrix.compute(data=data, image_dimenion=image.image_dimension)
            for matrix in matrix_instance.generate(prototype=MatrixRLM, n_slices=image.image_dimension[0])
        ]

        # Merge according to the spatial method.
        matrix_list = matrix_instance.merge(matrix_list, prototype=MatrixRLM)

        # Compute additional values from the individual matrices.
        matrix_list = [matrix.set_values_from_matrix() for matrix in matrix_list]

        return matrix_list

    def clear_cache(self):
        super().clear_cache()
        self._get_matrix.cache_clear()

    def compute(self, image: GenericImage, mask: BaseMask):
        # Compute or retrieve matrices from cache.
        matrices = self.get_matrix(image=image, mask=mask)

        # Compute feature value from matrices, and average over matrices.
        values = [self._compute(matrix=matrix) for matrix in matrices]
        self.value = np.nanmean(values)

    @staticmethod
    def _compute(matrix: MatrixRLM):
        raise NotImplementedError("Implement _compute for feature-specific computation.")


class FeatureRLMSRE(FeatureRLM):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - short runs emphasis"
        self.abbr_name = "rlm_sre"
        self.ibsi_id = "22OV"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.rj.rj / matrix.rj.j ** 2.0) / matrix.n_s


def get_rlm_class_dict() -> dict[str, FeatureRLM]:
    class_dict = {
        "rlm_sre": FeatureRLMSRE,
        "rlm_lre": 2,
        "rlm_lgre": 3,
        "rlm_hgre": 4,
        "rlm_srlge": 5,
        "rlm_srhge": 6,
        "rlm_lrlge": 7,
        "rlm_lrhge": 8,
        "rlm_glnu": 9,
        "rlm_glnu_norm": 10,
        "rlm_rlnu": 11,
        "rlm_rlnu_norm": 12,
        "rlm_r_perc": 13,
        "rlm_gl_var": 14,
        "rlm_rl_var": 15,
        "rlm_rl_entr": 16
    }

    return class_dict


def generate_rlm_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str]
) -> Generator[FeatureRLM, None, None]:
    class_dict = get_rlm_class_dict()
    rlm_features = set(class_dict.keys())

    # Populate features if available.
    if features is None and settings.has_glrlm_family():
        features = rlm_features

    # Terminate early if no features are set, and none are required.
    if features is None:
        return

    # Select only RLM-features, and return if none are present.
    features = set(features).intersection(rlm_features)
    if len(features) == 0:
        return

    # Features are parametrised by the choice of discretisation parameters and spatial methods..
    for discretisation_parameters in get_discretisation_parameters(
        settings=settings
    ):
        for spatial_method in settings.glrlm_spatial_method:
            for feature in features:
                yield class_dict[feature](
                    spatial_method=spatial_method,
                    **discretisation_parameters
                )
