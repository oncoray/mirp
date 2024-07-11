from functools import cache
from typing import Generator

import numpy as np

from mirp._features.cm_matrix import MatrixCM
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.histogram import get_discretisation_parameters
from mirp._features.texture_features import FeatureTexture
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class FeatureCM(FeatureTexture):

    def __init__(
            self,
            distance: int,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Lookup distance (in voxels)
        self.distance = distance

        # Features are always computed from symmetric co-occurrence matrices.
        self.symmetric_matrix = True

        # Perform close crop for CM.
        self.cropping_distance = 0.0

    def get_matrix(
            self,
            image: GenericImage,
            mask: BaseMask,
    ) -> list[MatrixCM]:
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
            spatial_method=self.spatial_method,
            distance=self.distance
        )

        return matrix_list

    @staticmethod
    @cache
    def _get_matrix(
            image: GenericImage,
            mask: BaseMask,
            spatial_method: str,
            distance: int
    ) -> list[MatrixCM]:
        # Represent image and mask as a dataframe.
        data = mask.as_pandas_dataframe(
            image=image,
            intensity_mask=True
        )

        # Instantiate a helper copy of the current class to be able to use class methods without tying the cache to the
        # instance of the original object from which this method is called.
        matrix_instance = MatrixCM(
            spatial_method=spatial_method,
            distance=distance
        )

        # Compute the required matrices.
        matrix_list = list(matrix_instance.generate(prototype=MatrixCM, n_slices=image.image_dimension[0]))
        for matrix in matrix_list:
            matrix.compute(data=data, image_dimension=image.image_dimension)

        # Merge according to the spatial method.
        matrix_list = matrix_instance.merge(matrix_list, prototype=MatrixCM)

        # Compute additional values from the individual matrices.
        for matrix in matrix_list:
            matrix.set_values_from_matrix(intensity_range=mask.intensity_range)
        print(f"CM Matrix being cached for: {spatial_method}.")
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
    def _compute(matrix: MatrixCM):
        raise NotImplementedError("Implement _compute for feature-specific computation.")

    def create_table_name(self):
        table_elements = (
                self._get_base_table_name_element()
                + ["d" + str(np.round(self.distance, 1))]
                + self._get_spatial_table_name_element()
                + self._get_discretisation_table_name_element()
        )
        self.table_name = "_".join(table_elements)


class FeatureCMJointMax(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - joint maximum"
        self.abbr_name = "cm_joint_max"
        self.ibsi_id = "GYBY"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.max(matrix.pij.pij)


class FeatureCMJointAverage(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - joint average"
        self.abbr_name = "cm_joint_avg"
        self.ibsi_id = "60VM"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return matrix.mu


class FeatureCMJointVariance(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - joint variance"
        self.abbr_name = "cm_joint_var"
        self.ibsi_id = "UR99"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum((matrix.pij.i - matrix.mu) ** 2.0 * matrix.pij.pij)


class FeatureCMJointEntropy(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - joint entropy"
        self.abbr_name = "cm_joint_entr"
        self.ibsi_id = "TU9B"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return -np.sum(matrix.pij.pij * np.log2(matrix.pij.pij))


class FeatureCMDifferenceAveraga(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - difference average"
        self.abbr_name = "cm_diff_avg"
        self.ibsi_id = "TF7R"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.pimj.k * matrix.pimj.pimj)


class FeatureCMDifferenceVariance(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - difference variance"
        self.abbr_name = "cm_diff_var"
        self.ibsi_id = "D3YU"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        mu = np.sum(matrix.pimj.k * matrix.pimj.pimj)
        return np.sum((matrix.pimj.k - mu) ** 2.0 * matrix.pimj.pimj)


class FeatureCMDifferenceEntropy(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - difference entropy"
        self.abbr_name = "cm_diff_entr"
        self.ibsi_id = "NTRS"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return -np.sum(matrix.pimj.pimj * np.log2(matrix.pimj.pimj))


class FeatureCMSumAverage(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - sum average"
        self.abbr_name = "cm_sum_avg"
        self.ibsi_id = "ZGXS"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.pipj.k * matrix.pipj.pipj)


def get_cm_class_dict() -> dict[str, FeatureCM]:
    class_dict = {
        "cm_joint_max": FeatureCMJointMax,
        "cm_joint_avg": FeatureCMJointAverage,
        "cm_joint_var": FeatureCMJointVariance,
        "cm_joint_entr": FeatureCMJointEntropy,
        "cm_diff_avg": FeatureCMDifferenceAveraga,
        "cm_diff_var": FeatureCMDifferenceVariance,
        "cm_diff_entr": FeatureCMDifferenceEntropy,
        "cm_sum_avg": FeatureCMSumAverage,
        "cm_sum_var": 1,
        "cm_sum_entr": 1,
        "cm_energy": 1,
        "cm_contrast": 1,
        "cm_dissimilarity": 1,
        "cm_inv_diff": 1,
        "cm_inv_diff_norm": 1,
        "cm_inv_diff_mom": 1,
        "cm_inv_diff_mom_norm": 1,
        "cm_inv_var": 1,
        "cm_corr": 1,
        "cm_auto_corr": 1,
        "cm_clust_tend": 1,
        "cm_clust_shade": 1,
        "cm_clust_prom": 1,
        "cm_info_corr1": 1,
        "cm_info_corr2": 1
    }

    return class_dict


def generate_cm_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str] = None
) -> Generator[FeatureCM, None, None]:
    class_dict = get_cm_class_dict()
    cm_features = list(class_dict.keys())

    # Populate features if available.
    if features is None and settings.has_glcm_family():
        features = cm_features

    # Terminate early if no features are set, and none are required.
    if features is None:
        return

    # Select only CM-features, and return if none are present.
    features = [feature for feature in features if feature in cm_features]
    if len(features) == 0:
        return

    # Features are parametrised by the choice of discretisation parameters, spatial methods and distances.
    for discretisation_parameters in get_discretisation_parameters(
        settings=settings
    ):
        for spatial_method in settings.glcm_spatial_method:
            for distance in settings.glcm_distance:
                for feature in features:
                    yield class_dict[feature](
                        spatial_method=spatial_method,
                        distance=distance,
                        **discretisation_parameters
                    )
