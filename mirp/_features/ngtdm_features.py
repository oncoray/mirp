from functools import lru_cache
from typing import Generator

import numpy as np

from mirp._features.ngtdm_matrix import MatrixNGTDM
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.histogram import get_discretisation_parameters
from mirp._features.texture_features import FeatureTexture
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class FeatureNGTDM(FeatureTexture):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Perform close crop for NGTDM.
        self.cropping_distance = 0.0

    def get_matrix(
            self,
            image: GenericImage,
            mask: BaseMask,
    ) -> list[MatrixNGTDM]:
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
    @lru_cache(maxsize=1)
    def _get_matrix(
            image: GenericImage,
            mask: BaseMask,
            spatial_method: str
    ) -> list[MatrixNGTDM]:
        # Represent image and mask as a dataframe.
        data = mask.as_pandas_dataframe(
            image=image,
            intensity_mask=True
        )

        # Instantiate a helper copy of the current class to be able to use class methods without tying the cache to the
        # instance of the original object from which this method is called.
        matrix_instance = MatrixNGTDM(
            spatial_method=spatial_method
        )

        # Compute the required matrices.
        matrix_list = list(matrix_instance.generate(prototype=MatrixNGTDM, n_slices=image.image_dimension[0]))
        for matrix in matrix_list:
            matrix.compute(data=data, image_dimension=image.image_dimension)

        # Merge according to the spatial method.
        matrix_list = matrix_instance.merge(matrix_list, prototype=MatrixNGTDM)

        # Compute additional values from the individual matrices.
        for matrix in matrix_list:
            matrix.set_values_from_matrix(intensity_range=mask.intensity_range)

        return matrix_list

    def clear_local_cache(self, other):
        if not isinstance(other, FeatureNGTDM):
            self._get_matrix.cache_clear()

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
    def _compute(matrix: MatrixNGTDM):
        raise NotImplementedError("Implement _compute for feature-specific computation.")

    def create_table_name(self):
        table_elements = (
                self._get_base_table_name_element()
                + self._get_spatial_table_name_element()
                + self._get_discretisation_table_name_element()
        )
        self.table_name = "_".join(table_elements)


class FeatureNGTDMCoarseness(FeatureNGTDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGTDM - coarseness"
        self.abbr_name = "ngt_coarseness"
        self.ibsi_id = "QCDE"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGTDM) -> float:
        if matrix.is_empty():
            return np.nan

        x = np.sum(matrix.pi.pi * matrix.pi.s)
        if x < 1.0E-6:
            return 1.0 / 1.0E-6
        else:
            return 1.0 / x


class FeatureNGTDMContrast(FeatureNGTDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGTDM - contrast"
        self.abbr_name = "ngt_contrast"
        self.ibsi_id = "65HE"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGTDM) -> float:
        if matrix.is_empty():
            return np.nan

        if matrix.n_p > 1.0:
            return (
                    np.sum(matrix.pij.pi * matrix.pij.pj * (matrix.pij.i - matrix.pij.j) ** 2.0)
                    / (matrix.n_p * (matrix.n_p - 1.0)) * np.sum(matrix.pi.s) / matrix.n_voxels
            )
        else:
            return 0.0


class FeatureNGTDMBusyness(FeatureNGTDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGTDM - busyness"
        self.abbr_name = "ngt_busyness"
        self.ibsi_id = "NQ30"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGTDM) -> float:
        if matrix.is_empty():
            return np.nan

        denominator = np.sum(np.abs(matrix.pij.i * matrix.pij.pi - matrix.pij.j * matrix.pij.pj))
        if matrix.n_p > 1.0 and denominator > 0.0:
            return np.sum(matrix.pi.pi * matrix.pi.s) / denominator
        else:
            return 0.0


class FeatureNGTDMComplexity(FeatureNGTDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGTDM - complexity"
        self.abbr_name = "ngt_complexity"
        self.ibsi_id = "HDEZ"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGTDM) -> float:
        if matrix.is_empty():
            return np.nan

        return np.sum(
            np.abs(matrix.pij.i - matrix.pij.j)
            * (matrix.pij.pi * matrix.pij.si + matrix.pij.pj * matrix.pij.sj)
            / (matrix.pij.pi + matrix.pij.pj)
        ) / matrix.n_voxels


class FeatureNGTDMStrength(FeatureNGTDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGTDM - strength"
        self.abbr_name = "ngt_strength"
        self.ibsi_id = "1X9X"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGTDM) -> float:
        if matrix.is_empty():
            return np.nan

        denominator = np.sum(matrix.pi.s)
        if denominator > 0.0:
            return np.sum((matrix.pij.pi + matrix.pij.pj) * (matrix.pij.i - matrix.pij.j) ** 2.0) / denominator
        else:
            return 0.0


def get_ngtdm_class_dict() -> dict[str, FeatureNGTDM]:
    class_dict = {
        "ngt_coarseness": FeatureNGTDMCoarseness,
        "ngt_contrast": FeatureNGTDMContrast,
        "ngt_busyness": FeatureNGTDMBusyness,
        "ngt_complexity": FeatureNGTDMComplexity,
        "ngt_strength": FeatureNGTDMStrength
    }

    return class_dict


def generate_ngtdm_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str] = None
) -> Generator[FeatureNGTDM, None, None]:
    class_dict = get_ngtdm_class_dict()
    ngt_features = list(class_dict.keys())

    # Populate features if available.
    if features is None and settings.has_ngtdm_family():
        features = ngt_features

    # Terminate early if no features are set, and none are required.
    if features is None:
        return

    # Select only NGTDM-features, and return if none are present.
    features = [feature for feature in features if feature in ngt_features]
    if len(features) == 0:
        return

    # Features are parametrised by the choice of discretisation parameters and spatial methods.
    for discretisation_parameters in get_discretisation_parameters(
        settings=settings
    ):
        for spatial_method in settings.ngtdm_spatial_method:
            for feature in features:
                yield class_dict[feature](
                    spatial_method=spatial_method,
                    **discretisation_parameters
                )
