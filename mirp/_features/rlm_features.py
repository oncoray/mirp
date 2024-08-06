from functools import lru_cache
from typing import Generator

import numpy as np

from mirp._features.rlm_matrix import MatrixRLM
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.histogram import get_discretisation_parameters
from mirp._features.texture_features import FeatureTexture
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class FeatureRLM(FeatureTexture):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

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
    @lru_cache(maxsize=1)
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
        matrix_list = list(matrix_instance.generate(prototype=MatrixRLM, n_slices=image.image_dimension[0]))
        for matrix in matrix_list:
            matrix.compute(data=data, image_dimension=image.image_dimension)

        # Merge according to the spatial method.
        matrix_list = matrix_instance.merge(matrix_list, prototype=MatrixRLM)

        # Compute additional values from the individual matrices.
        for matrix in matrix_list:
            matrix.set_values_from_matrix()
        print(f"RLM Matrix being cached for: {spatial_method}.")
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

    def create_table_name(self):
        table_elements = (
                self._get_base_table_name_element()
                + self._get_spatial_table_name_element()
                + self._get_discretisation_table_name_element()
        )
        self.table_name = "_".join(table_elements)


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


class FeatureRLMLRE(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - long runs emphasis"
        self.abbr_name = "rlm_lre"
        self.ibsi_id = "W4KF"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.rj.rj * matrix.rj.j ** 2.0) / matrix.n_s


class FeatureRLMLGRE(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - low grey level run emphasis"
        self.abbr_name = "rlm_lgre"
        self.ibsi_id = "V3SW"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.ri.ri / matrix.ri.i ** 2.0) / matrix.n_s


class FeatureRLMHGRE(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - high grey level run emphasis"
        self.abbr_name = "rlm_hgre"
        self.ibsi_id = "G3QZ"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.ri.ri * matrix.ri.i ** 2.0) / matrix.n_s


class FeatureRLMSRLGE(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - short run low grey level emphasis"
        self.abbr_name = "rlm_srlge"
        self.ibsi_id = "HTZT"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.rij.rij / (matrix.rij.i * matrix.rij.j) ** 2.0) / matrix.n_s


class FeatureRLMSRHGE(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - short run high grey level emphasis"
        self.abbr_name = "rlm_srhge"
        self.ibsi_id = "GD3A"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.rij.rij * matrix.rij.i ** 2.0 / matrix.rij.j ** 2.0) / matrix.n_s


class FeatureRLMLRLGE(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - long run low grey level emphasis"
        self.abbr_name = "rlm_lrlge"
        self.ibsi_id = "IVPO"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.rij.rij * matrix.rij.j ** 2.0 / matrix.rij.i ** 2.0) / matrix.n_s


class FeatureRLMLRHGE(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - long run high grey level emphasis"
        self.abbr_name = "rlm_lrhge"
        self.ibsi_id = "3KUM"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.rij.rij * matrix.rij.i ** 2.0 * matrix.rij.j ** 2.0) / matrix.n_s


class FeatureRLMGLNU(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - grey level non-uniformity"
        self.abbr_name = "rlm_glnu"
        self.ibsi_id = "R5YN"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.ri.ri ** 2.0) / matrix.n_s


class FeatureRLMGLNUNorm(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - normalised grey level non-uniformity"
        self.abbr_name = "rlm_glnu_norm"
        self.ibsi_id = "OVBL"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.ri.ri ** 2.0) / matrix.n_s ** 2.0


class FeatureRLMRLNU(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - run length non-uniformity)"
        self.abbr_name = "rlm_rlnu"
        self.ibsi_id = "W92Y"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.rj.rj ** 2.0) / matrix.n_s


class FeatureRLMRLNUNorm(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - normalised run length non-uniformity"
        self.abbr_name = "rlm_rlnu_norm"
        self.ibsi_id = "IC23"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.rj.rj ** 2.0) / matrix.n_s ** 2.0


class FeatureRLMRPerc(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - run percentage"
        self.abbr_name = "rlm_r_perc"
        self.ibsi_id = "9ZK5"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        return 1.0 * matrix.n_s / matrix.n_voxels


class FeatureRLMGLVar(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - grey level variance"
        self.abbr_name = "rlm_gl_var"
        self.ibsi_id = "8CE5"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        mu = np.sum(matrix.rij.rij * matrix.rij.i) / matrix.n_s
        return np.sum((matrix.rij.i - mu) ** 2.0 * matrix.rij.rij) / matrix.n_s


class FeatureRLMRLVar(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - run length variance"
        self.abbr_name = "rlm_rl_var"
        self.ibsi_id = "SXLW"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        mu = np.sum(matrix.rij.rij * matrix.rij.j) / matrix.n_s
        return np.sum((matrix.rij.j - mu) ** 2.0 * matrix.rij.rij) / matrix.n_s


class FeatureRLMRLEntr(FeatureRLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - run entropy"
        self.abbr_name = "rlm_rl_entr"
        self.ibsi_id = "HJ9O"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixRLM) -> float:
        if matrix.is_empty():
            return np.nan
        return - np.sum(matrix.rij.rij * np.log2(matrix.rij.rij / matrix.n_s)) / matrix.n_s


def get_rlm_class_dict() -> dict[str, FeatureRLM]:
    class_dict = {
        "rlm_sre": FeatureRLMSRE,
        "rlm_lre": FeatureRLMLRE,
        "rlm_lgre": FeatureRLMLGRE,
        "rlm_hgre": FeatureRLMHGRE,
        "rlm_srlge": FeatureRLMSRLGE,
        "rlm_srhge": FeatureRLMSRHGE,
        "rlm_lrlge": FeatureRLMLRLGE,
        "rlm_lrhge": FeatureRLMLRHGE,
        "rlm_glnu": FeatureRLMGLNU,
        "rlm_glnu_norm": FeatureRLMGLNUNorm,
        "rlm_rlnu": FeatureRLMRLNU,
        "rlm_rlnu_norm": FeatureRLMRLNUNorm,
        "rlm_r_perc": FeatureRLMRPerc,
        "rlm_gl_var": FeatureRLMGLVar,
        "rlm_rl_var": FeatureRLMRLVar,
        "rlm_rl_entr": FeatureRLMRLEntr
    }

    return class_dict


def generate_rlm_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str] = None
) -> Generator[FeatureRLM, None, None]:
    class_dict = get_rlm_class_dict()
    rlm_features = list(class_dict.keys())

    # Populate features if available.
    if features is None and settings.has_glrlm_family():
        features = rlm_features

    # Terminate early if no features are set, and none are required.
    if features is None:
        return

    # Select only RLM-features, and return if none are present.
    features = [feature for feature in features if feature in rlm_features]
    if len(features) == 0:
        return

    # Features are parametrised by the choice of discretisation parameters and spatial methods.
    for discretisation_parameters in get_discretisation_parameters(
        settings=settings
    ):
        for spatial_method in settings.glrlm_spatial_method:
            for feature in features:
                yield class_dict[feature](
                    spatial_method=spatial_method,
                    **discretisation_parameters
                )
