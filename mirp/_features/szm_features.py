from functools import lru_cache
from typing import Generator

import numpy as np

from mirp._features.szm_matrix import MatrixSZM
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.histogram import get_discretisation_parameters
from mirp._features.texture_features import FeatureTexture
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class FeatureSZM(FeatureTexture):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Perform close crop for SZM.
        self.cropping_distance = 0.0

    def get_matrix(
            self,
            image: GenericImage,
            mask: BaseMask,
    ) -> list[MatrixSZM]:
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
    ) -> list[MatrixSZM]:
        # Instantiate a helper copy of the current class to be able to use class methods without tying the cache to the
        # instance of the original object from which this method is called.
        matrix_instance = MatrixSZM(
            spatial_method=spatial_method
        )

        # Compute the required matrices.
        matrix_list = list(matrix_instance.generate(prototype=MatrixSZM, n_slices=image.image_dimension[0]))
        for matrix in matrix_list:
            matrix.compute(image=image, mask=mask)

        # Merge according to the spatial method.
        matrix_list = matrix_instance.merge(matrix_list, prototype=MatrixSZM)

        # Compute additional values from the individual matrices.
        for matrix in matrix_list:
            matrix.set_values_from_matrix()

        return matrix_list

    def clear_local_cache(self, other):
        if not isinstance(other, FeatureSZM):
            self._get_matrix.cache_clear()

    def clear_cache(self):
        super().clear_cache()
        self._get_matrix.cache_clear()

    def compute(self, image: GenericImage, mask: BaseMask):
        # Skip processing if input image and/or roi are missing
        if image is None or mask is None:
            return

        # Check if data actually exists
        if image.is_empty() or mask.roi_intensity.is_empty_mask():
            return

        # Compute or retrieve matrices from cache.
        matrices = self.get_matrix(image=image, mask=mask)

        # Compute feature value from matrices, and average over matrices.
        values = [self._compute(matrix=matrix) for matrix in matrices]
        self.value = np.nanmean(values)

    @staticmethod
    def _compute(matrix: MatrixSZM):
        raise NotImplementedError("Implement _compute for feature-specific computation.")

    def create_table_name(self):
        table_elements = (
                self._get_base_table_name_element()
                + self._get_spatial_table_name_element()
                + self._get_discretisation_table_name_element()
        )
        self.table_name = "_".join(table_elements)


class FeatureSZMSmallZoneEmphasis(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - small zone emphasis"
        self.abbr_name = "szm_sze"
        self.ibsi_id = "5QRC"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sj.sj / matrix.sj.j ** 2.0) / matrix.n_s


class FeatureSZMLargeZoneEmphasis(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - large zone emphasis"
        self.abbr_name = "szm_lze"
        self.ibsi_id = "48P8"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sj.sj * matrix.sj.j ** 2.0) / matrix.n_s


class FeatureSZMLowGreyLevelZoneEmphasis(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - low grey level zone emphasis"
        self.abbr_name = "szm_lgze"
        self.ibsi_id = "XMSY"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.si.si / matrix.si.i ** 2.0) / matrix.n_s


class FeatureSZMHighGreyLevelZoneEmphasis(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - high grey level zone emphasis "
        self.abbr_name = "szm_hgze"
        self.ibsi_id = "5GN9"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        return  np.sum(matrix.si.si * matrix.si.i ** 2.0) / matrix.n_s


class FeatureSZMSmallZoneLowGreyLevelEmphasis(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - small zone low grey level emphasis"
        self.abbr_name = "szm_szlge"
        self.ibsi_id = "5RAI"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sij.sij / (matrix.sij.i * matrix.sij.j) ** 2.0) / matrix.n_s


class FeatureSZMSmallZoneHighGreyLevelEmphasis(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - small zone high grey level emphasis"
        self.abbr_name = "szm_szhge"
        self.ibsi_id = "HW1V"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sij.sij * matrix.sij.i ** 2.0 / matrix.sij.j ** 2.0) / matrix.n_s


class FeatureSZMLargeZoneLowGreyLevelEmphasis(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - large zone low grey level emphasis"
        self.abbr_name = "szm_lzlge"
        self.ibsi_id = "YH51"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sij.sij * matrix.sij.j ** 2.0 / matrix.sij.i ** 2.0) / matrix.n_s


class FeatureSZMLargeZoneHighGreyLevelEmphasis(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - large zone high grey level emphasis"
        self.abbr_name = "szm_lzhge"
        self.ibsi_id = "J17V"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sij.sij * matrix.sij.i ** 2.0 * matrix.sij.j ** 2.0) / matrix.n_s


class FeatureSZMGreyLevelNonUniformity(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - grey level non-uniformity"
        self.abbr_name = "szm_glnu"
        self.ibsi_id = "JNSA"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.si.si ** 2.0) / matrix.n_s


class FeatureSZMNormalisedGreyLevelNonUniformity(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - normalised grey level non-uniformity"
        self.abbr_name = "szm_glnu_norm"
        self.ibsi_id = "Y1RO"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.si.si ** 2.0) / matrix.n_s ** 2.0


class FeatureSZMZoneSizeNonUniformity(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - zone size non-uniformity"
        self.abbr_name = "szm_zsnu"
        self.ibsi_id = "4JP3"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sj.sj ** 2.0) / matrix.n_s


class FeatureSZMNormalisedZoneSizeNonUniformity(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - Normalised zone size non-uniformity"
        self.abbr_name = "szm_zsnu_norm"
        self.ibsi_id = "VB3A"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sj.sj ** 2.0) / matrix.n_s ** 2.0


class FeatureSZMZonePercentage(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - zone percentage"
        self.abbr_name = "szm_z_perc"
        self.ibsi_id = "P30P"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        return matrix.n_s / matrix.n_voxels


class FeatureSZMGreyLevelVariance(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - grey level variance"
        self.abbr_name = "szm_gl_var"
        self.ibsi_id = "BYLV"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        mu = np.sum(matrix.sij.sij * matrix.sij.i) / matrix.n_s
        return np.sum((matrix.sij.i - mu) ** 2.0 * matrix.sij.sij) / matrix.n_s


class FeatureSZMZoneSizeVariance(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - zone size variance"
        self.abbr_name = "szm_zs_var"
        self.ibsi_id = "3NSA"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        mu = np.sum(matrix.sij.sij * matrix.sij.j) / matrix.n_s
        return np.sum((matrix.sij.j - mu) ** 2.0 * matrix.sij.sij) / matrix.n_s


class FeatureSZMZoneSizeEntropy(FeatureSZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SZM - zone size entropy"
        self.abbr_name = "szm_zs_entr"
        self.ibsi_id = "GU8N"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixSZM) -> float:
        if matrix.is_empty():
            return np.nan
        return -np.sum(matrix.sij.sij * np.log2(matrix.sij.sij / matrix.n_s)) / matrix.n_s


def get_szm_class_dict() -> dict[str, FeatureSZM]:
    class_dict = {
        "szm_sze": FeatureSZMSmallZoneEmphasis,
        "szm_lze": FeatureSZMLargeZoneEmphasis,
        "szm_lgze": FeatureSZMLowGreyLevelZoneEmphasis,
        "szm_hgze": FeatureSZMHighGreyLevelZoneEmphasis,
        "szm_szlge": FeatureSZMSmallZoneLowGreyLevelEmphasis,
        "szm_szhge": FeatureSZMSmallZoneHighGreyLevelEmphasis,
        "szm_lzlge": FeatureSZMLargeZoneLowGreyLevelEmphasis,
        "szm_lzhge": FeatureSZMLargeZoneHighGreyLevelEmphasis,
        "szm_glnu": FeatureSZMGreyLevelNonUniformity,
        "szm_glnu_norm": FeatureSZMNormalisedGreyLevelNonUniformity,
        "szm_zsnu": FeatureSZMZoneSizeNonUniformity,
        "szm_zsnu_norm": FeatureSZMNormalisedZoneSizeNonUniformity,
        "szm_z_perc": FeatureSZMZonePercentage,
        "szm_gl_var": FeatureSZMGreyLevelVariance,
        "szm_zs_var": FeatureSZMZoneSizeVariance,
        "szm_zs_entr": FeatureSZMZoneSizeEntropy
    }

    return class_dict


def generate_szm_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str] = None
) -> Generator[FeatureSZM, None, None]:
    class_dict = get_szm_class_dict()
    szm_features = list(class_dict.keys())

    # Populate features if available.
    if features is None and settings.has_glszm_family():
        features = szm_features

    # Terminate early if no features are set, and none are required.
    if features is None:
        return

    # Select only SZM-features, and return if none are present.
    features = [feature for feature in features if feature in szm_features]
    if len(features) == 0:
        return

    # Features are parametrised by the choice of discretisation parameters and spatial methods.
    for discretisation_parameters in get_discretisation_parameters(
        settings=settings
    ):
        for spatial_method in settings.glszm_spatial_method:
            for feature in features:
                yield class_dict[feature](
                    spatial_method=spatial_method,
                    **discretisation_parameters
                )
