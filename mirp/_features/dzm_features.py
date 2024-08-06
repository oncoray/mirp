from functools import lru_cache
from typing import Generator

import numpy as np

from mirp._features.dzm_matrix import MatrixDZM
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.histogram import get_discretisation_parameters
from mirp._features.texture_features import FeatureTexture
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class FeatureDZM(FeatureTexture):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Perform close crop for DZM.
        self.cropping_distance = 0.0

    def get_matrix(
            self,
            image: GenericImage,
            mask: BaseMask,
    ) -> list[MatrixDZM]:
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
    ) -> list[MatrixDZM]:
        # Extract data from the mask and image.
        data = mask.as_pandas_dataframe(
            image=image,
            intensity_mask=True,
            distance_map=True,
            by_slice=spatial_method in ["2d", "2.5d"]
        )

        # Instantiate a helper copy of the current class to be able to use class methods without tying the cache to the
        # instance of the original object from which this method is called.
        matrix_instance = MatrixDZM(
            spatial_method=spatial_method
        )

        # Compute the required matrices.
        matrix_list = list(matrix_instance.generate(prototype=MatrixDZM, n_slices=image.image_dimension[0]))
        for matrix in matrix_list:
            matrix.compute(data=data, image=image, mask=mask)

        # Merge according to the spatial method.
        matrix_list = matrix_instance.merge(matrix_list, prototype=MatrixDZM)

        # Compute additional values from the individual matrices.
        for matrix in matrix_list:
            matrix.set_values_from_matrix()

        return matrix_list

    def clear_local_cache(self, other):
        if not isinstance(other, FeatureDZM):
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
    def _compute(matrix: MatrixDZM):
        raise NotImplementedError("Implement _compute for feature-specific computation.")

    def create_table_name(self):
        table_elements = (
                self._get_base_table_name_element()
                + self._get_spatial_table_name_element()
                + self._get_discretisation_table_name_element()
        )
        self.table_name = "_".join(table_elements)


class FeatureDZMSmallDistanceEmphasis(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - small distance emphasis"
        self.abbr_name = "dzm_sde"
        self.ibsi_id = "0GBI"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.dj.dj / matrix.dj.j ** 2.0) / matrix.n_s


class FeatureDZMLargeDistanceEmphasis(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - large distance emphasis"
        self.abbr_name = "dzm_lde"
        self.ibsi_id = "MB4I"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.dj.dj * matrix.dj.j ** 2.0) / matrix.n_s


class FeatureDZMLowGreyLevelZoneEmphasis(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - low grey level zone emphasis"
        self.abbr_name = "dzm_lgze"
        self.ibsi_id = "S1RA"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.di.di / matrix.di.i ** 2.0) / matrix.n_s


class FeatureDZMHighGreyLevelZoneEmphasis(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - high grey level zone emphasis"
        self.abbr_name = "dzm_hgze"
        self.ibsi_id = "K26C"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.di.di * matrix.di.i ** 2.0) / matrix.n_s


class FeatureDZMSmallDistanceLowGreyLevelEmphasis(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - small distance low grey level emphasis"
        self.abbr_name = "dzm_sdlge"
        self.ibsi_id = "RUVG"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.dij.dij / (matrix.dij.i * matrix.dij.j) ** 2.0) / matrix.n_s


class FeatureDZMSmallDistanceHighGreyLevelEmphasis(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - small distance high grey level emphasis"
        self.abbr_name = "dzm_sdhge"
        self.ibsi_id = "DKNJ"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.dij.dij * matrix.dij.i ** 2.0 / matrix.dij.j ** 2.0) / matrix.n_s


class FeatureDZMLargeDistanceLowGreyLevelEmphasis(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - large distance low grey level emphasis"
        self.abbr_name = "dzm_ldlge"
        self.ibsi_id = "A7WM"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.dij.dij * matrix.dij.j ** 2.0 / matrix.dij.i ** 2.0) / matrix.n_s


class FeatureDZMLargeDistanceHighGreyLevelEmphasis(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - large distance high grey level emphasis"
        self.abbr_name = "dzm_ldhge"
        self.ibsi_id = "KLTH"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.dij.dij * matrix.dij.i ** 2.0 * matrix.dij.j ** 2.0) / matrix.n_s


class FeatureDZMGreyLevelNonUniformity(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - grey level non-uniformity"
        self.abbr_name = "dzm_glnu"
        self.ibsi_id = "VFT7"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.di.di ** 2.0) / matrix.n_s


class FeatureDZMNormalisedGreyLevelNonUniformity(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - normalised grey level non-uniformity"
        self.abbr_name = "dzm_glnu_norm"
        self.ibsi_id = "7HP3"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.di.di ** 2.0) / matrix.n_s ** 2.0


class FeatureDZMZoneDistanceNonUniformity(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - zone distance non-uniformity"
        self.abbr_name = "dzm_zdnu"
        self.ibsi_id = "V294"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.dj.dj ** 2.0) / matrix.n_s


class FeatureDZMNormalisedZoneDistanceNonUniformity(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - normalised zone distance non-uniformity"
        self.abbr_name = "dzm_zdnu_norm"
        self.ibsi_id = "IATH"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.dj.dj ** 2.0) / matrix.n_s ** 2.0


class FeatureDZMZonePercentage(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - zone percentage"
        self.abbr_name = "dzm_z_perc"
        self.ibsi_id = "VIWW"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        return matrix.n_s / matrix.n_voxels


class FeatureDZMGreyLevelVariance(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - grey level variance"
        self.abbr_name = "dzm_gl_var"
        self.ibsi_id = "QK93"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        mu = np.sum(matrix.dij.dij * matrix.dij.i) / matrix.n_s
        return np.sum((matrix.dij.i - mu) ** 2.0 * matrix.dij.dij) / matrix.n_s


class FeatureDZMZoneDistanceVariance(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - zone distance variance"
        self.abbr_name = "dzm_zd_var"
        self.ibsi_id = "7WT1"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        mu = np.sum(matrix.dij.dij * matrix.dij.j) / matrix.n_s
        return np.sum((matrix.dij.j - mu) ** 2.0 * matrix.dij.dij) / matrix.n_s


class FeatureDZMZoneDistanceEntropy(FeatureDZM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DZM - zone distance entropy"
        self.abbr_name = "dzm_zd_entr"
        self.ibsi_id = "GBDU"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixDZM) -> float:
        if matrix.is_empty():
            return np.nan
        return -np.sum(matrix.dij.dij * np.log2(matrix.dij.dij / matrix.n_s)) / matrix.n_s


def get_dzm_class_dict() -> dict[str, FeatureDZM]:
    class_dict = {
        "dzm_sde": FeatureDZMSmallDistanceEmphasis,
        "dzm_lde": FeatureDZMLargeDistanceEmphasis,
        "dzm_lgze": FeatureDZMLowGreyLevelZoneEmphasis,
        "dzm_hgze": FeatureDZMHighGreyLevelZoneEmphasis,
        "dzm_sdlge": FeatureDZMSmallDistanceLowGreyLevelEmphasis,
        "dzm_sdhge": FeatureDZMSmallDistanceHighGreyLevelEmphasis,
        "dzm_ldlge": FeatureDZMLargeDistanceLowGreyLevelEmphasis,
        "dzm_ldhge": FeatureDZMLargeDistanceHighGreyLevelEmphasis,
        "dzm_glnu": FeatureDZMGreyLevelNonUniformity,
        "dzm_glnu_norm": FeatureDZMNormalisedGreyLevelNonUniformity,
        "dzm_zdnu": FeatureDZMZoneDistanceNonUniformity,
        "dzm_zdnu_norm": FeatureDZMNormalisedZoneDistanceNonUniformity,
        "dzm_z_perc": FeatureDZMZonePercentage,
        "dzm_gl_var": FeatureDZMGreyLevelVariance,
        "dzm_zd_var": FeatureDZMZoneDistanceVariance,
        "dzm_zd_entr": FeatureDZMZoneDistanceEntropy
    }

    return class_dict


def generate_dzm_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str] = None
) -> Generator[FeatureDZM, None, None]:
    class_dict = get_dzm_class_dict()
    dzm_features = list(class_dict.keys())

    # Populate features if available.
    if features is None and settings.has_gldzm_family():
        features = dzm_features

    # Terminate early if no features are set, and none are required.
    if features is None:
        return

    # Select only DZM-features, and return if none are present.
    features = [feature for feature in features if feature in dzm_features]
    if len(features) == 0:
        return

    # Features are parametrised by the choice of discretisation parameters and spatial methods.
    for discretisation_parameters in get_discretisation_parameters(
        settings=settings
    ):
        for spatial_method in settings.gldzm_spatial_method:
            for feature in features:
                yield class_dict[feature](
                    spatial_method=spatial_method,
                    **discretisation_parameters
                )
