from functools import cache
from typing import Generator

import numpy as np

from mirp._features.ngldm_matrix import MatrixNGLDM
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.histogram import get_discretisation_parameters
from mirp._features.texture_features import FeatureTexture
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class FeatureNGLDM(FeatureTexture):

    def __init__(
            self,
            distance: float,
            coarseness: float,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Lookup distance (in voxels)
        self.distance = float(distance)

        # Coarseness
        self.coarseness = coarseness

        # Features are always computed from symmetric co-occurrence matrices.
        self.symmetric_matrix = True

        # Perform close crop for CM.
        self.cropping_distance = 0.0

    def get_matrix(
            self,
            image: GenericImage,
            mask: BaseMask,
    ) -> list[MatrixNGLDM]:
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
            distance=self.distance,
            coarseness=self.coarseness
        )

        return matrix_list

    @staticmethod
    @cache
    def _get_matrix(
            image: GenericImage,
            mask: BaseMask,
            spatial_method: str,
            distance: float,
            coarseness: float
    ) -> list[MatrixNGLDM]:
        # Represent image and mask as a dataframe.
        data = mask.as_pandas_dataframe(
            image=image,
            intensity_mask=True
        )

        # Instantiate a helper copy of the current class to be able to use class methods without tying the cache to the
        # instance of the original object from which this method is called.
        matrix_instance = MatrixNGLDM(
            spatial_method=spatial_method,
            distance=distance,
            coarseness=coarseness
        )

        # Compute the required matrices.
        matrix_list = list(matrix_instance.generate(
            prototype=MatrixNGLDM,
            n_slices=image.image_dimension[0],
            distance=distance,
            coarseness=coarseness
        ))
        for matrix in matrix_list:
            matrix.compute(data=data, image_dimension=image.image_dimension)

        # Merge according to the spatial method.
        matrix_list = matrix_instance.merge(
            matrix_list,
            prototype=MatrixNGLDM,
            distance=distance,
            coarseness=coarseness
        )

        # Compute additional values from the individual matrices.
        for matrix in matrix_list:
            matrix.set_values_from_matrix()
        print(f"NGLDM Matrix being cached for: {spatial_method}.")
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
    def _compute(matrix: MatrixNGLDM):
        raise NotImplementedError("Implement _compute for feature-specific computation.")

    def create_table_name(self):
        table_elements = (
                self._get_base_table_name_element()
                + ["d" + str(int(np.round(self.distance, decimals=0)))]
                + ["a" + str(np.round(self.coarseness, decimals=0))]
                + self._get_spatial_table_name_element()
                + self._get_discretisation_table_name_element()
        )
        self.table_name = "_".join(table_elements)


class FeatureNGLDMLowDependenceEmphasis(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - low dependence emphasis"
        self.abbr_name = "ngl_lde"
        self.ibsi_id = "SODN"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sj.sj / matrix.sj.j ** 2.0) / matrix.n_s


class FeatureNGLDMHighDependenceEmphasis(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - high dependence emphasis"
        self.abbr_name = "ngl_hde"
        self.ibsi_id = "IMOQ"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sj.sj * matrix.sj.j ** 2.0) / matrix.n_s


class FeatureNGLDMLowGreyLevelCountEmphasis(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - low grey level count emphasis"
        self.abbr_name = "ngl_lgce"
        self.ibsi_id = "TL9H"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.si.si / matrix.si.i ** 2.0) / matrix.n_s


class FeatureNGLDMHighGreyLevelCountEmphasis(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - high grey level count emphasis"
        self.abbr_name = "ngl_hgce"
        self.ibsi_id = "OAE7"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.si.si * matrix.si.i ** 2.0) / matrix.n_s


class FeatureNGLDMLowDependenceLowGreyLevelEmphasis(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - ow dependence low grey level emphasis"
        self.abbr_name = "ngl_ldlge"
        self.ibsi_id = "EQ3F"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sij.sij / (matrix.sij.i * matrix.sij.j) ** 2.0) / matrix.n_s


class FeatureNGLDMLowDependenceHighGreyLevelEmphasis(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - low dependence high grey level emphasis"
        self.abbr_name = "ngl_ldhge"
        self.ibsi_id = "JA6D"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sij.sij * matrix.sij.i ** 2.0 / matrix.sij.j ** 2.0) / matrix.n_s


class FeatureNGLDMHighDependenceLowGreyLevelEmphasis(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - high dependence low grey level emphasis"
        self.abbr_name = "ngl_hdlge"
        self.ibsi_id = "NBZI"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sij.sij * matrix.sij.j ** 2.0 / matrix.sij.i ** 2.0) / matrix.n_s


class FeatureNGLDMHighDependenceHighGreyLevelEmphasis(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - high dependence high grey level emphasis"
        self.abbr_name = "ngl_hdhge"
        self.ibsi_id = "9QMG"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sij.sij * matrix.sij.i ** 2.0 * matrix.sij.j ** 2.0) / matrix.n_s


class FeatureNGLDMGreyLevelNonUniformity(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - grey level non-uniformity"
        self.abbr_name = "ngl_glnu"
        self.ibsi_id = "FP8K"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.si.si ** 2.0) / matrix.n_s


class FeatureNGLDMNormalisedGreyLevelNonUniformity(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - normalised grey level non-uniformity"
        self.abbr_name = "ngl_glnu_norm"
        self.ibsi_id = "5SPA"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.si.si ** 2.0) / matrix.n_s ** 2.0


class FeatureNGLDMDependenceCountNonUniformity(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - dependence count non-uniformity"
        self.abbr_name = "ngl_dcnu"
        self.ibsi_id = "Z87G"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sj.sj ** 2.0) / matrix.n_s


class FeatureNGLDMNormalisedDependenceCountNonUniformity(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - normalised dependence count non-uniformity"
        self.abbr_name = "ngl_dcnu_norm"
        self.ibsi_id = "OKJI"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sj.sj ** 2.0) / matrix.n_s ** 2.0


class FeatureNGLDMDependenceCountPercentage(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - dependence count percentage"
        self.abbr_name = "ngl_dc_perc"
        self.ibsi_id = "6XV8"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return matrix.n_s / matrix.n_voxels


class FeatureNGLDMGreyLevelVariance(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - grey level variance"
        self.abbr_name = "ngl_gl_var"
        self.ibsi_id = "1PFV"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        mu = np.sum(matrix.sij.sij * matrix.sij.i) / matrix.n_s
        return np.sum((matrix.sij.i - mu) ** 2.0 * matrix.sij.sij) / matrix.n_s


class FeatureNGLDMDependenceCountVariance(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - dependence count variance"
        self.abbr_name = "ngl_dc_var"
        self.ibsi_id = "DNX2"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        mu = np.sum(matrix.sij.sij * matrix.sij.j) / matrix.n_s
        return np.sum((matrix.sij.j - mu) ** 2.0 * matrix.sij.sij) / matrix.n_s


class FeatureNGLDMDependenceCountEntropy(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - dependence count entropy"
        self.abbr_name = "ngl_dc_entr"
        self.ibsi_id = "FCBV"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return -np.sum(matrix.sij.sij * np.log2(matrix.sij.sij / matrix.n_s)) / matrix.n_s


class FeatureNGLDMDependenceCountEnergy(FeatureNGLDM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NGLDM - dependence count energy"
        self.abbr_name = "ngl_dc_energy"
        self.ibsi_id = "CAS9"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixNGLDM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.sij.sij ** 2.0) / (matrix.n_s ** 2.0)


def get_ngldm_class_dict() -> dict[str, FeatureNGLDM]:
    class_dict = {
        "ngl_lde": FeatureNGLDMLowDependenceEmphasis,
        "ngl_hde": FeatureNGLDMHighDependenceEmphasis,
        "ngl_lgce": FeatureNGLDMLowGreyLevelCountEmphasis,
        "ngl_hgce": FeatureNGLDMHighGreyLevelCountEmphasis,
        "ngl_ldlge": FeatureNGLDMLowDependenceLowGreyLevelEmphasis,
        "ngl_ldhge": FeatureNGLDMLowDependenceHighGreyLevelEmphasis,
        "ngl_hdlge": FeatureNGLDMHighDependenceLowGreyLevelEmphasis,
        "ngl_hdhge": FeatureNGLDMHighDependenceHighGreyLevelEmphasis,
        "ngl_glnu": FeatureNGLDMGreyLevelNonUniformity,
        "ngl_glnu_norm": FeatureNGLDMNormalisedGreyLevelNonUniformity,
        "ngl_dcnu": FeatureNGLDMDependenceCountNonUniformity,
        "ngl_dcnu_norm": FeatureNGLDMNormalisedDependenceCountNonUniformity,
        "ngl_dc_perc": FeatureNGLDMDependenceCountPercentage,
        "ngl_gl_var": FeatureNGLDMGreyLevelVariance,
        "ngl_dc_var": FeatureNGLDMDependenceCountVariance,
        "ngl_dc_entr": FeatureNGLDMDependenceCountEntropy,
        "ngl_dc_energy": FeatureNGLDMDependenceCountEnergy
    }

    return class_dict


def generate_ngldm_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str] = None
) -> Generator[FeatureNGLDM, None, None]:
    class_dict = get_ngldm_class_dict()
    ngldm_features = list(class_dict.keys())

    # Populate features if available.
    if features is None and settings.has_glcm_family():
        features = ngldm_features

    # Terminate early if no features are set, and none are required.
    if features is None:
        return

    # Select only NGLDM-features, and return if none are present.
    features = [feature for feature in features if feature in ngldm_features]
    if len(features) == 0:
        return

    # Features are parametrised by the choice of discretisation parameters, spatial methods, distance and coarseness.
    for discretisation_parameters in get_discretisation_parameters(
        settings=settings
    ):
        for spatial_method in settings.ngldm_spatial_method:
            for distance in settings.ngldm_dist:
                for coarseness in settings.ngldm_diff_lvl:
                    for feature in features:
                        yield class_dict[feature](
                            spatial_method=spatial_method,
                            distance=distance,
                            coarseness=coarseness,
                            **discretisation_parameters
                        )
