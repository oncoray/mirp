from functools import lru_cache
from typing import Generator

import numpy as np
import pandas as pd

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
        self.distance = int(distance)

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
    @lru_cache(maxsize=1)
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
        matrix_list = list(matrix_instance.generate(prototype=MatrixCM, n_slices=image.image_dimension[0], distance=distance))
        for matrix in matrix_list:
            matrix.compute(data=data, image_dimension=image.image_dimension)

        # Merge according to the spatial method.
        matrix_list = matrix_instance.merge(matrix_list, prototype=MatrixCM, distance=distance)

        # Compute additional values from the individual matrices.
        for matrix in matrix_list:
            matrix.set_values_from_matrix(intensity_range=mask.intensity_range)

        return matrix_list

    def clear_local_cache(self, other):
        if not isinstance(other, FeatureCM):
            self._get_matrix.cache_clear()

    def clear_cache(self):
        super().clear_cache()
        self._get_matrix.cache_clear()

    def compute(self, image: GenericImage, mask: BaseMask):
        # Skip processing if input image and/or roi are missing
        if image is None or mask is None:
            return None

        # Check if data actually exists
        if image.is_empty() or mask.roi_intensity.is_empty_mask():
            return

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
                + ["d" + str(self.distance)]
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


class FeatureCMDifferenceAverage(FeatureCM):
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


class FeatureCMSumVariance(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - sum variance"
        self.abbr_name = "cm_sum_var"
        self.ibsi_id = "OEEB"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        mu = np.sum(matrix.pipj.k * matrix.pipj.pipj)
        return np.sum((matrix.pipj.k - mu) ** 2.0 * matrix.pipj.pipj)


class FeatureCMSumEntropy(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - sum entropy"
        self.abbr_name = "cm_sum_entr"
        self.ibsi_id = "P6QZ1"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return -np.sum(matrix.pipj.pipj * np.log2(matrix.pipj.pipj))


class FeatureCMAngularSecondMoment(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - angular second moment"
        self.abbr_name = "cm_energy"
        self.ibsi_id = "8ZQL"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.pij.pij ** 2.0)


class FeatureCMContrast(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - contrast"
        self.abbr_name = "cm_contrast"
        self.ibsi_id = "ACUI"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum((matrix.pij.i - matrix.pij.j) ** 2.0 * matrix.pij.pij)


class FeatureCMDissimilarity(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - dissimilarity"
        self.abbr_name = "cm_dissimilarity"
        self.ibsi_id = "8S9J"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(np.abs(matrix.pij.i - matrix.pij.j) * matrix.pij.pij)


class FeatureCMInverseDifference(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - inverse difference"
        self.abbr_name = "cm_inv_diff"
        self.ibsi_id = "IB1Z"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.pij.pij / (1.0 + np.abs(matrix.pij.i - matrix.pij.j)))


class FeatureCMNormalisedInverseDifference(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - normalised inverse difference"
        self.abbr_name = "cm_inv_diff_norm"
        self.ibsi_id = "NDRX"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.pij.pij / (1.0 + np.abs(matrix.pij.i - matrix.pij.j) / matrix.n_g))


class FeatureCMInverseDifferenceMoment(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - inverse difference moment"
        self.abbr_name = "cm_inv_diff_mom"
        self.ibsi_id = "WF0Z"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.pij.pij / (1.0 + (matrix.pij.i - matrix.pij.j) ** 2.0))


class FeatureCMNormalisedInverseDifferenceMoment(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - normalised inverse difference moment"
        self.abbr_name = "cm_inv_diff_mom_norm"
        self.ibsi_id = "1QCO"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.pij.pij / (1.0 + (matrix.pij.i - matrix.pij.j) ** 2.0 / matrix.n_g ** 2.0))


class FeatureCMInverseVariance(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - inverse variance"
        self.abbr_name = "cm_inv_var"
        self.ibsi_id = "E8JP"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan

        # Select only off-diagonal elements.
        matrix = matrix.pij[matrix.pij.i != matrix.pij.j]
        if len(matrix) == 0:
            return np.nan

        return np.sum(matrix.pij / (matrix.i - matrix.j) ** 2.0)


class FeatureCMCorrelation(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - correlation"
        self.abbr_name = "cm_corr"
        self.ibsi_id = "NI2N"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan

        var_marg = np.sum((matrix.pi.i - matrix.mu_marg) ** 2.0 * matrix.pi.pi)

        if var_marg == 0.0:
            return 1.0
        else:
            return 1.0 / var_marg * (np.sum(matrix.pij.i * matrix.pij.j * matrix.pij.pij) - matrix.mu_marg ** 2.0)


class FeatureCMAutoCorrelation(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - autocorrelation"
        self.abbr_name = "cm_auto_corr"
        self.ibsi_id = "QWB0"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum(matrix.pij.i * matrix.pij.j * matrix.pij.pij)


class FeatureCMClusterTendency(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - cluster tendency"
        self.abbr_name = "cm_clust_tend"
        self.ibsi_id = "DG8W"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum((matrix.pij.i + matrix.pij.j - 2.0 * matrix.mu_marg) ** 2.0 * matrix.pij.pij)


class FeatureCMClusterShade(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - cluster shade"
        self.abbr_name = "cm_clust_shade"
        self.ibsi_id = "7NFM"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum((matrix.pij.i + matrix.pij.j - 2.0 * matrix.mu_marg) ** 3.0 * matrix.pij.pij)


class FeatureCMClusterProminence(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - cluster prominence"
        self.abbr_name = "cm_clust_prom"
        self.ibsi_id = "AE86"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        return np.sum((matrix.pij.i + matrix.pij.j - 2.0 * matrix.mu_marg) ** 4.0 * matrix.pij.pij)


class FeatureCMInformationCorrelation1(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - first measure of information correlation"
        self.abbr_name = "cm_info_corr1"
        self.ibsi_id = "R8DG"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan
        hxy = - np.sum(matrix.pij.pij * np.log2(matrix.pij.pij))
        hxy_1 = - np.sum(matrix.pij.pij * np.log2(matrix.pij.pi * matrix.pij.pj))
        hx = - np.sum(matrix.pi.pi * np.log2(matrix.pi.pi))
        if len(matrix.pij) == 1 or hx == 0.0:
            return 1.0
        else:
            return (hxy - hxy_1) / hx


class FeatureCMInformationCorrelation2(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - second measure of information correlation"
        self.abbr_name = "cm_info_corr2"
        self.ibsi_id = "JN9H"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan

        hxy = - np.sum(matrix.pij.pij * np.log2(matrix.pij.pij))
        hxy_2 = - np.sum(
            np.tile(matrix.pi.pi, len(matrix.pj))
            * np.repeat(matrix.pj.pj, len(matrix.pi))
            * np.log2(
                np.tile(matrix.pi.pi, len(matrix.pj))
                * np.repeat(matrix.pj.pj, len(matrix.pi))
            )
        )

        if hxy_2 < hxy:
            return 0.0
        else:
            return np.sqrt(1.0 - np.exp(-2.0 * (hxy_2 - hxy)))


class FeatureCMMaximumCorrelationCoefficient(FeatureCM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CM - maximum correlation coefficient"
        self.abbr_name = "cm_mcc"

        # MCC is not part of the IBSI corpus and has no assigned identifier.
        self.ibsi_compliant = False

    @staticmethod
    def _compute(matrix: MatrixCM) -> float:
        if matrix.is_empty():
            return np.nan

        # MCC is defined as the square root of the second largest eigen value of Q (Haralick, R.M., Shanmugam,
        # K. and Dinstein, I. (1973) ‘Textural Features for Image Classification’, IEEE transactions on systems, man,
        # and cybernetics, SMC-3(6), pp. 610–621. Available at: https://doi.org/10.1109/TSMC.1973.4309314.)

        # Special case: matrix with a single element.
        if matrix.n_g == 1.0:
            return 1.0

        # Initialise matrix for q.
        q_mat = np.zeros((int(matrix.n_g), int(matrix.n_g)), dtype=float)

        # Initialise full matrix for pij, pi and pj and fill. For pi and pj use 1s as default value to prevent division
        # by 0.
        pij = np.zeros((int(matrix.n_g), int(matrix.n_g)), dtype=float)
        pij[(matrix.pij.i.values.astype(int) - 1, matrix.pij.j.values.astype(int) - 1)] = matrix.pij.pij.values
        pi = np.ones((int(matrix.n_g)), dtype=float)
        pi[(matrix.pi.i.values.astype(int) - 1)] = matrix.pi.pi.values
        pj = np.ones((int(matrix.n_g)), dtype=float)
        pj[(matrix.pj.j.values.astype(int) - 1)] = matrix.pj.pj.values

        non_zero_elems = np.nonzero(pij)
        for kk in np.arange(len(non_zero_elems[0])):
            i = non_zero_elems[0][kk]
            j = non_zero_elems[1][kk]

            # sum of p(i,k) * p(j,k) / (px(i), py(k))
            q_mat[i,j] = np.sum(np.divide(np.multiply(pij[i, :], pij[j, :]), pi[i] * pj))

        # Compute eigen values and sort.
        eigen_values = np.linalg.eigvals(q_mat)
        eigen_values.sort()

        # Select second largest eigenvalue, and set minimum value to 0.
        second_largest_eigen_value = np.max([0.0, eigen_values[-2]])

        # Return square root of second largest eigenvalue.
        return np.sqrt(second_largest_eigen_value)


def get_cm_class_dict() -> dict[str, FeatureCM]:
    class_dict = {
        "cm_joint_max": FeatureCMJointMax,
        "cm_joint_avg": FeatureCMJointAverage,
        "cm_joint_var": FeatureCMJointVariance,
        "cm_joint_entr": FeatureCMJointEntropy,
        "cm_diff_avg": FeatureCMDifferenceAverage,
        "cm_diff_var": FeatureCMDifferenceVariance,
        "cm_diff_entr": FeatureCMDifferenceEntropy,
        "cm_sum_avg": FeatureCMSumAverage,
        "cm_sum_var": FeatureCMSumVariance,
        "cm_sum_entr": FeatureCMSumEntropy,
        "cm_energy": FeatureCMAngularSecondMoment,
        "cm_contrast": FeatureCMContrast,
        "cm_dissimilarity": FeatureCMDissimilarity,
        "cm_inv_diff": FeatureCMInverseDifference,
        "cm_inv_diff_norm": FeatureCMNormalisedInverseDifference,
        "cm_inv_diff_mom": FeatureCMInverseDifferenceMoment,
        "cm_inv_diff_mom_norm": FeatureCMNormalisedInverseDifferenceMoment,
        "cm_inv_var": FeatureCMInverseVariance,
        "cm_corr": FeatureCMCorrelation,
        "cm_auto_corr": FeatureCMAutoCorrelation,
        "cm_clust_tend": FeatureCMClusterTendency,
        "cm_clust_shade": FeatureCMClusterShade,
        "cm_clust_prom": FeatureCMClusterProminence,
        "cm_info_corr1": FeatureCMInformationCorrelation1,
        "cm_info_corr2": FeatureCMInformationCorrelation2,
        "cm_mcc": FeatureCMMaximumCorrelationCoefficient
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
