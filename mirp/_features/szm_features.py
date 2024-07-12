from functools import cache
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
    @cache
    def _get_matrix(
            image: GenericImage,
            mask: BaseMask,
            spatial_method: str
    ) -> list[MatrixSZM]:
        # Represent image and mask as a dataframe.
        data = mask.as_pandas_dataframe(
            image=image,
            intensity_mask=True
        )

        # Instantiate a helper copy of the current class to be able to use class methods without tying the cache to the
        # instance of the original object from which this method is called.
        matrix_instance = MatrixSZM(
            spatial_method=spatial_method
        )

        # Compute the required matrices.
        matrix_list = list(matrix_instance.generate(prototype=MatrixSZM, n_slices=image.image_dimension[0]))
        for matrix in matrix_list:
            matrix.compute(data=data, image_dimension=image.image_dimension)

        # Merge according to the spatial method.
        matrix_list = matrix_instance.merge(matrix_list, prototype=MatrixSZM)

        # Compute additional values from the individual matrices.
        for matrix in matrix_list:
            matrix.set_values_from_matrix()
        print(f"SZM Matrix being cached for: {spatial_method}.")
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
    def _compute(matrix: MatrixSZM):
        raise NotImplementedError("Implement _compute for feature-specific computation.")

    def create_table_name(self):
        table_elements = (
                self._get_base_table_name_element()
                + self._get_spatial_table_name_element()
                + self._get_discretisation_table_name_element()
        )
        self.table_name = "_".join(table_elements)




def get_szm_class_dict() -> dict[str, FeatureSZM]:
    class_dict = {
        "szm_sze": 1,
        "szm_lze": 1,
        "szm_lgze": 1,
        "szm_hgze": 1,
        "szm_szlge": 1,
        "szm_szhge": 1,
        "szm_lzlge": 1,
        "szm_lzhge": 1,
        "szm_glnu": 1,
        "szm_glnu_norm": 1,
        "szm_zsnu": 1,
        "szm_zsnu_norm": 1,
        "szm_z_perc": 1,
        "szm_gl_var": 1,
        "szm_zs_var": 1,
        "szm_zs_entr": 1
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
