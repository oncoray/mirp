from functools import cache
from typing import Generator

from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.base_feature import Feature
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class MatrixRLM(DirectionalMatrix):

    def __init__(self):
        pass




class FeatureRLM(Feature, HistogramBased):

    def __init__(
            self,
            spatial_method: str,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.spatial_method = spatial_method.lower()

    @staticmethod
    @cache
    def get_matrix(
            image: GenericImage,
            mask: BaseMask,
            spatial_method: str
    ) -> list[MatrixRLM]:
        pass


class FeatureRLMSRE(FeatureRLM):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RLM - short runs emphasis"
        self.abbr_name = "rlm_sre"
        self.ibsi_id = "22OV"
        self.ibsi_compliant = True

    def compute(self, image: GenericImage, mask: BaseMask):
        matrices = self.get_matrix()


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
