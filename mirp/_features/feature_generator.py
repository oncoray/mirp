from typing import Generator

from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
from mirp._features.base_feature import Feature
from mirp._features.rlm_features import generate_rlm_features


def generate_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str] = None
) -> Generator[Feature, None, None]:

    # Run length matrix features.
    yield from generate_rlm_features(settings=settings, features=features)
