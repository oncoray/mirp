from typing import Generator

import pandas as pd

from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
from mirp._features.base_feature import Feature
from mirp._features.cm_features import generate_cm_features
from mirp._features.rlm_features import generate_rlm_features
from mirp._features.szm_features import generate_szm_features


def generate_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str] = None
) -> Generator[Feature, None, None]:

    # Co-occurrence matrix features.
    yield from generate_cm_features(settings=settings, features=features)

    # Run length matrix features.
    yield from generate_rlm_features(settings=settings, features=features)

    # Size zone matrix features.
    yield from generate_szm_features(settings=settings, features=features)


def feature_to_table(features: list[Feature]) -> pd.DataFrame | None:
    if features is None or len(features) == 0:
        return None

    # Set feature name for feature tables.
    for feature in features:
        feature.create_table_name()

    return pd.DataFrame(dict([(feature.table_name, [feature.value]) for feature in features]))
