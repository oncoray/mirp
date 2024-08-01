from typing import Generator

import pandas as pd

from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
from mirp._features.base_feature import Feature
from mirp._features.morph_3d_features import generate_morph_3d_features
from mirp._features.local_intensity_features import generate_local_intensity_features
from mirp._features.stat_features import generate_stat_features
from mirp._features.ih_features import generate_ih_features
from mirp._features.cm_features import generate_cm_features
from mirp._features.dzm_features import generate_dzm_features
from mirp._features.rlm_features import generate_rlm_features
from mirp._features.szm_features import generate_szm_features
from mirp._features.ngtdm_features import generate_ngtdm_features
from mirp._features.ngldm_features import generate_ngldm_features


def generate_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str] = None
) -> Generator[Feature, None, None]:

    # Morphological features
    yield from generate_morph_3d_features(settings=settings, features=features)

    # Local intensity features
    yield from generate_local_intensity_features(settings=settings, features=features)

    # Statistical features
    yield from generate_stat_features(settings=settings, features=features)

    # Intensity histogram features
    yield from generate_ih_features(settings=settings, features=features)

    # Co-occurrence matrix features.
    yield from generate_cm_features(settings=settings, features=features)

    # Run length matrix features.
    yield from generate_rlm_features(settings=settings, features=features)

    # Size zone matrix features.
    yield from generate_szm_features(settings=settings, features=features)

    # Distance zone matrix features.
    yield from generate_dzm_features(settings=settings, features=features)

    # Neighbourhood grey tone difference matrix features.
    yield from generate_ngtdm_features(settings=settings, features=features)

    # Neighbouring grey level dependence matrix features.
    yield from generate_ngldm_features(settings=settings, features=features)


def feature_to_table(features: list[Feature]) -> pd.DataFrame | None:
    if features is None or len(features) == 0:
        return None

    # Set feature name for feature tables.
    for feature in features:
        feature.create_table_name()

    return pd.DataFrame(dict([(feature.table_name, [feature.value]) for feature in features]))
