from typing import Any, Generator
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


def get_discretisation_parameters(settings: FeatureExtractionSettingsClass) -> Generator[dict[str, Any], None, None]:
    ...