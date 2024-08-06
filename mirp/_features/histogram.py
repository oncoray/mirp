from typing import Any, Generator
from functools import lru_cache

from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
from mirp._features.base_feature import Feature
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask


def get_discretisation_parameters(settings: FeatureExtractionSettingsClass) -> Generator[dict[str, Any], None, None]:
    for discretisation_method in settings.discretisation_method:
        if discretisation_method in ["fixed_bin_size", "fixed_bin_size_pyradiomics"]:
            bin_width = settings.discretisation_bin_width
            for current_bin_width in bin_width:
                yield {
                    "discretisation_method": discretisation_method,
                    "bin_width": current_bin_width
                }

        elif discretisation_method in ["fixed_bin_number"]:
            bin_number = settings.discretisation_n_bins
            for current_bin_number in bin_number:
                yield {
                    "discretisation_method": discretisation_method,
                    "bin_number": current_bin_number
                }
        else:
            yield {"discretisation_method": discretisation_method}


class HistogramDerivedFeature(Feature):

    def __init__(
            self,
            discretisation_method: str,
            bin_width: None | float = None,
            bin_number: None | int = None,
            cropping_distance: None | float = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.discretisation_method = discretisation_method
        self.bin_width = bin_width
        self.bin_number = bin_number
        self.cropping_distance = cropping_distance

    @staticmethod
    @lru_cache(maxsize=1)
    def discretise_image(
            image: GenericImage,
            mask: BaseMask,
            discretisation_method: str,
            bin_width: float | None,
            bin_number: int | None,
            cropping_distance: float | None
    ) -> tuple[GenericImage, BaseMask]:
        from mirp._image_processing.cropping import crop
        from mirp._image_processing.discretise_image import discretise_image

        if cropping_distance is not None:
            image, mask = crop(
                image=image,
                masks=mask,
                boundary=cropping_distance,
                in_place=False
            )

        image, mask = discretise_image(
            image=image,
            mask=mask,
            discretisation_method=discretisation_method,
            bin_number=bin_number,
            bin_width=bin_width,
            in_place=False
        )

        return image, mask

    def clear_cache(self):
        super().clear_cache()
        self.discretise_image.cache_clear()

    def update_ibsi_compliance(self):
        if self.discretisation_method == "fixed_bin_size_pyradiomics":
            # pyradiomics with fixed bin size is explicitly not IBSI-compliant.
            self.ibsi_compliant = False

    def _get_discretisation_table_name_element(self) -> list[str]:
        table_elements = []
        if self.discretisation_method is not None:
            if self.discretisation_method == "none":
                pass
            elif self.discretisation_method == "fixed_bin_number":
                table_elements += ["fbn"]
                table_elements += ["n" + str(self.bin_number)]
            elif self.discretisation_method == "fixed_bin_size":
                table_elements += ["fbs"]
                table_elements += ["w" + str(self.bin_width)]
            elif self.discretisation_method == "fixed_bin_size_pyradiomics":
                table_elements += ["fbsp"]
                table_elements += ["w" + str(self.bin_width)]
            else:
                raise ValueError(f"Unknown discretisation method: {self.discretisation_method}")

        return table_elements
