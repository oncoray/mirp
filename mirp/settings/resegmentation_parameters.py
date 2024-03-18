from typing import Any
from dataclasses import dataclass
from mirp.settings.utilities import setting_def

import numpy as np


@dataclass
class ResegmentationSettingsClass:
    """
    Parameters related to mask resegmentation. Resegmentation is used to remove parts of the mask that correspond to
    undesired intensities that should be excluded, e.g. those corresponding to air. Resegmentation based on an
    intensity range is also required for using *Fixed Bin Size* discretisation to set the lower bound of the first bin.

    .. note::
        Even though intensity range resegmentation is usually required to perform *Fixed Bin Size* discretisation,
        default values are available for computed tomography (CT) and positron emission tomography (PET) imaging,
        and are set to -1000.0 Hounsfield Units and 0.0 SUV, respectively.

    Parameters
    ----------
    Sets parameters related to resegmentation of the segmentation mask.

    resegmentation_intensity_range: list of float, optional
        Intensity threshold for threshold-based re-segmentation ("threshold" and "range"). If set, requires two
        values for lower and upper range respectively. The upper range value can also be np.nan for half-open ranges.

    resegmentation_sigma: float, optional
        Number of standard deviations for outlier-based intensity re-segmentation ("sigma" and "outlier").

    **kwargs: dict, optional
        Unused keyword arguments.
    """
    def __init__(
            self,
            resegmentation_intensity_range: None | list[float] = None,
            resegmentation_sigma: None | float = None,
            **kwargs
    ):
        resegmentation_method = []
        if resegmentation_sigma is None and resegmentation_intensity_range is None:
            resegmentation_method += ["none"]
        if resegmentation_intensity_range is not None:
            resegmentation_method += ["range"]
        if resegmentation_sigma is not None:
            resegmentation_method += ["sigma"]

        # Set resegmentation method.
        self.resegmentation_method: list[str] = resegmentation_method

        # Set default value.
        if resegmentation_intensity_range is None:
            # Cannot define a proper range.
            resegmentation_intensity_range = [np.nan, np.nan]

        if not isinstance(resegmentation_intensity_range, list):
            raise TypeError(
                f"The resegmentation_intensity_range parameter should be a list with exactly two "
                f"values. Found: an object that is not a list.")

        if len(resegmentation_intensity_range) != 2:
            raise ValueError(
                f"The resegmentation_intensity_range parameter should be a list with exactly two "
                f"values. Found: list with {len(resegmentation_intensity_range)} values.")

        if not all(isinstance(ii, float) for ii in resegmentation_intensity_range):
            raise TypeError(
                f"The resegmentation_intensity_range parameter should be a list with exactly two "
                f"values. Found: one or more values that are not floating point values.")

        self.intensity_range: None | tuple[Any, Any] = tuple(resegmentation_intensity_range) if \
            resegmentation_intensity_range is not None else None

        # Set default value.
        if resegmentation_sigma is None:
            resegmentation_sigma = 3.0

        # Check that sigma is not negative.
        if resegmentation_sigma < 0.0:
            raise ValueError(f"The resegmentation_sigma parameter can not be negative. Found: {resegmentation_sigma}")

        self.sigma: float = resegmentation_sigma


def get_mask_resegmentation_settings() -> list[dict[str, Any]]:
    return [
        setting_def(
            "resegmentation_intensity_range", "float", to_list=True, xml_key=["intensity_range", "g_thresh"],
            class_key="intensity_range", test=[-10.0, 30.0]
        ),
        setting_def("resegmentation_sigma", "float", xml_key="sigma", class_key="sigma", test=1.0)
    ]
