from typing import Union, List, Optional, Tuple, Any

import numpy as np


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

    resegmentation_method: {"none", "threshold", "range", "sigma", "outlier"}, optional, default: "none"
        Re-segmentation method. "threshold" and "range" are synonymous and will perform resegmentation based on the
        intensity range provided as the `resegmentation_intensity_range` parameter below. "sigma" and "outlier" are
        synonymous for re-segmentation based on the distribution of intensities of voxels in the mask.

    resegmentation_intensity_range: list of float, optional, default: None
        Intensity threshold for threshold-based re-segmentation ("threshold" and "range"). If set, requires two
        values for lower and upper range respectively. The upper range value can also be np.nan for half-open ranges.

    resegmentation_sigma: float, optional, default: 3.0
        Number of standard deviations for outlier-based intensity re-segmentation ("sigma" and "outlier").

    **kwargs: dict, optional
        Unused keyword arguments.
    """
    def __init__(
            self,
            resegmentation_method: Union[str, List[str]] = "none",
            resegmentation_intensity_range: Union[None, List[float]] = None,
            resegmentation_sigma: float = 3.0,
            **kwargs):

        # Check values for resegmentation_method.
        if not isinstance(resegmentation_method, list):
            resegmentation_method = [resegmentation_method]

        # If "none" is present, remove all other methods.
        if "none" in resegmentation_method:
            resegmentation_method = ["none"]

        # Check that methods are valid.
        if not all([ii in ["none", "threshold", "range", "sigma", "outlier"] for ii in resegmentation_method]):
            raise ValueError(
                "The resegmentation_method parameter can only have the following values: 'none', "
                "'threshold', 'range', 'sigma' and 'outlier'.")

        # Remove redundant values.
        if "threshold" in resegmentation_method and "range" in resegmentation_method:
            resegmentation_method.remove("threshold")

        if "sigma" in resegmentation_method and "outlier" in resegmentation_method:
            resegmentation_method.remove("sigma")

        # Set resegmentation method.
        self.resegmentation_method: List[str] = resegmentation_method

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

        self.intensity_range: Optional[Tuple[Any, Any]] = tuple(resegmentation_intensity_range) if \
            resegmentation_intensity_range is not None else None

        # Check that sigma is not negative.
        if resegmentation_sigma < 0.0:
            raise ValueError(f"The resegmentation_sigma parameter can not be negative. Found: {resegmentation_sigma}")

        self.sigma: float = resegmentation_sigma
