import numpy as np

from typing import Any
from dataclasses import dataclass
from mirp.settings.utilities import setting_def


@dataclass
class ImagePostProcessingClass:
    """
    Parameters related to image processing. Note that parameters concerning image perturbation / augmentation and
    resampling are set separately, see :class:`~mirp.settings.perturbation_parameters.ImagePerturbationSettingsClass` and
    :class:`~mirp.settings.interpolation_parameters.ImageInterpolationSettingsClass`.

    Parameters
    ----------
    bias_field_correction: bool, optional, default: False
        Determines whether N4 bias field correction should be performed. When a tissue mask is present, bias field
        correction is conducted using the information contained within the mask. Bias-field correction can only be
        applied to MR imaging.

    bias_field_correction_n_fitting_levels: int, optional, default: 1
        The number of fitting levels for the N4 bias field correction algorithm.

    bias_field_correction_n_max_iterations: int or list of int, optional, default: 50
        The number of fitting iterations for the N4 bias field algorithm. A single integer, or a list of integers
        with a length equal to the number of fitting levels is expected.

    bias_field_convergence_threshold: float, optional, default: 0.001
        Convergence threshold for N4 bias field correction algorithm.

    intensity_normalisation: {"none", "range", "relative_range", "quantile_range", "standardisation"}, default: "none"
        Specifies the algorithm used to normalise intensities in the image. Will use only intensities in voxels
        masked by the tissue mask (of present). The following are possible:

        * "none": no normalisation
        * "range": normalises intensities based on a fixed mapping against the ``intensity_normalisation_range``
          parameter, which is interpreted to represent an intensity range.
        * "relative_range": normalises intensities based on a fixed mapping against the ``intensity_normalisation_range``
          parameter, which is interpreted to represent a relative intensity range.
        * "quantile_range": normalises intensities based on a fixed mapping against the
          ``intensity_normalisation_range`` parameter, which is interpreted to represent a quantile range.
        * "standardisation": normalises intensities by subtraction of the mean intensity and division by the standard
          deviation of intensities.

        .. note::
            Intensity normalisation may remove any physical meaning of intensity units. For example, intensity
            normalisation of CT images yield intensities that no longer represent Hounsfield Units.

    intensity_normalisation_range: list of float, optional
        Required for "range", "relative_range", and "quantile_range" intensity normalisation methods, and defines the
        intensities that are mapped to the [0.0, 1.0] range during normalisation. The default range depends on the
        type of normalisation method:

        * "range": [np.nan, np.nan]: the minimum and maximum intensity value present in the image are used to set the
          mapping range.
        * "relative_range": [0.0. 1.0]: the minimum (0.0) and maximum (1.0) intensity value present in the image are
          used to set the mapping range.
        * "quantile_range": [0.025, 0.975] the 2.5th and 97.5th percentiles of the intensities in the image are used
          to set the mapping range.

        The lower end of the range is mapped to 0.0 and the upper end to 1.0. However, if intensities below the lower
        end or above the upper end are present in the image, values below 0.0 or above 1.0 may be encountered after
        normalisation. Use ``intensity_normalisation_saturation`` to cap intensities after normalisation to a
        specific range.

    intensity_normalisation_saturation: list of float, optional, default: [np.nan, np.nan]
        Defines the start and endpoint for the saturation range. Normalised intensities that lie outside this
        range are mapped to the limits of the saturation range, e.g. with a range of [0.0, 0.8] all values greater
        than 0.8 are assigned a value of 0.8. np.nan can be used to define limits where the intensity values should
        not be saturated.

    intensity_scaling: float, optional
        Defines scaling parameter to linearly scale intensities with. The scaling parameter is applied after
        normalisation (if any). For example, `intensity_scaling = 1000.0`, combined with `intensity_normalisation =
        "range"` results in intensities being mapped to a [0.0, 1000.0] range instead of [0.0, 1.0].

    tissue_mask_type: {"none", "range", "relative_range"}, optional, default: "relative_range"
        Type of algorithm used to produce an approximate tissue mask of the tissue. Such masks can be used to select
        pixels for bias correction and intensity normalisation by excluding non-tissue voxels.

    tissue_mask_range: list of float, optional
        Range values for creating an approximate mask of the tissue. Required for "range" and "relative_range"
        options. Default: [0.02, 1.00] (``"relative_range"``); [np.nan, np.nan] (``"range"``; effectively all voxels
        are considered to represent tissue).

    **kwargs:
        Unused keyword arguments.
    """

    def __init__(
            self,
            bias_field_correction: bool = False,
            bias_field_correction_n_fitting_levels: int = 1,
            bias_field_correction_n_max_iterations: int | list[int] | None = None,
            bias_field_convergence_threshold: float = 0.001,
            intensity_normalisation: str = "none",
            intensity_normalisation_range: list[float] | None = None,
            intensity_normalisation_saturation: list[float] | None = None,
            intensity_scaling: float | None = None,
            tissue_mask_type: str = "relative_range",
            tissue_mask_range: list[float] | None = None,
            **kwargs
    ):

        # Set bias_field_correction parameter
        self.bias_field_correction = bias_field_correction

        # Check n_fitting_levels.
        if bias_field_correction:
            if not isinstance(bias_field_correction_n_fitting_levels, int):
                raise TypeError("The bias_field_correction_n_fitting_levels should be integer with value 1 or larger.")

            if bias_field_correction_n_fitting_levels < 1:
                raise ValueError(
                    f"The bias_field_correction_n_fitting_levels should be integer with value 1 or larger. "
                    f"Found: {bias_field_correction_n_fitting_levels}")

        else:
            bias_field_correction_n_fitting_levels = None

        # Set n_fitting_levels.
        self.n_fitting_levels: None | int = bias_field_correction_n_fitting_levels

        # Set default value for bias_field_correction_n_max_iterations. This is the number of iterations per fitting
        # level.
        if bias_field_correction_n_max_iterations is None and bias_field_correction:
            bias_field_correction_n_max_iterations = [50 for ii in range(bias_field_correction_n_fitting_levels)]

        if bias_field_correction:
            # Parse to list, if a single value is provided.
            if not isinstance(bias_field_correction_n_max_iterations, list):
                bias_field_correction_n_max_iterations = [bias_field_correction_n_max_iterations]

            # Ensure that the list of maximum iteration values equals the number of fitting levels.
            if bias_field_correction_n_fitting_levels > 1 and len(bias_field_correction_n_max_iterations) == 1:
                bias_field_correction_n_max_iterations = [
                    bias_field_correction_n_max_iterations[0]
                    for ii in range(bias_field_correction_n_fitting_levels)
                ]

            # Check that the list of maximum iteration values is equal to the number of fitting levels.
            if len(bias_field_correction_n_max_iterations) != bias_field_correction_n_fitting_levels:
                raise ValueError(
                    f"The bias_field_correction_n_max_iterations parameter should be a list with a length equal to the"
                    f" number of fitting levels ({bias_field_correction_n_fitting_levels}). Found list with "
                    f"{len(bias_field_correction_n_max_iterations)} values.")

            # Check that all values are integers.
            if not all(isinstance(ii, int) for ii in bias_field_correction_n_max_iterations):
                raise TypeError(
                    f"The bias_field_correction_n_max_iterations parameter should be a list of positive "
                    f"integer values. At least one value was not an integer.")

            # Check that all values are positive.
            if not all([ii > 0 for ii in bias_field_correction_n_max_iterations]):
                raise ValueError(
                    f"The bias_field_correction_n_max_iterations parameter should be a list of positive "
                    f"integer values. At least one value was zero or negative.")

        else:
            bias_field_correction_n_max_iterations = None

        # Set n_max_iterations attribute.
        self.n_max_iterations: list[int] | None = bias_field_correction_n_max_iterations

        # Check that the convergence threshold is a non-negative number.
        if bias_field_correction:

            # Check that the value is a float.
            if not isinstance(bias_field_convergence_threshold, float):
                raise TypeError(
                    f"The bias_field_convergence_threshold parameter is expected to be a non-negative "
                    f"floating point value. Found: a value that was not a floating point value.")

            if bias_field_convergence_threshold <= 0.0:
                raise TypeError(
                    f"The bias_field_convergence_threshold parameter is expected to be a non-positive floating point "
                    f"value. Found: a value that was 0.0 or negative ({bias_field_convergence_threshold}).")

        else:
            bias_field_convergence_threshold = None

        # Set convergence_threshold attribute.
        self.convergence_threshold: None | float = bias_field_convergence_threshold

        # Check that intensity_normalisation has the correct values.
        if intensity_normalisation not in ["none", "range", "relative_range", "quantile_range", "standardisation"]:
            raise ValueError(
                f"The intensity_normalisation parameter is expected to have one of the following values: "
                f"'none', 'range', 'relative_range', 'quantile_range', 'standardisation'. Found: "
                f"{intensity_normalisation}.")

        # Set intensity_normalisation parameter.
        self.intensity_normalisation = intensity_normalisation

        # Set default value.
        if intensity_normalisation_range is None:
            if intensity_normalisation == "range":
                # Cannot define a proper range.
                intensity_normalisation_range = [np.nan, np.nan]

            elif intensity_normalisation == "relative_range":
                intensity_normalisation_range = [0.0, 1.0]

            elif intensity_normalisation == "quantile_range":
                intensity_normalisation_range = [0.025, 0.975]

        if intensity_normalisation == "range":
            # Check that the range has length 2 and contains floating point values.
            if not isinstance(intensity_normalisation_range, list):
                raise TypeError(
                    f"The intensity_normalisation_range parameter for range-based normalisation should "
                    f"be a list with exactly two values, which are mapped to 0.0 and 1.0 respectively. "
                    f"Found: an object that is not a list.")

            if len(intensity_normalisation_range) != 2:
                raise ValueError(
                    f"The intensity_normalisation_range parameter for range-based normalisation should "
                    f"be a list with exactly two values, which are mapped to 0.0 and 1.0 respectively. "
                    f"Found: list with {len(intensity_normalisation_range)} values.")

            if not all(isinstance(ii, float) for ii in intensity_normalisation_range):
                raise TypeError(
                    f"The intensity_normalisation_range parameter for range-based normalisation should "
                    f"be a list with exactly two floating point values, which are mapped to 0.0 and 1.0 "
                    f"respectively. Found: one or more values that are not floating point values.")

        elif intensity_normalisation in ["relative_range", "quantile_range"]:
            # Check that the range has length 2 and contains floating point values between 0.0 and 1.0.
            if intensity_normalisation == "relative_range":
                intensity_normalisation_specifier = "relative range-based normalisation"
            else:
                intensity_normalisation_specifier = "quantile range-based normalisation"

            if not isinstance(intensity_normalisation_range, list):
                raise TypeError(
                    f"The intensity_normalisation_range parameter for {intensity_normalisation_specifier} "
                    f"should be a list with exactly two values, which are mapped to 0.0 and 1.0 "
                    f"respectively. Found: an object that is not a list.")

            if len(intensity_normalisation_range) != 2:
                raise ValueError(
                    f"The intensity_normalisation_range parameter for {intensity_normalisation_specifier} "
                    f"should be a list with exactly two values, which are mapped to 0.0 and 1.0 "
                    f"respectively. Found: list with {len(intensity_normalisation_range)} values.")

            if not all(isinstance(ii, float) for ii in intensity_normalisation_range):
                raise TypeError(
                    f"The intensity_normalisation_range parameter for {intensity_normalisation_specifier} "
                    f"should be a list with exactly two values, which are mapped to 0.0 and 1.0 "
                    f"respectively. Found: one or more values that are not floating point values.")

            if not all([0.0 <= ii <= 1.0 for ii in intensity_normalisation_range]):
                raise TypeError(
                    f"The intensity_normalisation_range parameter for {intensity_normalisation_specifier} "
                    f"should be a list with exactly two values, which are mapped to 0.0 and 1.0 "
                    f"respectively. Found: one or more values that are outside the [0.0, 1.0] range.")

        else:
            # None and standardisation do not use this range.
            intensity_normalisation_range = None

        # Set normalisation range.
        self.intensity_normalisation_range: None | list[float] = intensity_normalisation_range

        # Check intensity normalisation saturation range.
        if intensity_normalisation_saturation is None:
            intensity_normalisation_saturation = [np.nan, np.nan]

        if not isinstance(intensity_normalisation_saturation, list):
            raise TypeError(
                "The intensity_normalisation_saturation parameter is expected to be a "
                "list of two floating point values."
            )

        if not len(intensity_normalisation_saturation) == 2:
            raise ValueError(
                f"The intensity_normalisation_saturation parameter should consist of two values. Found: "
                f"{len(intensity_normalisation_saturation)} values.")

        if not all(isinstance(ii, float) for ii in intensity_normalisation_saturation):
            raise TypeError(
                "The intensity_normalisation_saturation parameter can only contain floating point or np.nan values."
            )

        # intensity_normalisation_saturation parameter
        self.intensity_normalisation_saturation: None | list[float] = intensity_normalisation_saturation

        # Check intensity_scaling
        if intensity_scaling is not None:
            if not isinstance(intensity_scaling, float):
                raise TypeError("The intensity_scaling parameter is expected to be a single floating point.")
            if intensity_scaling == 0.0:
                raise ValueError("The intensity_scaling parameter cannot have a value of 0.0.")
        else:
            intensity_scaling = 1.0

        self.intensity_scaling: float = intensity_scaling

        # Check tissue_mask_type
        if tissue_mask_type not in ["none", "range", "relative_range"]:
            raise ValueError(
                f"The tissue_mask_type parameter is expected to have one of the following values: "
                f"'none', 'range', or 'relative_range'. Found: {tissue_mask_type}."
            )

        # Set tissue_mask_type
        self.tissue_mask_type: str = tissue_mask_type

        # Set the default value for tissue_mask_range.
        if tissue_mask_range is None:
            if tissue_mask_type == "relative_range":
                tissue_mask_range = [0.02, 1.00]
            elif tissue_mask_type == "range":
                tissue_mask_range = [np.nan, np.nan]
            else:
                tissue_mask_range = [np.nan, np.nan]

        # Perform checks on tissue_mask_range.
        if tissue_mask_type != "none":
            if not isinstance(tissue_mask_range, list):
                raise TypeError(
                    "The tissue_mask_range parameter is expected to be a list of two floating point values.")

            if not len(tissue_mask_range) == 2:
                raise ValueError(
                    f"The tissue_mask_range parameter should consist of two values. Found: "
                    f"{len(tissue_mask_range)} values.")

            if not all(isinstance(ii, float) for ii in tissue_mask_range):
                raise TypeError("The tissue_mask_range parameter can only contain floating point or np.nan values.")

            if tissue_mask_type == "relative_range":
                if not all([(0.0 <= ii <= 1.0) or np.isnan(ii) for ii in tissue_mask_range]):
                    raise ValueError(
                        "The tissue_mask_range parameter should consist of two values between 0.0 and 1.0.")

        # Set tissue_mask_range.
        self.tissue_mask_range: tuple[float, ...] = tuple(tissue_mask_range)


def get_post_processing_settings() -> list[dict[str, Any]]:

    return [
        setting_def("bias_field_correction", "bool", test=True),
        setting_def(
            "bias_field_correction_n_fitting_levels", "int", xml_key="n_fitting_levels",
            class_key="n_fitting_levels", test=2
        ),
        setting_def(
            "bias_field_correction_n_max_iterations", "int", xml_key="n_max_iterations",
            class_key="n_max_iterations", to_list=True, test=[1000, 1000]
        ),
        setting_def(
            "bias_field_convergence_threshold", "float", xml_key="convergence_threshold",
            class_key="convergence_threshold", test=0.1
        ),
        setting_def("intensity_normalisation", "str", test="relative_range"),
        setting_def("intensity_normalisation_range", "float", to_list=True, test=[0.10, 0.90]),
        setting_def("intensity_normalisation_saturation", "float", to_list=True, test=[0.00, 10.00]),
        setting_def("intensity_scaling", "float", test=3.0),
        setting_def("tissue_mask_type", "str", test="range"),
        setting_def("tissue_mask_range", "float", to_list=True, test=[0.00, 10.00])
    ]
