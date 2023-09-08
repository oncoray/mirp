from typing import Union, List, Tuple

import numpy as np


class ImagePostProcessingClass:

    def __init__(
            self,
            bias_field_correction: bool = False,
            bias_field_correction_n_fitting_levels: int = 3,
            bias_field_correction_n_max_iterations: Union[int, List[int], None] = None,
            bias_field_convergence_threshold: float = 0.001,
            intensity_normalisation: str = "none",
            intensity_normalisation_range: Union[List[float], None] = None,
            intensity_normalisation_saturation: Union[List[float], None] = None,
            tissue_mask_type: str = "relative_range",
            tissue_mask_range: Union[List[float], None] = None,
            **kwargs):
        """
        Sets parameters related to image post-processing. The current parameters can be used to post-process MR
        imaging.

        :param bias_field_correction: Determines whether N4 bias field correction should be performed. Only
            applicable to MR imaging. When a tissue mask is present, bias field correction is conducted using the
            information contained within the mask. Default: False
        :param bias_field_correction_n_fitting_levels: The number of fitting levels for the N4 bias field correction
            algorithm. Default: 3.
        :param bias_field_correction_n_max_iterations: The number of fitting iterations for the N4 bias field
            algorithm. A single integer, or a list of integers with a length equal to the number of fitting levels is
            expected. Default: 100.
        :param bias_field_convergence_threshold: Convergence threshold for N4 bias field correction algorithm.
            Default: 0.001.
        :param intensity_normalisation: Specifies the algorithm used to normalise intensities in the image. Will use
            only intensities in voxels masked by the tissue mask. Can be "none" (no normalisation), "range" (maps to an
            intensity range), "relative_range" (maps to a range of relative intensities), "quantile_range" (maps to a
            range of relative intensities based on intensity percentiles), and "standardisation" (performs
             z-normalisation). Default: "none".
        :param intensity_normalisation_range: Defines the start and endpoint of the range that are mapped to the [
            0.0, 1.0] interval for "range", "relative_range", and "quantile_range" intensity normalisation
            algorithms. Default: [np.nan, np.nan] ("range"; this uses the intensity range in the image (mask) for
            normalisation); [0.0, 1.0] ("relative_range"; this also uses the full intensity range); or [0.025,
            0.975] ("quantile_range": the range is defined by the 2.5th and 97.5th percentiles of the intensities in
            the image).
        :param intensity_normalisation_saturation: Defines the start and endpoint for the saturation range. Values
            after normalisation that lie outside this range are mapped to the limits of the range, e.g. with a range
            of [0.0, 0.8] all values greater than 0.8 are assigned a value of 0.8. np.nan can be used to define
            limits where the intensity values should not be saturated. Default: [np.nan, np.nan].
        :param tissue_mask_type: Type of algorithm used to produce an approximate mask of the tissue. Such masks
            can be used to select pixels for bias correction and intensity normalisation by excluding air. The mask
            type can be "none", "range" (requires intensity values as `tissue_mask_range`), or "relative_range" (
            requires fractions as `tissue_mask_range`). Default: "relative_range".
        :param tissue_mask_range:  Range values for creating an approximate mask of the tissue. Required for range
            and relative range options. Default: [0.02, 1.00] (relative_range); [0.0, np.nan] (range; effectively none).
        :param kwargs: unused keyword arguments.

        :returns: A :class:`mirp.importSettings.ImagePostProcessingClass` object with configured parameters.
        """

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
        self.n_fitting_levels: Union[None, int] = bias_field_correction_n_fitting_levels

        # Set default value for bias_field_correction_n_max_iterations. This is the number of iterations per fitting
        # level.
        if bias_field_correction_n_max_iterations is None and bias_field_correction:
            bias_field_correction_n_max_iterations = [100 for ii in range(bias_field_correction_n_fitting_levels)]

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
        self.n_max_iterations: Union[List[int], None] = bias_field_correction_n_max_iterations

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
        self.convergence_threshold: Union[None, float] = bias_field_convergence_threshold

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
        self.intensity_normalisation_range: Union[None, List[float]] = intensity_normalisation_range

        # Check intensity normalisation saturation range.
        if intensity_normalisation_saturation is None:
            intensity_normalisation_saturation = [np.nan, np.nan]

        if not isinstance(intensity_normalisation_saturation, list):
            raise TypeError("The tissue_mask_range parameter is expected to be a list of two floating point values.")

        if not len(intensity_normalisation_saturation) == 2:
            raise ValueError(
                f"The tissue_mask_range parameter should consist of two values. Found: "
                f"{len(intensity_normalisation_saturation)} values.")

        if not all(isinstance(ii, float) for ii in intensity_normalisation_saturation):
            raise TypeError("The tissue_mask_range parameter can only contain floating point or np.nan values.")

        # intensity_normalisation_saturation parameter
        self.intensity_normalisation_saturation: Union[None, List[float]] = intensity_normalisation_saturation

        # Check tissue_mask_type
        if tissue_mask_type not in ["none", "range", "relative_range"]:
            raise ValueError(
                f"The tissue_mask_type parameter is expected to have one of the following values: "
                f"'none', 'range', or 'relative_range'. Found: {tissue_mask_type}.")

        # Set tissue_mask_type
        self.tissue_mask_type: str = tissue_mask_type

        # Set the default value for tissue_mask_range.
        if tissue_mask_range is None:
            if tissue_mask_type == "relative_range":
                tissue_mask_range = [0.02, 1.00]
            elif tissue_mask_type == "range":
                tissue_mask_range = [0.0, np.nan]
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
        self.tissue_mask_range: Tuple[float, ...] = tuple(tissue_mask_range)
