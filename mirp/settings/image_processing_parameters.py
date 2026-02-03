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
    image_denoise_method: {"none", "median", "gaussian", "susan"}, default: "none"
        Method used for removing noise from the image. The following are possible:

        * "none": images are not denoised.
        * "median": images are denoised using a median filter.
        * "gaussian": images are denoised using a gaussian filter.
        * "susan": images are denoised using the SUSAN noise filtering algorithm. Implemented after eq. 36 of Smith,
          S.M. and Brady, J.M. (1997) ‘SUSAN—A New Approach to Low Level Image Processing’, International Journal of
          Computer Vision, 23(1), pp. 45–78. Available at: https://doi.org/10.1023/a:1007963824710.

        If `by_slice=True`, 2D variants of the denoisers are used, otherwise 3D variants are used instead. Gaussian and
        SUSAN denoisers take voxel spacing into account.

    image_denoiser_median_size: int or list of int, optional, default: 3
        Size of the neighbourhood for the median filter. By default, the voxel neighbourhood is 3 by 3 (by_slice = True)
        or 3 by 3 by 3 (by_slice = False) voxels. 1 or 3 odd, positive integers are expected. If 3 integers are
        provided, these will be used for z, y, and x directions, respectively.

    image_denoiser_gaussian_sigma: float
        Width of the Gaussian filter in physical dimensions (e.g. mm).

    image_denoiser_susan_sigma: float
        Width of the Gaussian filter in SUSAN, in physical dimensions (e.g. mm). The maximum neighbourhood considered
        by the denoiser has a radius of 3 sigma.

    image_denoiser_susan_intensity_threshold: float or None, optional, default: None
        Threshold for the brightness threshold in SUSAN. If None (default), the measure of noise will be determined
        based on the individual image.

    bias_field_correction: bool, optional, default: False
        Determines whether N4 bias field correction should be performed. When a tissue mask is present, bias field
        correction is conducted using the information contained within the mask.

        .. note::
            Bias-field correction can only be applied to MR imaging.

        .. note::
            Bias-field correction using the N4 algorithm is computationally expensive, and it may be preferable to
            perform this correction prior to feeding MR data to MIRP.

    bias_field_correction_n_fitting_levels: int, optional, default: 3
        The number of fitting levels for the N4 bias field correction algorithm. The default in ITK (1) prevents

    bias_field_correction_n_max_iterations: int or list of int, optional, default: 50
        The number of fitting iterations for the N4 bias field algorithm. A single integer, or a list of integers
        with a length equal to the number of fitting levels is expected.

    bias_field_convergence_threshold: float, optional, default: 0.001
        Convergence threshold for N4 bias field correction algorithm.

    pet_suv_conversion: {"body_weight", "body_surface_area", "lean_body_mass", "lean_body_mass_bmi", "ideal_body_weight". "none"}, default: "body_weight"
        Intensities in PET imaging are often stored as detected radiotracer activity. To make detected activity more
        comparable between patients, these are converted to standardised uptake values. The following are possible:

        * "body_weight": activity is normalised by body weight.
        * "body_surface_area": activity is normalised by body surface area according to DuBois (A formula to estimate
          the approximate surface area if height and weight be known. Arch intern med. 1916;17:863-71).
        * "lean_body_mass": activity is normalised by lean body mass according to James et al. (DHSS/MRC Group on
          Obesity Research, James WP, Waterlow JC. Research on Obesity: A Report of the DHSS/MRC Group; Compiled by
          WPT James. HM Stationery Office; 1976).
        * "lean_body_mass_bmi": activity is normalised by lean body mass according to Janmahasatian et al.
          (Quantification of lean bodyweight. Clinical pharmacokinetics. 2005 Oct;44:1051-65).
        * "ideal_body_weight": activity is normalised by ideal body weight according to Zasadny and Wahl (
          Standardized uptake values of normal tissues at PET with 2-[fluorine-18]-fluoro-2-deoxy-D-glucose:
          variations with body weight and a method for correction. Radiology. 1993 Dec;189(3):847-50).
        * "none": activity is not normalised.

        .. note::
            Conversion of activity to standardised uptake values can only be performed for PET data.

        .. warning::
            Conversion of activity to standardised uptake values requires metadata related to image acquisition and to
            the patient. These metadata are only present in DICOM files. MIRP cannot convert activity to standardised
            uptake values for images in different formats, and this parameter will have no effect.

    intensity_normalisation: {"none", "range", "relative_range", "quantile_range", "standardisation", "custom_scale", "histogram_equalisation", "adaptive_equalisation", "match_uniform","match_sigmoid", "match_reference", "match_reference_normalised"}, default: "none"
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
        * "custom_scale": Uses the ``intensity_normalisation_standardisation_shift`` and
        intensity_normalisation_standardisation_scale`` parameters for applying custom shift and scale to the data.
        * "histogram_equalisation": Attempts to generate a flat histogram of intensities. Technically,
          this is a mapping to a uniform distribution. However, the underlying implementation relies on
          ``scikit-image`` and bins intensities prior to mapping. The result is normalised to [0, 1].
        * "adaptive_equalisation": Implementation of Contrast Limited Adaptive Histogram Equalisation (CLAHE) from
          ``scikit-image``. This method cannot use a tissue mask.
        * "match_uniform": Maps intensities to a uniform distribution in [0, 1].
        * "match_sigmoid": Maps intensities to a standard normal distribution.
        * "match_reference": Maps intensities to the distribution of a reference image or vector.
        * "match_reference_normalised": Maps intensities to the distribution of a reference image or vector,
          and normalises the result to [0, 1].

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

    intensity_normalisation_standardisation_shift: float, optional
        Defines the shift parameter for custom scaling. If present, ``intensity_normalisation`` is set to
        ``"custom_scale"``.

    intensity_normalisation_standardisation_scale: float, optional
        Defines the scale parameter for custom scaling. Must be a value > 0. If present, ``intensity_normalisation`` is
        set to ``"custom_scale"``.

    intensity_normalisation_reference: list of float, np.ndarray, optional
        Image, array or list of intensity values that is used to compute the reference intensity distribution for the
        ``"match_reference"`` or ``"match_reference_normalised"`` method.

    intensity_scaling: float, optional
        Defines scaling parameter to linearly scale intensities with. The scaling parameter is applied after
        normalisation (if any). For example, ``intensity_scaling = 1000.0``, combined with ``intensity_normalisation =
        "range"`` results in intensities being mapped to a [0.0, 1000.0] range instead of [0.0, 1.0].

    tissue_mask_type: {"none", "range", "relative_range", "reference"}, optional, default: "relative_range"
        Type of algorithm used to produce an approximate tissue mask of the tissue. Such masks can be used to select
        pixels for bias correction and intensity normalisation by excluding non-tissue voxels. If "reference" an
        external mask is expected.

    tissue_mask_range: list of float, optional
        Range values for creating an approximate mask of the tissue. Required for "range" and "relative_range"
        options. Default: [0.02, 1.00] (``"relative_range"``); [np.nan, np.nan] (``"range"``; effectively all voxels
        are considered to represent tissue).

    tissue_mask_name: str, optional
        Name of the mask to be used as a tissue mask. The mask should be present among the masks provided as regions
        of interest. If provided, the ``tissue_mask_type`` parameter is automatically set to ``"reference"``,
        and other values are ignored.

    **kwargs:
        Unused keyword arguments.
    """

    def __init__(
            self,
            image_denoise_method: str = "none",
            image_denoiser_median_size: int | list[int] = 3,
            image_denoiser_gaussian_sigma: float | None = None,
            image_denoiser_susan_sigma: float | None = None,
            image_denoiser_susan_intensity_threshold: float | None = None,
            bias_field_correction: bool = False,
            bias_field_correction_n_fitting_levels: int = 3,
            bias_field_correction_n_max_iterations: int | list[int] | None = None,
            bias_field_convergence_threshold: float = 0.001,
            pet_suv_conversion: str = "body_weight",
            intensity_normalisation: str = "none",
            intensity_normalisation_range: list[float] | None = None,
            intensity_normalisation_saturation: list[float] | None = None,
            intensity_normalisation_standardisation_shift: float | None = None,
            intensity_normalisation_standardisation_scale: float | None = None,
            intensity_normalisation_reference: list[float] | np.ndarray | None = None,
            intensity_scaling: float | None = None,
            tissue_mask_type: str = "relative_range",
            tissue_mask_range: list[float] | None = None,
            tissue_mask_name: str | None = None,
            **kwargs
    ):
        # Image denoise method parameter
        if not isinstance(image_denoise_method, str):
            raise TypeError("The image_denoise_method parameter is expected to be a string: none, median, gaussian or susan.")
        if not image_denoise_method in ["none", "median", "gaussian", "susan"]:
            raise ValueError(
                f"The image_denoise_method parameter should be one of none, median, gaussian or susan. Found: {image_denoise_method}"
            )
        self.image_denoise_method: str = image_denoise_method

        if self.image_denoise_method == "median":
            # Update the image_denoiser_median_size parameter if a length 1 list is provided.
            if isinstance(image_denoiser_median_size, list) and len(image_denoiser_median_size) == 1:
                image_denoiser_median_size = image_denoiser_median_size[0]

            x = image_denoiser_median_size
            if isinstance(x, int):
                x = [x, x, x]

            if not isinstance(x, list):
                raise TypeError("The image_denoiser_median_size parameter should be an integer or a list of integers.")

            if not len(x) == 3:
                raise ValueError("The image_denoiser_median_size parameter should be an integer or a list of "
                                 f"1 or 3 integers. Found: a list with a length {len(x)}.")

            if not all(isinstance(xi, int) for xi in x):
                raise TypeError("The image_denoiser_median_size parameter should be an integer or a list of integers.")

            if not all(xi % 2 == 1 for xi in x):
                raise ValueError("The image_denoiser_median_size parameter should be an odd integer or a list of "
                                 "odd integers. Found: one or more even integers.")

            if not all(xi > 0 for xi in x):
                raise ValueError("The image_denoiser_median_size parameter should be an odd integer > 0 or a list of "
                                 "odd integers > 0. Found: one or more integers 0 or smaller.")

        self.image_denoiser_median_size: int | list[int] = image_denoiser_median_size

        # Check image_denoiser_gaussian_sigma
        if self.image_denoise_method == "gaussian":

            # Check that the sigma values are floating points.
            if not isinstance(image_denoiser_gaussian_sigma, float):
                raise TypeError(
                    f"The image_denoiser_gaussian_sigma parameter is expected to be a floating point with value "
                    f"greater than 0.0. Found: not a floating point value."
                )

            if not image_denoiser_gaussian_sigma > 0.0:
                raise ValueError(
                    f"The image_denoiser_gaussian_sigma parameter is expected to be a floating point with value "
                    f"greater than 0.0. Found: {image_denoiser_gaussian_sigma}."
                )

        self.image_denoiser_gaussian_sigma = image_denoiser_gaussian_sigma

        # Check image_denoiser_susan_sigma and image_denoiser_susan_intensity_threshold
        if self.image_denoise_method == "susan":

            # Check that the sigma values are floating points.
            if not isinstance(image_denoiser_susan_sigma, float):
                raise TypeError(
                    f"The image_denoiser_susan_sigma parameter is expected to be a floating point with value "
                    f"greater than 0.0. Found: not a floating point value."
                )

            if not image_denoiser_susan_sigma > 0.0:
                raise ValueError(
                    f"The image_denoiser_susan_sigma parameter is expected to be a floating point with value "
                    f"greater than 0.0. Found: {image_denoiser_susan_sigma}."
                )

            # Check that SUSAN intensity threshold is None or correct floating points.
            if not isinstance(image_denoiser_susan_intensity_threshold, float) and \
                    image_denoiser_susan_intensity_threshold is not None:
                raise TypeError(
                    f"The image_denoiser_susan_intensity_threshold parameter is expected to be a floating point with "
                    f"value greater than 0.0, or None (default). Found: not a floating point value or None."
                )

            if isinstance(image_denoiser_susan_intensity_threshold, float) and not image_denoiser_susan_sigma > 0.0:
                raise ValueError(
                    f"The image_denoiser_susan_intensity_threshold parameter is expected to be a floating point with "
                    f"value greater than 0.0. Found: {image_denoiser_susan_intensity_threshold}."
                )

        self.image_denoiser_susan_sigma = image_denoiser_susan_sigma
        self.image_denoiser_susan_intensity_threshold = image_denoiser_susan_intensity_threshold

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

        # Check that pet_suv_conversion has the correct values.
        if pet_suv_conversion not in [
            "body_weight", "body_surface_area", "lean_body_mass", "lean_body_mass_bmi", "ideal_body_weight", "none"
        ]:
            raise ValueError(
                f"The pet_suv_conversion parameter is expected to have one of the following values: ",
                f"'body_weight', `body_surface_area`, `lean_body_mass`, `lean_body_mass_bmi`, `ideal_body_weight` or "
                f"'none'. Found: {pet_suv_conversion}"
            )

        # Set suv_conversion_type parameter.
        self.suv_conversion_type = pet_suv_conversion

        # Check that intensity_normalisation has the correct values.
        if intensity_normalisation not in self._get_available_intensity_normalisation_methods():
            raise ValueError(
                f"The intensity_normalisation parameter is expected to have one of the following values: "
                f"{', '.join(self._get_available_intensity_normalisation_methods())}. Found: {intensity_normalisation}"
            )

        if intensity_normalisation_standardisation_scale is not None or \
                intensity_normalisation_standardisation_scale is not None:
            intensity_normalisation = "custom_scale"

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

        if self.intensity_normalisation == "custom_scale":
            if not isinstance(intensity_normalisation_standardisation_shift, float):
                raise TypeError(
                    "The intensity_normalisation_standardisation_shift parameter should be a floating point value."
                )
            if not isinstance(intensity_normalisation_standardisation_scale, float):
                raise TypeError(
                    "The intensity_normalisation_standardisation_scale parameter should be a floating point value > 0."
                )

            if intensity_normalisation_standardisation_scale <= 0.0:
                raise ValueError(
                    f"The intensity_normalisation_standardisation_scale parameter should be a floating point value > "
                    f"0. Found: {intensity_normalisation_standardisation_scale}"
                )

        self.intensity_normalisation_standardisation_shift = intensity_normalisation_standardisation_shift
        self.intensity_normalisation_standardisation_scale = intensity_normalisation_standardisation_scale

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

        # Check intensity_normalisation_reference.
        if self.intensity_normalisation not in ["match_reference", "match_reference_normalised"]:
            # Avoid passing intensity_normalisation_reference if it is not used.
            intensity_normalisation_reference = None
        elif self.intensity_normalisation in ["match_reference", "match_reference_normalised"] and \
                intensity_normalisation_reference is None:
            # This parameter must be provided for the above methods.
            raise ValueError("The intensity_normalisation_reference parameter must be provided for the "
                             "`match_reference` and `match_reference_normalised` methods.")

        if intensity_normalisation_reference is not None:
            if isinstance(intensity_normalisation_reference, list):
                intensity_normalisation_reference = np.ndarray(intensity_normalisation_reference)
            elif isinstance(intensity_normalisation_reference, np.ndarray):
                pass
            else:
                raise TypeError("intensity_normalisation_reference should be a list or np.ndarray")

            # Check if the provided array can be cast to floating points.
            if not np.can_cast(intensity_normalisation_reference, float):
                raise TypeError("intensity_normalisation_reference should be a list or np.ndarray of floats, "
                                "or castable to float.")

        self.intensity_normalisation_reference: None | np.ndarray = intensity_normalisation_reference

        # Check intensity_scaling
        if intensity_scaling is not None:
            if not isinstance(intensity_scaling, float):
                raise TypeError("The intensity_scaling parameter is expected to be a single floating point.")
            if intensity_scaling == 0.0:
                raise ValueError("The intensity_scaling parameter cannot have a value of 0.0.")
        else:
            intensity_scaling = 1.0

        self.intensity_scaling: float = intensity_scaling

        # Override tissue_mask_type in case a mask is explicitly provided.
        if isinstance(tissue_mask_name, str):
            tissue_mask_type = "reference"

        # Check tissue_mask_type
        if tissue_mask_type not in ["none", "range", "relative_range", "reference"]:
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

        # Check tissue_mask_name. Note that the actual content needs to be checked at run-time, because at this point
        # we don't know if the mask corresponding to the name is actually present.
        if self.tissue_mask_type == "reference":
            if not isinstance(tissue_mask_name, str):
                raise TypeError(
                    "The tissue_mask_name parameter is expected to be a string indicated the mask to be "
                    "used as the tissue mask."
                )

        self.tissue_mask_name: str = tissue_mask_name

    @staticmethod
    def _get_available_intensity_normalisation_methods():
        return [
            "none", "range", "relative_range", "quantile_range", "standardisation", "custom_scale",
            "histogram_equalisation", "adaptive_equalisation", "match_uniform","match_sigmoid", "match_reference",
            "match_reference_normalised"
        ]

def get_post_processing_settings() -> list[dict[str, Any]]:

    return [
        setting_def("image_denoise_method", typing="str", test="median"),
        setting_def("image_denoiser_median_size", typing="int", to_list=True, test=5),
        setting_def("image_denoiser_gaussian_sigma", typing="float", test=3.0),
        setting_def("image_denoiser_susan_sigma", typing="float", test=4.0),
        setting_def("image_denoiser_susan_intensity_threshold", typing="float", test=6.0),
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
        setting_def("pet_suv_conversion", "str", class_key="suv_conversion_type", test="none"),
        setting_def("intensity_normalisation", "str", test="relative_range"),
        setting_def("intensity_normalisation_range", "float", to_list=True, test=[0.10, 0.90]),
        setting_def("intensity_normalisation_saturation", "float", to_list=True, test=[0.00, 10.00]),
        setting_def("intensity_normalisation_reference", "float", to_list=True),
        setting_def("intensity_normalisation_standardisation_shift", "float"),
        setting_def("intensity_normalisation_standardisation_scale", "float"),
        setting_def("intensity_scaling", "float", test=3.0),
        setting_def("tissue_mask_type", "str", test="range"),
        setting_def("tissue_mask_range", "float", to_list=True, test=[0.00, 10.00]),
        setting_def("tissue_mask_name", "str")
    ]
