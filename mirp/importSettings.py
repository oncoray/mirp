import copy
import xml.etree.ElementTree
import xml.etree.ElementTree as ElemTree

import numpy as np
import pandas as pd
import sys

from typing import Union, List


class GeneralSettingsClass:

    def __init__(self,
                 by_slice: Union[str, bool] = False,
                 config_str: str = "",
                 divide_disconnected_roi: str = "keep_as_is",
                 no_approximation: bool = False,
                 **kwargs):
        """
        Sets general experiment parameters.

        :param by_slice: Defines whether calculations should be performed in 2D (True) or 3D (False),
            or alternatively only in the largest slice (Largest). Default: False
        :param config_str: Sets a configuration string, which can be used to differentiate files or analyses created
            from the same data but a different configuration.
        :param divide_disconnected_roi: Defines how ROI segmentations are treated after being loaded. Can be
            "keep_as_is" (default), "keep_largest", "combine".
        :param no_approximation: Disables approximation of features, such as Geary's c-measure. Can be True or False
            (default).
        :param kwargs: Unused keyword arguments.

        :returns: A :class:`mirp.importSettings.GeneralSettingsClass` object with configured parameters.
        """

        # Parse and check slice information.
        if isinstance(by_slice, str):
            if by_slice.lower() in ["true", "t", "1"]:
                by_slice = True
                select_slice = "all"
            elif by_slice.lower() in ["false", "f", "0"]:
                by_slice = False
                select_slice = "all"
            elif by_slice.lower() in ["largest"]:
                by_slice = True
                select_slice = "largest"
            else:
                raise ValueError(
                    f"The by_slice parameter should be true, false, t, f, 1, 0 or largest. Found: {by_slice}")

        elif isinstance(by_slice, bool):
            select_slice = "all"

        else:
            raise ValueError("The by_slice parameter should be a string or boolean.")

        # Set by_slice and select_slice parameters.
        self.by_slice: bool = by_slice
        self.select_slice: str = select_slice

        # Set configuration string.
        self.config_str: str = config_str

        # Check divide_disconnected_roi
        if divide_disconnected_roi not in ["keep_as_is", "keep_largest", "combine"]:
            raise ValueError(f"The divide_disconnected_roi parameter should be 'keep_as_is', 'keep_largest', "
                             f"'combine'. Found: {divide_disconnected_roi}")

        # Set divide_disconnected_roi
        self.divide_disconnected_roi: str = divide_disconnected_roi

        # Set approximation of features.
        self.no_approximation: bool = no_approximation


class ImageInterpolationSettingsClass:

    def __init__(self,
                 by_slice: bool,
                 interpolate: bool = False,
                 spline_order: int = 3,
                 new_spacing: Union[float, int, List[int], List[float], None] = None,
                 anti_aliasing: bool = True,
                 smoothing_beta: float = 0.98,
                 **kwargs):
        """
        Sets parameters related to image interpolation.

        :param by_slice: Defines whether the experiment is by slice (True) or volumetric (False).
            See :class:`mirp.importSettings.GeneralSettingsClass`.
        :param interpolate: Controls whether interpolation of images to a common grid is performed at all.
        :param spline_order: Sets the spline order used for spline interpolation. Default: 3 (tricubic spline).
        :param new_spacing: Sets voxel spacing after interpolation. A single value represents the spacing that
            will be applied in all directions. Non-uniform voxel spacing may also be provided, but requires 3 values
            for z, y, and x directions (if `by_slice = False`) or 2 values for y and x directions (otherwise).
            Multiple spacings may be defined by creating a nested list, e.g. [[1.0], [1.5], [2.0]] to resample the
            same image multiple times to different (here: isotropic) voxel spacings, namely 1.0, 1.5 and 2.0. Units
            are defined by the headers of the image files. These are typically millimeters for radiological images.
        :param anti_aliasing: Determines whether to perform anti-aliasing, which is done to mitigate aliasing
            artifacts when downsampling. Default: True
        :param smoothing_beta: Determines the smoothness of the Gaussian filter used for anti-aliasing. A value of
            1.00 equates no anti-aliasing, with lower values producing increasingly smooth imaging. Values above 0.90
            are recommended. Default: 0.98
        :param kwargs: Unused keyword arguments.

        :returns: A :class:`mirp.importSettings.ImageInterpolationSettingsClass` object with configured parameters.
        """

        # Set interpolate parameter.
        self.interpolate: bool = interpolate

        # Check if the spline order is valid.
        if spline_order < 0 or spline_order > 5:
            raise ValueError(f"The interpolation spline order should be an integer between 0 and 5. Found:"
                             f" {spline_order}")

        # Set spline order for the interpolating spline.
        self.spline_order: int = spline_order

        # Check
        if not interpolate:
            new_spacing = None

        else:
            # When interpolation is desired, check that the desired spacing is set.
            if new_spacing is None:
                raise ValueError("The desired voxel spacing for resampling is required if interpolation=True. "
                                 "However, no new spacing was defined.")

            # Parse value to list of floating point values to facilitate checks.
            if isinstance(new_spacing, (int, float)):
                new_spacing = [new_spacing]

            # Check if nested list elements are present.
            if any(isinstance(ii, list) for ii in new_spacing):
                new_spacing = [self._check_new_sample_spacing(by_slice=by_slice, new_spacing=new_spacing_element)
                               for new_spacing_element in new_spacing]

            else:
                new_spacing = [self._check_new_sample_spacing(by_slice=by_slice, new_spacing=new_spacing)]

            # Check that new spacing is now a nested list.
            if not all(isinstance(ii, list) for ii in new_spacing):
                raise TypeError(f"THe new_spacing variable should now be represented as a nested list.")

        # Set spacing for resampling. Note that new_spacing should now either be None or a nested list, i.e. a list
        # containing other lists.
        self.new_spacing: Union[None, List] = new_spacing

        # Set anti-aliasing.
        self.anti_aliasing: bool = anti_aliasing

        # Check that smoothing beta lies between 0.0 and 1.0.
        if anti_aliasing:
            if smoothing_beta <= 0.0 or smoothing_beta > 1.0:
                raise ValueError(f"The value of the smoothing_beta parameter should lie between 0.0 and 1.0, "
                                 f"not including 0.0. Found: {smoothing_beta}")

        # Set smoothing beta.
        self.smoothing_beta: float = smoothing_beta

    @staticmethod
    def _check_new_sample_spacing(by_slice,
                                  new_spacing):
        # Checks whether sample spacing is correctly set, and parses it.

        # Parse value to list of floating point values to facilitate checks.
        if isinstance(new_spacing, (int, float)):
            new_spacing = [new_spacing]

        # Convert to floating point values.
        new_spacing: List[Union[float, None]] = [float(new_spacing_element) for new_spacing_element in new_spacing]

        if by_slice:
            # New spacing is expect to consist of at most two values when the experiment is based on slices. A
            # placeholder for the z-direction is set here.
            if len(new_spacing) == 1:
                # This creates isotropic spacing.
                new_spacing = [None, new_spacing[0], new_spacing[0]]

            elif len(new_spacing) == 2:
                # Insert a placeholder for the z-direction.
                new_spacing.insert(0, None)

            else:
                raise ValueError(f"The desired voxel spacing for in-slice resampling should consist of two "
                                 f"elements. Found: {len(new_spacing)} elements.")
        else:
            if len(new_spacing) == 1:
                # This creates isotropic spacing.
                new_spacing = [new_spacing[0], new_spacing[0], new_spacing[0]]

            elif len(new_spacing) == 3:
                # Do nothing.
                pass

            else:
                raise ValueError(f"The desired voxel spacing for volumetric resampling should consist of three "
                                 f"elements. Found: {len(new_spacing)} elements.")

        return new_spacing


class RoiInterpolationSettingsClass:

    def __init__(self,
                 roi_spline_order: int = 1,
                 roi_interpolation_mask_inclusion_threshold: float = 0.5,
                 **kwargs):
        """
        Sets interpolation parameters for the region of interest mask. MIRP actively maps the interpolation mask to the
        image, which is interpolated prior to the mask. Therefore, parameters such as new_spacing are missing.

        :param roi_spline_order: Sets the spline order used for spline interpolation of the mask. Default: 1 (
            trilinear interpolation).
        :param roi_interpolation_mask_inclusion_threshold: Threshold for ROIs with partial volumes after
            interpolation. All voxels with a value equal to or greater than this threshold are assigned to the mask.
            Default: 0.5
        :param kwargs: Unused keyword arguments.

        :returns: A :class:`mirp.importSettings.RoiInterpolationSettingsClass` object with configured parameters.
        """

        # Check if the spline order is valid.
        if roi_spline_order < 0 or roi_spline_order > 5:
            raise ValueError(f"The interpolation spline order for the ROI should be an integer between 0 and 5. Found:"
                             f" {roi_spline_order}")

        # Set spline order.
        self.spline_order = roi_spline_order

        # Check if the inclusion threshold is between 0 and 1.
        if roi_interpolation_mask_inclusion_threshold <= 0.0 or roi_interpolation_mask_inclusion_threshold > 1.0:
            raise ValueError(f"The inclusion threshold for the ROI mask should be between 0.0 and 1.0, excluding 0.0. Found: {roi_interpolation_mask_inclusion_threshold}")

        self.incl_threshold = roi_interpolation_mask_inclusion_threshold


class ImagePostProcessingClass:

    def __init__(self,
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
                raise ValueError(f"The bias_field_correction_n_fitting_levels should be integer with value 1 or larger. "
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
                bias_field_correction_n_max_iterations = [bias_field_correction_n_max_iterations[0] for ii in range(
                    bias_field_correction_n_fitting_levels)]

            # Check that the list of maximum iteration values is equal to the number of fitting levels.
            if len(bias_field_correction_n_max_iterations) != bias_field_correction_n_fitting_levels:
                raise ValueError(f"The bias_field_correction_n_max_iterations parameter should be a list with a "
                                 f"length equal to the number of fitting levels ("
                                 f"{bias_field_correction_n_fitting_levels}). Found list with "
                                 f"{len(bias_field_correction_n_max_iterations)} values.")

            # Check that all values are integers.
            if not all(isinstance(ii, int) for ii in bias_field_correction_n_max_iterations):
                raise TypeError(f"The bias_field_correction_n_max_iterations parameter should be a list of positive "
                                f"integer values. At least one value was not an integer.")

            # Check that all values are positive.
            if not all([ii > 0 for ii in bias_field_correction_n_max_iterations]):
                raise ValueError(f"The bias_field_correction_n_max_iterations parameter should be a list of positive "
                                 f"integer values. At least one value was zero or negative.")

        else:
            bias_field_correction_n_max_iterations = None

        # Set n_max_iterations attribute.
        self.n_max_iterations: Union[List[int], None] = bias_field_correction_n_max_iterations

        # Check that the convergence threshold is a non-negative number.
        if bias_field_correction:

            # Check that the value is a float.
            if not isinstance(bias_field_convergence_threshold, float):
                raise TypeError(f"The bias_field_convergence_threshold parameter is expected to be a non-negative "
                                f"floating point value. Found: a value that was not a floating point value.")

            if bias_field_convergence_threshold <= 0.0:
                raise TypeError(f"The bias_field_convergence_threshold parameter is expected to be a non-positive "
                                f"floating point value. Found: a value that was 0.0 or negative ({bias_field_convergence_threshold}).")

        else:
            bias_field_convergence_threshold = None

        # Set convergence_threshold attribute.
        self.convergence_threshold: Union[None, float] = bias_field_convergence_threshold

        # Check that intensity_normalisation has the correct values.
        if intensity_normalisation not in ["none", "range", "relative_range", "quantile_range", "standardisation"]:
            raise ValueError(f"The intensity_normalisation parameter is expected to have one of the following values:"
                             f" 'none', 'range', 'relative_range', 'quantile_range', 'standardisation'. Found:"
                             f" {intensity_normalisation}.")

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
                raise TypeError(f"The intensity_normalisation_range parameter for range-based normalisation should "
                                f"be a list with exactly two values, which are mapped to 0.0 and 1.0 respectively. "
                                f"Found: an object that is not a list.")

            if len(intensity_normalisation_range) != 2:
                raise ValueError(f"The intensity_normalisation_range parameter for range-based normalisation should "
                                 f"be a list with exactly two values, which are mapped to 0.0 and 1.0 respectively. "
                                 f"Found: list with {len(intensity_normalisation_range)} values.")

            if not all(isinstance(ii, float) for ii in intensity_normalisation_range):
                raise TypeError(f"The intensity_normalisation_range parameter for range-based normalisation should "
                                f"be a list with exactly two floating point values, which are mapped to 0.0 and 1.0 "
                                f"respectively. Found: one or more values that are not floating point values.")

        elif intensity_normalisation in ["relative_range", "quantile_range"]:
            # Check that the range has length 2 and contains floating point values between 0.0 and 1.0.
            if intensity_normalisation == "relative_range":
                intensity_normalisation_specifier = "relative range-based normalisation"
            else:
                intensity_normalisation_specifier = "quantile range-based normalisation"

            if not isinstance(intensity_normalisation_range, list):
                raise TypeError(f"The intensity_normalisation_range parameter for {intensity_normalisation_specifier} "
                                f"should be a list with exactly two values, which are mapped to 0.0 and 1.0 "
                                f"respectively. Found: an object that is not a list.")

            if len(intensity_normalisation_range) != 2:
                raise ValueError(f"The intensity_normalisation_range parameter for {intensity_normalisation_specifier} "
                                 f"should be a list with exactly two values, which are mapped to 0.0 and 1.0 "
                                 f"respectively. Found: list with {len(intensity_normalisation_range)} values.")

            if not all(isinstance(ii, float) for ii in intensity_normalisation_range):
                raise TypeError(f"The intensity_normalisation_range parameter for {intensity_normalisation_specifier} "
                                f"should be a list with exactly two values, which are mapped to 0.0 and 1.0 "
                                f"respectively. Found: one or more values that are not floating point values.")

            if not all([0.0 <= ii <= 1.0 for ii in intensity_normalisation_range]):
                raise TypeError(f"The intensity_normalisation_range parameter for {intensity_normalisation_specifier} "
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
            raise TypeError("The tissue_mask_range parameter is expected to be a list of two floating point "
                            "values.")

        if not len(intensity_normalisation_saturation) == 2:
            raise ValueError(f"The tissue_mask_range parameter should consist of two values. Found: "
                             f"{len(intensity_normalisation_saturation)} values.")

        if not all(isinstance(ii, float) for ii in intensity_normalisation_saturation):
            raise TypeError("The tissue_mask_range parameter can only contain floating point or np.nan values.")

        # intensity_normalisation_saturation parameter
        self.intensity_normalisation_saturation: Union[None, List[float]] = intensity_normalisation_saturation

        # Check tissue_mask_type
        if tissue_mask_type not in ["none", "range", "relative_range"]:
            raise ValueError(f"The tissue_mask_type parameter is expected to have one of the following values: "
                             f"'none', 'range', or 'relative_range'. Found: {tissue_mask_type}.")

        # Set tissue_mask_type
        self.tissue_mask_type: str = tissue_mask_type

        # Set the default value for tissue_mask_range.
        if tissue_mask_range is None:
            if tissue_mask_type == "relative_range":
                tissue_mask_range = [0.02, 1.00]
            elif tissue_mask_type == "range":
                tissue_mask_range = [0.0, np.nan]

        # Perform checks on tissue_mask_range.
        if tissue_mask_type != "none":
            if not isinstance(tissue_mask_range, list):
                raise TypeError("The tissue_mask_range parameter is expected to be a list of two floating point "
                                "values.")

            if not len(tissue_mask_range) == 2:
                raise ValueError(f"The tissue_mask_range parameter should consist of two values. Found: "
                                 f"{len(tissue_mask_range)} values.")

            if not all(isinstance(ii, float) for ii in tissue_mask_range):
                raise TypeError("The tissue_mask_range parameter can only contain floating point or np.nan values.")

            if tissue_mask_type == "relative_range":
                if not all([(0.0 <= ii <= 1.0) or np.isnan(ii) for ii in tissue_mask_range]):
                    raise ValueError("The tissue_mask_range parameter should consist of two values between 0.0 and "
                                     "1.0.")

        # Set tissue_mask_range.
        self.tissue_mask_range: List[float] = tissue_mask_range


class ImagePerturbationSettingsClass:

    def __init__(self,
                 crop_around_roi: bool = False,
                 crop_distance: float = 150.0,
                 perturbation_noise_repetitions: int = 0,
                 perturbation_noise_level: Union[None, float] = None,
                 perturbation_rotation_angles: Union[None, List[float], float] = 0.0,
                 perturbation_translation_fraction: Union[None, List[float], float] = 0.0,
                 perturbation_roi_adapt_type: str = "distance",
                 perturbation_roi_adapt_size: Union[None, List[float], float] = 0.0,
                 perturbation_roi_adapt_max_erosion: float = 0.8,
                 perturbation_randomise_roi_repetitions: int = 0,
                 roi_split_boundary_size: Union[None, List[float], float] = 0.0,
                 roi_split_max_erosion: float = 0.6,
                 **kwargs):
        """
        Sets parameters for perturbing the image.

        :param crop_around_roi: Determines whether the image may be cropped around the regions of interest. Setting
            this to True may speed up calculations and save memory. Default: False.
        :param crop_distance: Physical distance around the ROI mask that should be maintained when cropping the image.
            When using convolutional kernels for filtering an image, we recommend to leave some distance to prevent
            boundary effects from interfering with the contents in the ROI. A crop distance of 0.0 crops the image
            tightly around the ROI. Default: 150 units (usually mm).
        :param perturbation_noise_repetitions: Number of times noise is randomly added to the image. Used in noise
            addition image perturbations. Default: 0 (no noise is added).
        :param perturbation_noise_level: Set the noise level in intensity units. This determines the width of the
            normal distribution used to generate random noise. If None, noise is determined from the image itself.
            Default: None
        :param perturbation_rotation_angles: Angles (in degrees) over which the image and mask are rotated. This
            rotation is only in the x-y (axial) plane. Multiple angles can be provided. Used in the rotation image
            perturbation. Default: 0.0 (no rotation)
        :param perturbation_translation_fraction: Sub-voxel translation distance fractions of the interpolation
            grid. This forces the interpolation grid to shift slightly and interpolate at different points. Multiple
            values can be provided. Value should be between 0.0 and 1.0. Used in translation perturbations.
            Default: 0.0 (no shifting).
        :param perturbation_roi_adapt_type: Determines how the ROI mask is grown or shrunk. Can be either "fraction"
            or "distance". "fraction" is used to grow or shrink the ROI mask by a certain fraction (see the
            ``perturbation_roi_adapt_size`` parameter and is used in the volume growth/shrinkage image perturbation.
            "distance" is used to grow or shrink the ROI by a certain physical distance, defined using the
            ``perturbation_roi_adapt_size`` parameter. Default: "distance"
        :param perturbation_roi_adapt_size: Determines the extent of growth/shrinkage of the ROI mask.
            The use of this parameter depends on the growth/shrinkage type (``perturbation_roi_adapt_type``),
            For "distance", this parameter defines growth/shrinkage in physical units, typically mm. For "fraction":
            growth/shrinkage in volume fraction. For either type, positive values indicate growing the ROI mask,
            whereas negative values indicate its shrinkage. Multiple values can be provided to perturb the volume of
            the ROI mask. Default: 0.0 (no changes).
        :param perturbation_roi_adapt_max_erosion: Limit to shrinkage of the ROI by distance-based adaptations.
            Fraction of the original volume. Only used when ``perturbation_roi_adapt_type=="distance"``. Default: 0.8
        :param perturbation_randomise_roi_repetitions: Number of repetitions of supervoxel-based randomisation of
            the ROI mask. Default: 0 (no changes).
        :param roi_split_boundary_size: Split ROI mask into a bulk and a boundary rim section. This parameter
            determines the width of the rim. Multiple values can be provided to generate rims of different widths.
            Default: 0.0
        :param roi_split_max_erosion: Determines the minimum volume of the bulk ROI mask when splitting the ROI into
            bulk and rim sections. Fraction of the original volume. Default: 0.6
        :param kwargs: unused keyword arguments.

        :returns: A :class:`mirp.importSettings.ImagePerturbationSettingsClass` object with configured parameters.
        """

        # Set crop_around_roi
        self.crop_around_roi: bool = crop_around_roi

        # Check that crop distance is not negative.
        if crop_distance < 0.0 and crop_around_roi:
            raise ValueError(f"The cropping distance cannot be negative. Found: {crop_distance}")

        # Set crop_distance.
        self.crop_distance: float = crop_distance

        # Check that noise repetitions is not negative.
        perturbation_noise_repetitions = int(perturbation_noise_repetitions)
        if perturbation_noise_repetitions < 0:
            raise ValueError(f"The number of repetitions where noise is added to the image cannot be negative. Found: {perturbation_noise_repetitions}")

        # Set noise repetitions.
        self.add_noise: bool = perturbation_noise_repetitions > 0
        self.noise_repetitions: int = perturbation_noise_repetitions

        # Check noise level.
        if perturbation_noise_level is not None:
            if perturbation_noise_level < 0.0:
                raise ValueError(f"The noise level cannot be negative. Found: {perturbation_noise_level}")

        # Set noise level.
        self.noise_level: Union[None, float] = perturbation_noise_level

        # Convert perturbation_rotation_angles to list, if necessary.
        if not isinstance(perturbation_rotation_angles, list):
            perturbation_rotation_angles = [perturbation_rotation_angles]

        # Check that the rotation angles are floating points.
        if not all(isinstance(ii, float) for ii in perturbation_rotation_angles):
            raise TypeError(f"Not all values for perturbation_rotation_angles are floating point values.")

        # Set rotation_angles.
        self.rotation_angles: List[float] = perturbation_rotation_angles

        # Convert perturbation_translation_fraction to list, if necessary.
        if not isinstance(perturbation_translation_fraction, list):
            perturbation_translation_fraction = [perturbation_translation_fraction]

        # Check that the translation fractions are floating points.
        if not all(isinstance(ii, float) for ii in perturbation_translation_fraction):
            raise TypeError(f"Not all values for perturbation_translation_fraction are floating point values.")

        # Check that the translation fractions lie between 0.0 and 1.0.
        if not all(0.0 <= ii < 1.0 for ii in perturbation_translation_fraction):
            raise ValueError("Not all values for perturbation_translation_fraction lie between 0.0 and 1.0, "
                             "not including 1.0.")

        # Set translation_fraction.
        self.translation_fraction: List[float] = perturbation_translation_fraction

        # Check roi adaptation type.
        if perturbation_roi_adapt_type not in ["distance", "fraction"]:
            raise ValueError(f"The perturbation ROI adaptation type should be one of 'distance' or 'fraction'. Found: {perturbation_roi_adapt_type}")

        # Set roi_adapt_type
        self.roi_adapt_type: str = perturbation_roi_adapt_type

        # Convert to perturbation_roi_adapt_size to list.
        if not isinstance(perturbation_roi_adapt_size, list):
            perturbation_roi_adapt_size = [perturbation_roi_adapt_size]

        # Check that the adapt sizes are floating points.
        if not all(isinstance(ii, float) for ii in perturbation_roi_adapt_size):
            raise TypeError(f"Not all values for perturbation_roi_adapt_size are floating point values.")

        # Check that values do not go below 0.
        if perturbation_roi_adapt_type == "fraction" and any([ii <= -1.0 for ii in perturbation_roi_adapt_size]):
            raise ValueError("All values for perturbation_roi_adapt_size should be greater than -1.0. However, "
                             "one or more values were less.")

        # Set roi_adapt_size
        self.roi_adapt_size: List[float] = perturbation_roi_adapt_size

        # Check that perturbation_roi_adapt_max_erosion is between 0.0 and 1.0.
        if not 0.0 <= perturbation_roi_adapt_max_erosion <= 1.0:
            raise ValueError(f"The perturbation_roi_adapt_max_erosion parameter must have a value between 0.0 and "
                             f"1.0. Found: {perturbation_roi_adapt_max_erosion}")

        # Set max volume erosion.
        self.max_volume_erosion: float = perturbation_roi_adapt_max_erosion

        # Check that ROI randomisation representation is not negative.
        perturbation_randomise_roi_repetitions = int(perturbation_randomise_roi_repetitions)
        if perturbation_randomise_roi_repetitions < 0:
            raise ValueError(
                f"The number of repetitions where the ROI mask is randomised cannot be negative. Found: "
                f"{perturbation_randomise_roi_repetitions}")

        # Set ROI mask randomisation repetitions.
        self.randomise_roi: bool = perturbation_randomise_roi_repetitions > 0
        self.roi_random_rep: int = perturbation_randomise_roi_repetitions

        # Check that roi_split_max_erosion is between 0.0 and 1.0.
        if not 0.0 <= roi_split_max_erosion <= 1.0:
            raise ValueError(f"The roi_split_max_erosion parameter must have a value between 0.0 and "
                             f"1.0. Found: {roi_split_max_erosion}")

        # Division of roi into bulk and boundary
        self.max_bulk_volume_erosion: float = roi_split_max_erosion

        # Convert roi_split_boundary_size to list, if necessary.
        if not isinstance(roi_split_boundary_size, list):
            roi_split_boundary_size = [roi_split_boundary_size]

        # Check that the translation fractions are floating points.
        if not all(isinstance(ii, float) for ii in roi_split_boundary_size):
            raise TypeError(f"Not all values for roi_split_boundary_size are floating point values.")

        # Check that the translation fractions lie between 0.0 and 1.0.
        if not all(ii >= 0.0 for ii in roi_split_boundary_size):
            raise ValueError("Not all values for roi_split_boundary_size are positive.")

        # Set roi_boundary_size.
        self.roi_boundary_size: List[float] = roi_split_boundary_size

        # Initially local variables
        self.translate_x: Union[None, float] = None
        self.translate_y: Union[None, float] = None
        self.translate_z: Union[None, float] = None


class ResegmentationSettingsClass:

    def __init__(self,
                 resegmentation_method: Union[str, List[str]] = "none",
                 resegmentation_intensity_range: Union[None, List[float]] = None,
                 resegmentation_sigma: float = 3.0,
                 **kwargs):
        """
        Sets parameters related to resegmentation of the ROI mask. Resegmentation is used to remove parts of the
        mask that correspond to undesired intensities that should be excluded, e.g. those corresponding to air.

        :param resegmentation_method: ROI re-segmentation method for intensity-based re-segmentation. Options are
            "none", "threshold", "range", "sigma" and "outlier". Multiple options can be provided, and re-segmentation
            will take place in the given order. "threshold" and "range" are synonyms, as well as "sigma" and "outlier".
            Default: "none"
        :param resegmentation_intensity_range: Intensity threshold for threshold-based re-segmentation ("threshold" and
            "range"). If set, requires two values for lower and upper range respectively. The upper range value can
            also be np.nan for half-open ranges. Default: None
        :param resegmentation_sigma:  Number of standard deviations for outlier-based intensity re-segmentation.
            Default: 3.0
        :param kwargs: unused keyword arguments.

        :returns: A :class:`mirp.importSettings.ResegmentationSettingsClass` object with configured parameters.
        """

        # Check values for resegmentation_method.
        if not isinstance(resegmentation_method, list):
            resegmentation_method = [resegmentation_method]

        # If "none" is present, remove all other methods.
        if "none" in resegmentation_method:
            resegmentation_method = ["none"]

        # Check that methods are valid.
        if not all([ii in ["none", "threshold", "range", "sigma", "outlier"] for ii in resegmentation_method]):
            raise ValueError("The resegmentation_method parameter can only have the following values: 'none', "
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
            raise TypeError(f"The resegmentation_intensity_range parameter should be a list with exactly two "
                            f"values. Found: an object that is not a list.")

        if len(resegmentation_intensity_range) != 2:
            raise ValueError(f"The resegmentation_intensity_range parameter should be a list with exactly two "
                             f"values. Found: list with {len(resegmentation_intensity_range)} values.")

        if not all(isinstance(ii, float) for ii in resegmentation_intensity_range):
            raise TypeError(f"The resegmentation_intensity_range parameter should be a list with exactly two "
                            f"values. Found: one or more values that are not floating point values.")

        self.intensity_range: List[float] = resegmentation_intensity_range

        # Check that sigma is not negative.
        if resegmentation_sigma < 0.0:
            raise ValueError(f"The resegmentation_sigma parameter can not be negative. Found: {resegmentation_sigma}")

        self.sigma: float = resegmentation_sigma


class FeatureExtractionSettingsClass:

    def __init__(self,
                 by_slice: bool,
                 no_approximation: bool,
                 ibsi_compliant: bool = True,
                 base_feature_families: Union[None, str, List[str]] = "all",
                 base_discretisation_method: Union[None, str, List[str]] = None,
                 base_discretisation_n_bins: Union[None, int, List[int]] = None,
                 base_discretisation_bin_width: Union[None, int, List[int]] = None,
                 ivh_discretisation_method: str = "none",
                 ivh_discretisation_n_bins: Union[None, int] = 1000,
                 ivh_discretisation_bin_width: Union[None, int] = None,
                 glcm_distance: Union[float, List[float]] = 1.0,
                 glcm_spatial_method: Union[None, str, List[str]] = None,
                 glrlm_spatial_method: Union[None, str, List[str]] = None,
                 glszm_spatial_method: Union[None, str, List[str]] = None,
                 gldzm_spatial_method: Union[None, str, List[str]] = None,
                 ngtdm_spatial_method: Union[None, str, List[str]] = None,
                 ngldm_distance: Union[float, List[float]] = 1.0,
                 ngldm_difference_level: Union[float, List[float]] = 0.0,
                 ngldm_spatial_method: Union[None, str, List[str]] = None,
                 **kwargs):
        """
        Sets feature computation parameters for computation from the base image, without the image undergoing
        convolutional filtering.

        :param by_slice: Defines whether the experiment is by slice (True) or volumetric (False).
            See :class:`mirp.importSettings.GeneralSettingsClass`.
        :param no_approximation: Disables approximation of features, such as Geary's c-measure. See
            :class:`mirp.importSettings.GeneralSettingsClass`.
        :param base_feature_families: Determines the feature families for which features are computed. Radiomics
            features are implemented as defined in the IBSI reference manual. The following feature families are
            currently present, and can be added using the tags mentioned:

            * Morphological features: "mrp", "morph", "morphology", and "morphological".
            * Local intensity features: "li", "loc.int", "loc_int", "local_int", and "local_intensity".
            * Intensity-based statistical features: "st", "stat", "stats", "statistics", and "statistical".
            * Intensity histogram features: "ih", "int_hist", "int_histogram", and "intensity_histogram".
            * Intensity-volume histogram features: "ivh", "int_vol_hist", and "intensity_volume_histogram".
            * Grey level co-occurence matrix (GLCM) features: "cm", "glcm", "grey_level_cooccurrence_matrix", and
            "cooccurrence_matrix".
            * Grey level run length matrix (GLRLM) features: "rlm", "glrlm", "grey_level_run_length_matrix", and
            "run_length_matrix".
            * Grey level size zone matrix (GLSZM) features: "szm", "glszm", "grey_level_size_zone_matrix", and
            "size_zone_matrix".
            * Grey level distance zone matrix (GLDZM) features: "dzm", "gldzm", "grey_level_distance_zone_matrix", and
            "distance_zone_matrix".
            * Neighbourhood grey tone difference matrix (NGTDM) features: "tdm", "ngtdm",
            "neighbourhood_grey_tone_difference_matrix", and "grey_tone_difference_matrix".
            * Neighbouring grey level dependence matrix (NGLDM) features: "ldm", "ngldm",
            "neighbouring_grey_level_dependence_matrix", and "grey_level_dependence_matrix".

            A list of tags may be provided to select multiple feature families.
            Default: "all" (features from all feature families are computed)
        :param base_discretisation_method: Method used for discretising intensities. Used to compute intensity
            histogram as well as texture features. "fixed_bin_size", "fixed_bin_number" and "none" methods are
            implemented. The "fixed_bin_size" method uses the lower boundary of the resegmentation range
            (``resegmentation_intensity_range``) as the edge of the initial bin. If this unset, a value of -1000.0,
            0.0 or the minimum value in the ROI are used for CT, PET and other imaging modalities respectively. From
            this starting point each bin has a fixed width defined by the ``base_discretisation_bin_width`` parameter.
            The "fixed_bin_number" method divides the intensity range within the ROI into a number of bins,
            defined by the ``base_discretisation_bin_width`` parameter. The "none" method assign each unique
            intensity value is assigned its own bin. There is no default value.
        :param base_discretisation_n_bins: Number of bins used for the "fixed_bin_number" discretisation method. No
            default value.
        :param base_discretisation_bin_width: Width of each bin in the "fixed_bin_size" discretisation method. No
            default value.
        :param ivh_discretisation_method: Discretisation method used to generate intensity bins for the
            intensity-volume histogram. One of "fixed_bin_width", "fixed_bin_size" or "none". Default: "none".
        :param ivh_discretisation_n_bins: Number of bins used for the "fixed_bin_number" discretisation method.
        Default: 1000
        :param ivh_discretisation_bin_width:  Width of each bin in the "fixed_bin_size" discretisation method. No
            default value.
        :param glcm_distance: Distance (in voxels) for GLCM for determining the neighbourhood. Chebyshev,
            or checkerboard, distance is used. A value of 1.0 will therefore consider all (diagonally) adjacent
            voxels as its neighbourhood. A list of values can be provided to compute GLCM features at different scales.
            Default: 1.0
        :param glcm_spatial_method: Determines how the cooccurrence matrices are formed and aggregated. One of the
            following:

             * "2d_average": features are calculated from all matrices then averaged [IBSI:BTW3].
             * "2d_slice_merge": matrices in the same slice are merged, features calculated and then averaged [
                IBSI:SUJT].
             * "2.5d_direction_merge": matrices for the same direction are merged, features calculated and then averaged
                [IBSI:JJUI].
             * "2.5d_volume_merge": all matrices are merged and a single feature is calculated [IBSI:ZW7Z].
             * "3d_average": features are calculated from all matrices then averaged [IBSI:ITBB].
             * "3d_volume_merge": all matrices are merged and a single feature is calculated [IBSI:IAZD].

              A list of values may be provided to extract features for multiple spatial methods.
              Default: "2d_slice_merge" (by slice) or "3d_volume_merge" (volumetric).

        :param glrlm_spatial_method:  Determines how run length matrices are formed and aggregated. One of the
            following:

             * "2d_average": features are calculated from all matrices then averaged [IBSI:BTW3].
             * "2d_slice_merge": matrices in the same slice are merged, features calculated and then averaged [
                IBSI:SUJT].
             * "2.5d_direction_merge": matrices for the same direction are merged, features calculated and then averaged
                [IBSI:JJUI].
             * "2.5d_volume_merge": all matrices are merged and a single feature is calculated [IBSI:ZW7Z].
             * "3d_average": features are calculated from all matrices then averaged [IBSI:ITBB].
             * "3d_volume_merge": all matrices are merged and a single feature is calculated [IBSI:IAZD].

              A list of values may be provided to extract features for multiple spatial methods.
              Default: "2d_slice_merge" (by slice) or "3d_volume_merge" (volumetric).

        :param glszm_spatial_method: Determines how the size zone matrices are formed and aggregated. One of "2d",
            "2.5d" or "3d". The latter is only available when a volumetric analysis is conducted. For "2d",
            features are computed from individual matrices and subsequently averaged [IBSI:8QNN]. For "2.5d" all 2D
            matrices are merged and features are computed from this single matrix [IBSI:62GR]. For "3d" features are
            computed from a single 3D matrix [IBSI:KOBO]. A list of values may be provided to extract features for
            multiple spatial methods. Default: "2d" (by slice) or "3d" (volumetric).
        :param gldzm_spatial_method: Determines how the distance zone matrices are formed and aggregated. One of "2d",
            "2.5d" or "3d". The latter is only available when a volumetric analysis is conducted. For "2d",
            features are computed from individual matrices and subsequently averaged [IBSI:8QNN]. For "2.5d" all 2D
            matrices are merged and features are computed from this single matrix [IBSI:62GR]. For "3d" features are
            computed from a single 3D matrix [IBSI:KOBO]. A list of values may be provided to extract features for
            multiple spatial methods. Default: "2d" (by slice) or "3d" (volumetric).
        :param ngtdm_spatial_method: Determines how the neighbourhood grey tone difference matrices are formed and
            aggregated. One of "2d", "2.5d" or "3d". The latter is only available when a volumetric analysis is
            conducted. For "2d", features are computed from individual matrices and subsequently averaged [IBSI:8QNN].
            For "2.5d" all 2D matrices are merged and features are computed from this single matrix [IBSI:62GR].
            For "3d" features are computed from a single 3D matrix [IBSI:KOBO]. A list of values may be provided to
            extract features for multiple spatial methods. Default: "2d" (by slice) or "3d" (volumetric).
        :param ngldm_distance: Distance (in voxels) for NGLDM for determining the neighbourhood. Chebyshev,
            or checkerboard, distance is used. A value of 1.0 will therefore consider all (diagonally) adjacent
            voxels as its neighbourhood. A list of values can be provided to compute NGLDM features at different scales.
            Default: 1.0
        :param ngldm_difference_level: Difference level (alpha) for NGLDM. Determines which discretisations are
            grouped together in the matrix.
        :param ngldm_spatial_method: Determines how the neighbourhood grey level dependence matrices are formed and
            aggregated. One of "2d", "2.5d" or "3d". The latter is only available when a volumetric analysis is
            conducted. For "2d", features are computed from individual matrices and subsequently averaged [IBSI:8QNN].
            For "2.5d" all 2D matrices are merged and features are computed from this single matrix [IBSI:62GR].
            For "3d" features are computed from a single 3D matrix [IBSI:KOBO]. A list of values may be provided to
            extract features for multiple spatial methods. Default: "2d" (by slice) or "3d" (volumetric).
        :param kwargs: unused keyword arguments.

        :returns: A :class:`mirp.importSettings.FeatureExtractionSettingsClass` object with configured parameters.
        """
        # Set by slice.
        self.by_slice: bool = by_slice

        # Set approximation flag.
        self.no_approximation: bool = no_approximation

        # Set IBSI-compliance flag.
        self.ibsi_compliant: bool = ibsi_compliant

        if base_feature_families is None:
            base_feature_families = "none"

        # Check families.
        if not isinstance(base_feature_families, list):
            base_feature_families = [base_feature_families]

        # Check which entries are valid.
        valid_families: List[bool] = [ii in [
            "mrp", "morph", "morphology", "morphological", "li", "loc.int", "loc_int", "local_int", "local_intensity",
            "st", "stat", "stats", "statistics", "statistical", "ih", "int_hist", "int_histogram", "intensity_histogram",
            "ivh", "int_vol_hist", "intensity_volume_histogram", "cm", "glcm", "grey_level_cooccurrence_matrix",
            "cooccurrence_matrix", "rlm", "glrlm", "grey_level_run_length_matrix", "run_length_matrix",
            "szm", "glszm", "grey_level_size_zone_matrix", "size_zone_matrix", "dzm", "gldzm",
            "grey_level_distance_zone_matrix", "distance_zone_matrix", "tdm", "ngtdm",
            "neighbourhood_grey_tone_difference_matrix", "grey_tone_difference_matrix", "ldm", "ngldm",
            "neighbouring_grey_level_dependence_matrix", "grey_level_dependence_matrix", "all", "none"
        ] for ii in base_feature_families]

        if not all(valid_families):
            raise ValueError(f"One or more families in the base_feature_families parameter were not recognised: "
                             f"{', '.join([base_feature_families[ii] for ii, is_valid in enumerate(valid_families) if not is_valid])}")

        # Set families.
        self.families: List[str] = base_feature_families

        if self.has_discretised_family():
            # Check if discretisation_method is None.
            if base_discretisation_method is None:
                raise ValueError("The base_discretisation_method parameter has no default and must be set.")

            if not isinstance(base_discretisation_method, list):
                base_discretisation_method = [base_discretisation_method]

            if not all(discretisation_method in ["fixed_bin_size", "fixed_bin_number", "none"] for
                       discretisation_method in base_discretisation_method):
                raise ValueError("Available values for the base_discretisation_method parameter are 'fixed_bin_size', "
                                 "'fixed_bin_number', and 'none'. One or more values were not recognised.")

            # Check discretisation_n_bins
            if "fixed_bin_number" in base_discretisation_method:
                if base_discretisation_n_bins is None:
                    raise ValueError("The base_discretisation_n_bins parameter has no default and must be set")

                if not isinstance(base_discretisation_n_bins, list):
                    base_discretisation_n_bins = [base_discretisation_n_bins]

                if not all(isinstance(n_bins, int) for n_bins in base_discretisation_n_bins):
                    raise TypeError("The base_discretisation_n_bins parameter is expected to contain integers with "
                                    "value 2 or larger. Found one or more values that were not integers.")

                if not all(n_bins >= 2 for n_bins in base_discretisation_n_bins):
                    raise ValueError("The base_discretisation_n_bins parameter is expected to contain integers with "
                                     "value 2 or larger. Found one or more values that were less than 2.")

            else:
                base_discretisation_n_bins = None

            # Check discretisation_bin_width
            if "fixed_bin_size" in base_discretisation_method:
                if base_discretisation_bin_width is None:
                    raise ValueError("The base_discretisation_bin_width parameter has no default and must be set")

                if not isinstance(base_discretisation_bin_width, list):
                    base_discretisation_bin_width = [base_discretisation_bin_width]

                if not all(isinstance(bin_size, float) for bin_size in base_discretisation_bin_width):
                    raise TypeError("The base_discretisation_bin_width parameter is expected to contain floating "
                                    "point values greater than 0.0. Found one or more values that were not floating "
                                    "points.")

                if not all(bin_size > 0.0 for bin_size in base_discretisation_bin_width):
                    raise ValueError("The base_discretisation_bin_width parameter is expected to contain floating "
                                     "point values greater than 0.0. Found one or more values that were 0.0 or less.")

            else:
                base_discretisation_bin_width = None

        else:
            base_discretisation_method = None
            base_discretisation_n_bins = None
            base_discretisation_bin_width = None

        # Set discretisation method-related parameters.
        self.discretisation_method: Union[None, List[str]] = base_discretisation_method
        self.discretisation_n_bins: Union[None, List[int]] = base_discretisation_n_bins
        self.discretisation_bin_width: Union[None, List[float]] = base_discretisation_bin_width

        if self.has_ivh_family():
            if ivh_discretisation_method not in ["fixed_bin_size", "fixed_bin_number", "none"]:
                raise ValueError("Available values for the ivh_discretisation_method parameter are 'fixed_bin_size', "
                                 "'fixed_bin_number', and 'none'. One or more values were not recognised.")

            # Check discretisation_n_bins
            if "fixed_bin_number" in ivh_discretisation_method:

                if not isinstance(ivh_discretisation_n_bins, int):
                    raise TypeError("The ivh_discretisation_n_bins parameter is expected to be an integer with "
                                    "value 2 or greater. Found: a value that was not an integer.")

                if not ivh_discretisation_n_bins >= 2:
                    raise ValueError("The ivh_discretisation_n_bins parameter is expected to be an integer with "
                                     f"value 2 or greater. Found: {ivh_discretisation_n_bins}")

            else:
                ivh_discretisation_n_bins = None

            # Check discretisation_bin_width
            if "fixed_bin_size" in ivh_discretisation_method:

                if not isinstance(ivh_discretisation_bin_width, float):
                    raise TypeError("The ivh_discretisation_bin_width parameter is expected to be a floating "
                                    "point value greater than 0.0. Found a value that was not a floating "
                                    "point.")

                if not ivh_discretisation_bin_width > 0.0:
                    raise ValueError("The ivh_discretisation_bin_width parameter is expected to  be a floating "
                                     f"point value greater than 0.0. Found: {ivh_discretisation_bin_width}")

            else:
                ivh_discretisation_bin_width = None

        else:
            ivh_discretisation_method = None
            ivh_discretisation_n_bins = None
            ivh_discretisation_bin_width = None

        # Set parameters
        self.ivh_discretisation_method: Union[None, str] = ivh_discretisation_method
        self.ivh_discretisation_n_bins: Union[None, int] = ivh_discretisation_n_bins
        self.ivh_discretisation_bin_width: Union[None, float] = ivh_discretisation_bin_width

        # Set GLCM attributes.
        if self.has_glcm_family():
            # Check distance parameter.
            if not isinstance(glcm_distance, list):
                glcm_distance = [glcm_distance]

            if not all(isinstance(distance, float) for distance in glcm_distance):
                raise TypeError("The glcm_distance parameter is expected to contain floating point values of 1.0 "
                                "or greater. Found one or more values that were not floating points.")

            if not all(distance >= 1.0 for distance in glcm_distance):
                raise ValueError("The glcm_distance parameter is expected to contain floating point values of 1.0 "
                                 "or greater. Found one or more values that were less than 1.0.")

            # Check spatial method.
            glcm_spatial_method = self.check_valid_directional_spatial_method(glcm_spatial_method,
                                                                              "glcm_spatial_method")

        else:
            glcm_distance = None
            glcm_spatial_method = None

        self.glcm_distance: Union[None, List[float]] = glcm_distance
        self.glcm_spatial_method: Union[None, List[str]] = glcm_spatial_method

        # Set GLRLM attributes.
        if self.has_glrlm_family():
            # Check spatial method.
            glrlm_spatial_method = self.check_valid_directional_spatial_method(glrlm_spatial_method,
                                                                               "glrlm_spatial_method")

        else:
            glrlm_spatial_method = None

        self.glrlm_spatial_method: Union[None, List[str]] = glrlm_spatial_method

        # Set GLSZM attributes.
        if self.has_glszm_family():
            # Check spatial method.
            glszm_spatial_method = self.check_valid_omnidirectional_spatial_method(glszm_spatial_method,
                                                                                   "glszm_spatial_method")
        else:
            glszm_spatial_method = None

        self.glszm_spatial_method: Union[None, List[str]] = glszm_spatial_method

        # Set GLDZM attributes.
        if self.has_gldzm_family():
            # Check spatial method.
            gldzm_spatial_method = self.check_valid_omnidirectional_spatial_method(gldzm_spatial_method,
                                                                                   "gldzm_spatial_method")

        else:
            gldzm_spatial_method = None

        self.gldzm_spatial_method: Union[None, List[str]] = gldzm_spatial_method

        # Set NGTDM attributes.
        if self.has_ngtdm_family():
            # Check spatial method
            ngtdm_spatial_method = self.check_valid_omnidirectional_spatial_method(ngtdm_spatial_method,
                                                                                   "ngtdm_spatial_method")

        else:
            ngtdm_spatial_method = None

        self.ngtdm_spatial_method: Union[None, List[str]] = ngtdm_spatial_method

        # Set NGLDM attributes
        if self.has_ngldm_family():

            # Check distance.
            if not isinstance(ngldm_distance, list):
                ngldm_distance = [ngldm_distance]

            if not all(isinstance(distance, float) for distance in ngldm_distance):
                raise TypeError("The ngldm_distance parameter is expected to contain floating point values of 1.0 "
                                "or greater. Found one or more values that were not floating points.")

            if not all(distance >= 1.0 for distance in ngldm_distance):
                raise ValueError("The ngldm_distance parameter is expected to contain floating point values of 1.0 "
                                 "or greater. Found one or more values that were less than 1.0.")

            # Check spatial method
            ngldm_spatial_method = self.check_valid_omnidirectional_spatial_method(ngldm_spatial_method,
                                                                                   "ngldm_spatial_method")

            # Check difference level.
            if not isinstance(ngldm_difference_level, list):
                ngldm_difference_level = [ngldm_difference_level]

            if not all(isinstance(difference, float) for difference in ngldm_difference_level):
                raise TypeError("The ngldm_difference_level parameter is expected to contain floating point values of 0.0 "
                                "or greater. Found one or more values that were not floating points.")

            if not all(difference >= 0.0 for difference in ngldm_difference_level):
                raise ValueError("The ngldm_difference_level parameter is expected to contain floating point values "
                                 "of 0.0 or greater. Found one or more values that were less than 0.0.")

        else:
            ngldm_spatial_method = None
            ngldm_distance = None
            ngldm_difference_level = None

        self.ngldm_dist: Union[None, List[float]] = ngldm_distance
        self.ngldm_diff_lvl: Union[None, List[float]] = ngldm_difference_level
        self.ngldm_spatial_method: Union[None, List[str]] = ngldm_spatial_method

    def has_discretised_family(self):
        return self.has_ih_family() or self.has_glcm_family() or self.has_glrlm_family() or self.has_glszm_family() \
               or self.has_gldzm_family() or self.has_ngtdm_family() or self.has_ngldm_family()

    def has_morphology_family(self):
        return any(family in ["mrp", "morph", "morphology", "morphological", "all"] for family in self.families)

    def has_local_intensity_family(self):
        return any(family in ["li", "loc.int", "loc_int", "local_int", "local_intensity", "all"] for family in self.families)

    def has_stats_family(self):
        return any(family in ["st", "stat", "stats", "statistics", "statistical", "all"] for family in self.families)

    def has_ih_family(self):
        return any(family in ["ih", "int_hist", "int_histogram", "intensity_histogram", "all"] for family in self.families)

    def has_ivh_family(self):
        return any(family in ["ivh", "int_vol_hist", "intensity_volume_histogram", "all"] for family in self.families)

    def has_glcm_family(self):
        return any(family in ["cm", "glcm", "grey_level_cooccurrence_matrix", "cooccurrence_matrix", "all"] for family in self.families)

    def has_glrlm_family(self):
        return any(family in ["rlm", "glrlm", "grey_level_run_length_matrix", "run_length_matrix", "all"] for family in self.families)

    def has_glszm_family(self):
        return any(family in ["szm", "glszm", "grey_level_size_zone_matrix", "size_zone_matrix", "all"] for family in self.families)

    def has_gldzm_family(self):
        return any(family in ["dzm", "gldzm", "grey_level_distance_zone_matrix", "distance_zone_matrix", "all"] for family in self.families)

    def has_ngtdm_family(self):
        return any(family in ["tdm", "ngtdm", "neighbourhood_grey_tone_difference_matrix", "grey_tone_difference_matrix", "all"] for family in self.families)

    def has_ngldm_family(self):
        return any(family in ["ldm", "ngldm", "neighbouring_grey_level_dependence_matrix", "grey_level_dependence_matrix", "all"] for family in self.families)

    def check_valid_directional_spatial_method(self, x, var_name):

        # Set defaults
        if x is None and self.by_slice:
            x = ["2d_slice_merge"]

        elif x is None and not self.by_slice:
            x = ["3d_volume_merge"]

        # Check that x is a list.
        if not isinstance(x, list):
            x = [x]

        all_spatial_method = ["2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge"]
        if not self.by_slice:
            all_spatial_method += ["3d_average", "3d_volume_merge"]

        # Check that x contains strings.
        if not all(isinstance(spatial_method, str) for spatial_method in x):
            raise TypeError(f"The {var_name} parameter expects one or more of the following values: "
                            f"{', '.join(all_spatial_method)}."
                            f" Found: one or more values that were not strings.")

        # Check spatial method.
        valid_spatial_method = [spatial_method in all_spatial_method for spatial_method in x]

        if not all(valid_spatial_method):
            raise ValueError(f"The {var_name} parameter expects one or more of the following values: "
                             f"{', '.join(all_spatial_method)}. Found: "
                             f"{', '.join([spatial_method for spatial_method in x if spatial_method in all_spatial_method])}")

        return x

    def check_valid_omnidirectional_spatial_method(self, x, var_name):

        # Set defaults
        if x is None and self.by_slice:
            x = ["2d"]

        elif x is None and not self.by_slice:
            x = ["3d"]

        # Check that x is a list.
        if not isinstance(x, list):
            x = [x]

        all_spatial_method = ["2d", "2.5d"]
        if not self.by_slice:
            all_spatial_method += ["3d"]

        # Check that x contains strings.
        if not all(isinstance(spatial_method, str) for spatial_method in x):
            raise TypeError(f"The {var_name} parameter expects one or more of the following values: "
                            f"{', '.join(all_spatial_method)}. "
                            f"Found: one or more values that were not strings.")

        # Check spatial method.
        valid_spatial_method = [spatial_method in all_spatial_method for spatial_method in x]

        if not all(valid_spatial_method):
            raise ValueError(f"The {var_name} parameter expects one or more of the following values: "
                             f"{', '.join(all_spatial_method)}. Found: "
                             f"{', '.join([spatial_method for spatial_method in x if spatial_method in all_spatial_method])}")

        return x


class ImageTransformationSettingsClass:

    def __init__(self,
                 by_slice: bool,
                 response_map_feature_settings: Union[FeatureExtractionSettingsClass, None],
                 response_map_feature_families: Union[None, str, List[str]] = "statistical",
                 response_map_discretisation_method: Union[None, str, List[str]] = None,
                 response_map_discretisation_n_bins: Union[None, int, List[int]] = None,
                 response_map_discretisation_bin_width: Union[None, int, List[int]] = None,
                 filter_kernels: Union[None, str, List[str]] = None,
                 boundary_condition: Union[None, str] = "mirror",
                 separable_wavelet_families: Union[None, str, List[str]] = None,
                 separable_wavelet_set: Union[None, str, List[str]] = None,
                 separable_wavelet_stationary: bool = True,
                 separable_wavelet_decomposition_level: Union[None, int, List[int]] = 1,
                 separable_wavelet_rotation_invariance: bool = True,
                 separable_wavelet_pooling_method: str = "max",
                 separable_wavelet_boundary_condition: Union[None, str] = None,
                 nonseparable_wavelet_families: Union[None, str, List[str]] = None,
                 nonseparable_wavelet_decomposition_level: Union[None, int, List[int]] = 1,
                 nonseparable_wavelet_response: Union[None, str] = "real",
                 nonseparable_wavelet_boundary_condition: Union[None, str] = None,
                 gaussian_sigma: Union[None, float, List[float]] = None,
                 gaussian_kernel_truncate: Union[None, float] = 4.0,
                 gaussian_kernel_boundary_condition: Union[None, str] = None,
                 laplacian_of_gaussian_sigma: Union[None, float, List[float]] = None,
                 laplacian_of_gaussian_kernel_truncate: Union[None, float] = 4.0,
                 laplacian_of_gaussian_pooling_method: str = "none",
                 laplacian_of_gaussian_boundary_condition: Union[None, str] = None,
                 laws_kernel: Union[None, str, List[str]] = None,
                 laws_delta: Union[int, List[int]] = 7,
                 laws_compute_energy: bool = True,
                 laws_rotation_invariance: bool = True,
                 laws_pooling_method: str = "max",
                 laws_boundary_condition: Union[None, str] = None,
                 gabor_sigma: Union[None, float, List[float]] = None,
                 gabor_lambda: Union[None, float, List[float]] = None,
                 gabor_kernel_truncate: float = 10.0,
                 gabor_gamma: Union[None, float, List[float]] = 1.0,
                 gabor_theta: Union[None, float, List[float]] = 0.0,
                 gabor_theta_step: Union[None, float] = None,
                 gabor_response: str = "modulus",
                 gabor_rotation_invariance: bool = False,
                 gabor_pooling_method: str = "max",
                 gabor_boundary_condition: Union[None, str] = None,
                 mean_filter_kernel_size: Union[None, int, List[int]] = None,
                 mean_filter_boundary_condition: Union[None, str] = None,
                 riesz_filter_order: Union[None, int, List[int]] = None,
                 riesz_filter_tensor_sigma: Union[None, float, List[float]] = None,
                 **kwargs
                 ):
        """
        Sets parameters for filter transformation.

        :param by_slice: Defines whether the experiment is by slice (True) or volumetric (False).
            See :class:`mirp.importSettings.GeneralSettingsClass`.
        :param response_map_feature_families: Determines the feature families for which features are computed. Radiomics
            features are implemented as defined in the IBSI reference manual. The following feature families are
            currently present, and can be added using the tags mentioned:

            * Local intensity features: "li", "loc.int", "loc_int", "local_int", and "local_intensity".
            * Intensity-based statistical features: "st", "stat", "stats", "statistics", and "statistical".
            * Intensity histogram features: "ih", "int_hist", "int_histogram", and "intensity_histogram".
            * Intensity-volume histogram features: "ivh", "int_vol_hist", and "intensity_volume_histogram".
            * Grey level co-occurence matrix (GLCM) features: "cm", "glcm", "grey_level_cooccurrence_matrix", and
            "cooccurrence_matrix".
            * Grey level run length matrix (GLRLM) features: "rlm", "glrlm", "grey_level_run_length_matrix", and
            "run_length_matrix".
            * Grey level size zone matrix (GLSZM) features: "szm", "glszm", "grey_level_size_zone_matrix", and
            "size_zone_matrix".
            * Grey level distance zone matrix (GLDZM) features: "dzm", "gldzm", "grey_level_distance_zone_matrix", and
            "distance_zone_matrix".
            * Neighbourhood grey tone difference matrix (NGTDM) features: "tdm", "ngtdm",
            "neighbourhood_grey_tone_difference_matrix", and "grey_tone_difference_matrix".
            * Neighbouring grey level dependence matrix (NGLDM) features: "ldm", "ngldm",
            "neighbouring_grey_level_dependence_matrix", and "grey_level_dependence_matrix".

            A list of tags may be provided to select multiple feature families. Note that morphological features
            cannot be computed for response maps, because these are mostly identical to morphological features
            computed from the base image.
            Default: "statistics" (intensity-based statistical features)
        :param response_map_discretisation_method: Method used for discretising intensities in the response map.
            Used to compute intensity
            histogram as well as texture features. "fixed_bin_size", "fixed_bin_number" and "none" methods are
            implemented. The "fixed_bin_size" method uses the lower boundary of the resegmentation range
            (``resegmentation_intensity_range``) as the edge of the initial bin. If this unset, a value of -1000.0,
            0.0 or the minimum value in the ROI are used for CT, PET and other imaging modalities respectively. From
            this starting point each bin has a fixed width defined by the ``base_discretisation_bin_width`` parameter.
            The "fixed_bin_number" method divides the intensity range within the ROI into a number of bins,
            defined by the ``base_discretisation_bin_width`` parameter. The "none" method assign each unique
            intensity value is assigned its own bin.  Use of "fixed_bin_width" is generally discouraged because most
            filters lack a meaningful range. Default: "fixed_bin_number".
        :param response_map_discretisation_n_bins: Number of bins used for the "fixed_bin_number" discretisation method. No
            default value.
        :param response_map_discretisation_bin_width: Width of each bin in the "fixed_bin_size" discretisation method. No
            default value.
        :param filter_kernels: Names of the filter kernels for which response maps should be created. The following
            filters are supported:

            * Mean filters: "mean"
            * Gaussian filters: "gaussian", "riesz_gaussian", and "riesz_steered_gaussian"
            * Laplacian-of-Gaussian filters: "laplacian_of_gaussian", "log", "riesz_laplacian_of_gaussian",
                "riesz_log", "riesz_steered_laplacian_of_gaussian", and "riesz_steered_log".
            * Laws kernels: "laws"
            * Gabor kernels: "gabor", "riesz_gabor", and "riesz_steered_gabor"
            * Separable wavelets: "separable_wavelet"
            * Non-separable wavelets: "nonseparable_wavelet", "riesz_nonseparable_wavelet",
                and "riesz_steered_nonseparable_wavelet"

            Filters with names that preceded by "riesz" undergo a Riesz transformation. If the filter name is
            preceded by "riesz_steered", a steerable riesz filter is used. Riesz transformation and steerable riesz
            transformations are experimental.

            More than one filter name can be provided. Default: None (no response maps are created)

        :param boundary_condition: Sets the boundary condition, which determines how filter kernels behave at the
            edge of the image. MIRP uses the same nomenclature for boundary conditions as scipy.ndimage: "reflect",
            "constant", "nearest", "mirror", "wrap". Default: "mirror".
        :param separable_wavelet_families: Name of separable wavelet kernels as implemented in pywavelets. No default.
        :param separable_wavelet_set: Filter orientation of separable wavelets. Allows for specifying combinations
            for high and low-pass filters. For 2D filters, the following sets are possible: "hh", "hl", "lh",
            "ll" (y-x directions). For 3D filters, the set of possibilities is larger: "hhh", "hhl", "hlh", "lhh",
            "hll", "lhl", "llh", "lll". More than one orientation may be set. Default: "hh" (2d) or "hhh (3d).
        :param separable_wavelet_stationary: Determines if wavelets are stationary or not. Stationary wavelets
            maintain the image dimensions after decomposition. Default: True
        :param separable_wavelet_decomposition_level: Decomposition level. For the first decomposition level,
            the base image is used as input to generate a response map. For decomposition levels greater than 1,
            the low-pass image from the previous level is used as input. More than 1 value may be specified in a list.
            Default: 1
        :param separable_wavelet_rotation_invariance: Determines whether separable filters are applied in a
            pseudo-rotational invariant manner. This generates permutations of the filter and, as a consequence,
            additional response maps. These maps are then merged using the pooling method (
            ``separable_wavelet_pooling_method``).
            Default: True
        :param separable_wavelet_pooling_method: Determines the method used for pooling response maps from permuted
            filters. Options are: "max", "min", "mean", "sum". Default: "max".
        :param separable_wavelet_boundary_condition: Sets the boundary condition for separable wavelets. This
            supersedes any value set by the general ``boundary_condition`` parameter.
            Default: same as ``boundary_condition``.
        :param nonseparable_wavelet_families: Name of non-separable wavelet kernels. Currently "shannon" and
            "simoncelli". No default.
        :param nonseparable_wavelet_decomposition_level: Decomposition level. Unlike the decomposition level
            in separable wavelets, decomposition of non-separable wavelets is purely a filter-based operation.
        :param nonseparable_wavelet_response: Type of response map created by nonseparable wavelet filters.
            Nonseparable wavelets produce response maps with complex numbers. The complex-valued response map is
            converted to a real-valued response map using the specified method; one of "modulus", "abs", "magnitude",
             "angle", "phase", "argument", "real", "imaginary". Default: "real"
        :param nonseparable_wavelet_boundary_condition: Sets the boundary condition for non-separable wavelets. This
            supersedes any value set by the general ``boundary_condition`` parameter. Default: same as ``boundary_condition``.
        :param gaussian_sigma: Width of the Gaussian filter in physical dimensions (e.g. mm). Multiple
            values can be specified. No default.
        :param gaussian_kernel_truncate: Width, in sigma, at which the filter is truncated. Default: 4.0
        :param gaussian_kernel_boundary_condition: Sets the boundary condition for the Gaussian filter. This
            supersedes any value set by the general ``boundary_condition`` parameter. Default: same as ``boundary_condition``.
        :param laplacian_of_gaussian_sigma: Width of the Gaussian filter in physical dimensions (e.g. mm). Multiple
            values can be specified. No default.
        :param laplacian_of_gaussian_kernel_truncate: Width, in sigma, at which the filter is truncated. Default: 4.0
        :param laplacian_of_gaussian_pooling_method: Determines whether and how response maps for filters with
            different widths are pooled. Default: "none"
        :param laplacian_of_gaussian_boundary_condition: Sets the boundary condition for the Laplacian-of-Gaussian
            filter. This supersedes any value set by the general ``boundary_condition`` parameter. Default: same as
            ``boundary_condition``.
        :param laws_kernel: Compute specific Laws kernels these typically are specific combinations of kernels such
        as L5S5E5, E5E5E5. No default.
        :param laws_delta: Delta for chebyshev distance between center voxel and neighbourhood boundary used to
            calculate energy maps: integer, default: 7
        :param laws_compute_energy: Determine whether an energy image should be computed, or just the response map.
            Default: True
        :param laws_rotation_invariance: Determines whether separable filters are applied in a
            pseudo-rotational invariant manner. This generates permutations of the filter and, as a consequence,
            additional response maps. These maps are then merged using the pooling method (
            ``laws_pooling_method``).
            Default: True
        :param laws_pooling_method: Determines the method used for pooling response maps from permuted
            filters. Options are: "max", "min", "mean", "sum". Default: "max".
        :param laws_boundary_condition: Sets the boundary condition for Laws kernels. This supersedes any
        value set by the general ``boundary_condition`` parameter. Default: same as ``boundary_condition``.
        :param gabor_sigma: Width of the Gaussian envelope in physical dimensions (e.g. mm). No default.
        :param gabor_lambda: Wavelength of the oscillator. No default.
        :param gabor_kernel_truncate: Width, in sigma, at which the filter is truncated. Default: 10.0
        :param gabor_gamma: Eccentricity parameter of the Gaussian envelope of the Gabor kernel. Defines width of y-axis
            relative to x-axis for 0-angle Gabor kernel. Default: 1.0
        :param gabor_theta: Initial angle of the Gabor filter in degrees. Default: 0.0
        :param gabor_theta_step: Angle step size in degrees for in-plane rotational invariance. A value of 0.0 or None (
            default) disables stepping. Default: None
        :param gabor_response: Type of response map created by Gabor filters. Gabor kernels consist of complex
            numbers, and the directly computed response map will be complex as well. The complex-valued response map is
            converted to a real-valued response map using the specified method; one of "modulus", "abs", "magnitude",
             "angle", "phase", "argument", "real", "imaginary". Default: "modulus"
        :param gabor_rotation_invariance: Determines whether (2D) Gabor filters are applied in a
            pseudo-rotational invariant manner. If True, Gabor filters are applied in each of the orthogonal planes.
            Default: False
        :param gabor_pooling_method: Determines the method used for pooling response maps from permuted
            filters. Options are: "max", "min", "mean", "sum". Default: "max".
        :param gabor_boundary_condition: Sets the boundary condition for Gabor filter. This supersedes any value set
            by the general ``boundary_condition`` parameter. Default: same as ``boundary_condition``.
        :param mean_filter_kernel_size: Length of the kernel in pixels.
        :param mean_filter_boundary_condition: Sets the boundary condition for mean filters. This supersedes any value
            set by the general ``boundary_condition`` parameter. Default: same as ``boundary_condition``.
        :param riesz_filter_order: Riesz-transformation order. If required, should be a 2 (2D filter), or 3-element (3D filter) integer
             vector, e.g. [0,0,1]. Multiple sets can be provided by nesting the list, e.g. [[0, 0, 1],
             [0, 1, 0]]. If an integer is provided, a set of filters is created. For example when
             riesz_filter_order = 2 and a 2D filter is used, the following Riesz-transformations are performed: [2,
             0], [1, 1] and [0, 2].  Note: order is (z, y, x). No default.
        :param riesz_filter_tensor_sigma: Determines width of Gaussian filter used with Riesz filter banks. No default.
        :param kwargs: unused keyword arguments.

        :returns: A :class:`mirp.importSettings.ImageTransformationSettingsClass` object with configured parameters.
        """
        # Set by slice
        self.by_slice: bool = by_slice

        # Check filter kernels
        if not isinstance(filter_kernels, list):
            filter_kernels = [filter_kernels]

        if any(filter_kernel is None for filter_kernel in filter_kernels):
            filter_kernels = None

        if filter_kernels is not None:
            # Check validity of the filter kernel names.
            valid_kernels: List[bool] = [ii in [
                "separable_wavelet", "nonseparable_wavelet", "riesz_nonseparable_wavelet",
                "riesz_steered_nonseparable_wavelet", "gaussian",
                "riesz_gaussian", "riesz_steered_gaussian", "laplacian_of_gaussian", "log",
                "riesz_laplacian_of_gaussian", "riesz_steered_laplacian_of_gaussian", "riesz_log",
                "riesz_steered_log", "laws", "gabor", "riesz_gabor", "riesz_steered_gabor", "mean"] for ii in
                                         filter_kernels]

            if not all(valid_kernels):
                raise ValueError(f"One or more kernels are not implemented, or were spelled incorrectly: "
                                 f"{', '.join([filter_kernel for ii, filter_kernel in filter_kernels if not valid_kernels[ii]])}")

        self.spatial_filters: Union[None, List[str]] = filter_kernels

        # Check families.
        if response_map_feature_families is None:
            response_map_feature_families = "none"

        if not isinstance(response_map_feature_families, list):
            response_map_feature_families = [response_map_feature_families]

        # Check which entries are valid.
        valid_families: List[bool] = [ii in [
            "li", "loc.int", "loc_int", "local_int", "local_intensity", "st", "stat", "stats", "statistics",
            "statistical", "ih", "int_hist", "int_histogram", "intensity_histogram",
            "ivh", "int_vol_hist", "intensity_volume_histogram", "cm", "glcm", "grey_level_cooccurrence_matrix",
            "cooccurrence_matrix", "rlm", "glrlm", "grey_level_run_length_matrix", "run_length_matrix",
            "szm", "glszm", "grey_level_size_zone_matrix", "size_zone_matrix", "dzm", "gldzm",
            "grey_level_distance_zone_matrix", "distance_zone_matrix", "tdm", "ngtdm",
            "neighbourhood_grey_tone_difference_matrix", "grey_tone_difference_matrix", "ldm", "ngldm",
            "neighbouring_grey_level_dependence_matrix", "grey_level_dependence_matrix", "all", "none"
        ] for ii in response_map_feature_families]

        if not all(valid_families):
            raise ValueError(f"One or more families in the base_feature_families parameter were not recognised: "
                             f"{', '.join([response_map_feature_families[ii] for ii, is_valid in enumerate(valid_families) if not is_valid])}")

        # Create a temporary feature settings object. If response_map_feature_settings is not present, this object is
        # used. Otherwise, response_map_feature_settings is copied, and then updated.
        temp_feature_settings = FeatureExtractionSettingsClass(
            by_slice=by_slice,
            no_approximation=False,
            base_feature_families=response_map_feature_families,
            base_discretisation_method=response_map_discretisation_method,
            base_discretisation_bin_width=response_map_discretisation_bin_width,
            base_discretisation_n_bins=response_map_discretisation_n_bins)

        if response_map_feature_settings is not None:
            filter_feature_settings = copy.deepcopy(response_map_feature_settings)
            filter_feature_settings.families = temp_feature_settings.families
            filter_feature_settings.discretisation_method = temp_feature_settings.discretisation_method
            filter_feature_settings.discretisation_n_bins = temp_feature_settings.discretisation_n_bins
            filter_feature_settings.discretisation_bin_width = temp_feature_settings.discretisation_bin_width

        else:
            filter_feature_settings = temp_feature_settings

        # Set feature settings.
        self.feature_settings: FeatureExtractionSettingsClass = filter_feature_settings

        # Check boundary condition.
        self.boundary_condition = boundary_condition
        self.boundary_condition: str = self.check_boundary_condition(boundary_condition,
                                                                     "boundary_condition")

        # Check mean filter settings
        if self.has_mean_filter():
            # Check filter size.
            if not isinstance(mean_filter_kernel_size, list):
                mean_filter_kernel_size = [mean_filter_kernel_size]

            if not all(isinstance(kernel_size, int) for kernel_size in mean_filter_kernel_size):
                raise TypeError(f"All kernel sizes for the mean filter are expected to be integer values equal or "
                                f"greater than 1. Found: one or more kernel sizes that were not integers.")

            if not all(kernel_size >= 1 for kernel_size in mean_filter_kernel_size):
                raise ValueError(f"All kernel sizes for the mean filter are expected to be integer values equal or "
                                 f"greater than 1. Found: one or more kernel sizes less then 1.")

            # Check boundary condition
            mean_filter_boundary_condition = self.check_boundary_condition(mean_filter_boundary_condition,
                                                                           "mean_filter_boundary_condition")

        else:
            mean_filter_kernel_size = None
            mean_filter_boundary_condition = None

        self.mean_filter_size: Union[None, List[int]] = mean_filter_kernel_size
        self.mean_filter_boundary_condition: Union[None, str] = mean_filter_boundary_condition

        # Check Gaussian kernel settings.
        if self.has_gaussian_filter():
            # Check sigma.
            gaussian_sigma = self.check_sigma(gaussian_sigma,
                                              "gaussian_sigma")

            # Check filter truncation.
            gaussian_kernel_truncate = self.check_truncation(gaussian_kernel_truncate,
                                                             "gaussian_kernel_truncate")

            # Check boundary condition
            gaussian_kernel_boundary_condition = self.check_boundary_condition(gaussian_kernel_boundary_condition,
                                                                               "gaussian_kernel_boundary_condition")

        else:
            gaussian_sigma = None
            gaussian_kernel_truncate = None
            gaussian_kernel_boundary_condition = None

        self.gaussian_sigma: Union[None, List[float]] = gaussian_sigma
        self.gaussian_sigma_truncate: Union[None, float] = gaussian_kernel_truncate
        self.gaussian_boundary_condition: Union[None, str] = gaussian_kernel_boundary_condition

        # Check laplacian-of-gaussian filter settings
        if self.has_laws_filter():
            # Check sigma.
            laplacian_of_gaussian_sigma = self.check_sigma(laplacian_of_gaussian_sigma,
                                                           "laplacian_of_gaussian_sigma")

            # Check filter truncation.
            laplacian_of_gaussian_kernel_truncate = self.check_truncation(laplacian_of_gaussian_kernel_truncate,
                                                                          "laplacian_of_gaussian_kernel_truncate")

            # Check pooling method.
            laplacian_of_gaussian_pooling_method = self.check_pooling_method(laplacian_of_gaussian_pooling_method,
                                                                             "laplacian_of_gaussian_pooling_method",
                                                                             allow_none=True)

            # Check boundary condition.
            laplacian_of_gaussian_boundary_condition = self.check_boundary_condition(
                laplacian_of_gaussian_boundary_condition, "laplacian_of_gaussian_boundary_condition")

        else:
            laplacian_of_gaussian_sigma = None
            laplacian_of_gaussian_kernel_truncate = None
            laplacian_of_gaussian_pooling_method = None
            laplacian_of_gaussian_boundary_condition = None

        self.log_sigma: Union[None, List[float]] = laplacian_of_gaussian_sigma
        self.log_sigma_truncate: Union[None, float] = laplacian_of_gaussian_kernel_truncate
        self.log_pooling_method: Union[None, str] = laplacian_of_gaussian_pooling_method
        self.log_boundary_condition: Union[None, str] = laplacian_of_gaussian_boundary_condition

        # Check Laws kernel filter settings
        if self.has_laws_filter():
            # Check kernel.
            laws_kernel = self.check_laws_kernels(laws_kernel,
                                                  "laws_kernel")

            # Check energy computation.
            if not isinstance(laws_compute_energy, bool):
                raise TypeError("The laws_compute_energy parameter is expected to be a boolean value.")

            if laws_compute_energy:

                # Check delta.
                if not isinstance(laws_delta, list):
                    laws_delta = [laws_delta]

                if not all(isinstance(delta, int) for delta in laws_delta):
                    raise TypeError("The laws_delta parameter is expected to be one or more integers with value 0 or "
                                    "greater. Found: one or more values that are not integer.")

                if not all(delta >= 0 for delta in laws_delta):
                    raise ValueError("The laws_delta parameter is expected to be one or more integers with value 0 or "
                                     "greater. Found: one or more values that are less than 0.")

            else:
                laws_delta = None

            # Check invariance.
            if not isinstance(laws_rotation_invariance, bool):
                raise TypeError("The laws_rotation_invariance parameter is expected to be a boolean value.")

            # Check pooling method.
            laws_pooling_method = self.check_pooling_method(laws_pooling_method,
                                                            "laws_pooling_method")

            # Check boundary condition
            laws_boundary_condition = self.check_boundary_condition(
                laws_boundary_condition, "laws_boundary_condition")

        else:
            laws_kernel = None
            laws_compute_energy = None,
            laws_delta = None
            laws_rotation_invariance = None
            laws_pooling_method = None
            laws_boundary_condition = None

        self.laws_calculate_energy: Union[None, bool] = laws_compute_energy
        self.laws_kernel: Union[None, List[str]] = laws_kernel
        self.laws_delta: Union[None, bool] = laws_delta
        self.laws_rotation_invariance: Union[None, bool] = laws_rotation_invariance
        self.laws_pooling_method: Union[None, str] = laws_pooling_method
        self.laws_boundary_condition: Union[None, str] = laws_boundary_condition

        # Check Gabor filter settings.
        if self.has_gabor_filter():
            # Check sigma.
            gabor_sigma = self.check_sigma(gabor_sigma, "gabor_sigma")

            # Check kernel truncation width.
            gabor_kernel_truncate = self.check_truncation(gabor_kernel_truncate,
                                                          "gabor_kernel_truncate")

            # Check gamma. Gamma behaves like sigma.
            gabor_gamma = self.check_sigma(gabor_gamma,
                                           "gabor_gamma")

            # Check lambda. Lambda behaves like sigma
            gabor_lambda = self.check_sigma(gabor_lambda,
                                            "gabor_lambda")

            # Check theta step.
            if gabor_theta_step is not None:
                if not isinstance(gabor_theta_step, (float, int)):
                    raise TypeError("The gabor_theta_step parameter is expected to be an angle, in degrees. Found a "
                                    "value that was not a number.")

                if gabor_theta_step == 0.0:
                    gabor_theta_step = None

            if gabor_theta_step is not None:
                # Check that the step would divide the 360 degree circle into a integer number of steps.
                if not (360.0 / gabor_theta_step).is_integer():
                    raise ValueError(f"The gabor_theta_step parameter should divide a circle into equal portions. "
                                     f"The current settings would create {360.0 / gabor_theta_step} portions.")

            # Check theta.
            gabor_pool_theta = gabor_theta_step is not None

            if not isinstance(gabor_theta, list):
                gabor_theta = [gabor_theta]

            if gabor_theta_step is not None and len(gabor_theta) > 1:
                raise ValueError(f"The gabor_theta parameter cannot have more than one value when used in conjunction"
                                 f" with the gabor_theta_step parameter")

            if not all(isinstance(theta, (float, int)) for theta in gabor_theta):
                raise TypeError(f"The gabor_theta parameter is expected to be one or more values indicating angles in"
                                f" degrees. Found: one or more values that were not numeric.")

            if gabor_theta_step is not None:
                gabor_theta = [gabor_theta[0] + ii * gabor_theta_step for ii in np.arange(0.0, 360.0, gabor_theta_step)]

            # Check filter response.
            gabor_response = self.check_response(gabor_response,
                                                 "gabor_response")

            # Check rotation invariance
            if not isinstance(gabor_rotation_invariance, bool):
                raise TypeError("The gabor_rotation_invariance parameter is expected to be a boolean value.")

            # Check pooling method
            gabor_pooling_method = self.check_pooling_method(gabor_pooling_method,
                                                             "gabor_pooling_method")

            # Check boundary condition
            gabor_boundary_condition = self.check_boundary_condition(
                gabor_boundary_condition, "gabor_boundary_condition")

        else:
            gabor_sigma = None
            gabor_kernel_truncate = None
            gabor_gamma = None
            gabor_lambda = None
            gabor_theta = None
            gabor_pool_theta = None
            gabor_response = None
            gabor_rotation_invariance = None
            gabor_pooling_method = None
            gabor_boundary_condition = None

        self.gabor_sigma: Union[None, List[float]] = gabor_sigma
        self.gabor_sigma_truncate: Union[None, float] = gabor_kernel_truncate
        self.gabor_gamma: Union[None, List[float]] = gabor_gamma
        self.gabor_lambda: Union[None, List[float]] = gabor_lambda
        self.gabor_theta: Union[None, List[float], List[int]] = gabor_theta
        self.gabor_pool_theta: Union[None, bool] = gabor_pool_theta
        self.gabor_response: Union[None, str] = gabor_response
        self.gabor_rotation_invariance: Union[None, str] = gabor_rotation_invariance
        self.gabor_pooling_method: Union[None, str] = gabor_pooling_method
        self.gabor_boundary_condition: Union[None, str] = gabor_boundary_condition

        # Check separable wavelet settings.
        if self.has_separable_wavelet_filter():
            # Check wavelet families.
            separable_wavelet_families = self.check_separable_wavelet_families(separable_wavelet_families,
                                                                               "separable_wavelet_families")

            # Check wavelet filter sets.
            separable_wavelet_set = self.check_separable_wavelet_sets(separable_wavelet_set,
                                                                      "separable_wavelet_set")

            # Check if wavelet is stationary
            if not isinstance(separable_wavelet_stationary, bool):
                raise TypeError(f"The separable_wavelet_stationary parameter is expected to be a boolean value.")

            # Check decomposition level
            separable_wavelet_decomposition_level = self.check_decomposition_level(
                separable_wavelet_decomposition_level, "separable_wavelet_decomposition_level")

            # Check rotation invariance
            if not isinstance(separable_wavelet_rotation_invariance, bool):
                raise TypeError("The separable_wavelet_rotation_invariance parameter is expected to be a boolean value.")

            # Check pooling method.
            separable_wavelet_pooling_method = self.check_pooling_method(separable_wavelet_pooling_method,
                                                                         "separable_wavelet_pooling_method")

            # Check boundary condition.
            separable_wavelet_boundary_condition = self.check_boundary_condition(
                separable_wavelet_boundary_condition, "separable_wavelet_boundary_condition")

        else:
            separable_wavelet_families = None
            separable_wavelet_set = None
            separable_wavelet_stationary = None
            separable_wavelet_decomposition_level = None
            separable_wavelet_rotation_invariance = None
            separable_wavelet_pooling_method = None
            separable_wavelet_boundary_condition = None

        self.separable_wavelet_families: Union[None, List[str]] = separable_wavelet_families
        self.separable_wavelet_filter_set: Union[None, List[str]] = separable_wavelet_set
        self.separable_wavelet_stationary: Union[None, bool] = separable_wavelet_stationary
        self.separable_wavelet_decomposition_level: Union[None, List[int]] = separable_wavelet_decomposition_level
        self.separable_wavelet_rotation_invariance: Union[None, bool] = separable_wavelet_rotation_invariance
        self.separable_wavelet_pooling_method: Union[None, str] = separable_wavelet_pooling_method
        self.separable_wavelet_boundary_condition: Union[None, str] = separable_wavelet_boundary_condition

        # Set parameters for non-separable wavelets.
        if self.has_nonseparable_wavelet_filter():
            # Check wavelet families.
            nonseparable_wavelet_families = self.check_nonseparable_wavelet_families(
                nonseparable_wavelet_families, "nonseparable_wavelet_families")

            # Check decomposition level.
            nonseparable_wavelet_decomposition_level = self.check_decomposition_level(
                nonseparable_wavelet_decomposition_level, "nonseparable_wavelet_decomposition_level")

            # Check filter response.
            nonseparable_wavelet_response = self.check_response(nonseparable_wavelet_response,
                                                                "nonseparable_wavelet_response")

            # Check boundary condition.
            nonseparable_wavelet_boundary_condition = self.check_boundary_condition(
                nonseparable_wavelet_boundary_condition, "nonseparable_wavelet_boundary_condition")

        else:
            nonseparable_wavelet_families = None
            nonseparable_wavelet_decomposition_level = None
            nonseparable_wavelet_response = None
            nonseparable_wavelet_boundary_condition = None

        self.nonseparable_wavelet_families: Union[None, List[str]] = nonseparable_wavelet_families
        self.nonseparable_wavelet_decomposition_level: Union[None, List[int]] = nonseparable_wavelet_decomposition_level
        self.nonseparable_wavelet_response: Union[None, str] = nonseparable_wavelet_response
        self.nonseparable_wavelet_boundary_condition: Union[None, str] = nonseparable_wavelet_boundary_condition

        # Check Riesz filter orders.
        if self.has_riesz_filter():
            riesz_filter_order = self.check_riesz_filter_order(riesz_filter_order,
                                                               "riesz_filter_order")

        else:
            riesz_filter_order = None

        if self.has_steered_riesz_filter():
            riesz_filter_tensor_sigma = self.check_sigma(riesz_filter_tensor_sigma,
                                                         "riesz_filter_tensor_sigma")

        else:
            riesz_filter_tensor_sigma = None

        self.riesz_order: Union[None, List[List[int]]] = riesz_filter_order
        self.riesz_filter_tensor_sigma: Union[None, List[float]] = riesz_filter_tensor_sigma

    def check_boundary_condition(self, x, var_name):
        if x is None:
            if self.boundary_condition is not None:
                # Avoid updating by reference.
                x = copy.deepcopy(self.boundary_condition)

            else:
                raise ValueError(f"No value for the {var_name} parameter could be set, due to a lack of a"
                                 f"default.")

        # Check value
        if x not in ["reflect", "constant", "nearest", "mirror", "wrap"]:
            raise ValueError(f"The provided value for the {var_name} is not valid. One of 'reflect', 'constant', "
                             f"'nearest', 'mirror' or 'wrap' was expected. Found: {x}")

        return x

    @staticmethod
    def check_pooling_method(x, var_name, allow_none=False):

        valid_pooling_method = ["max", "min", "mean", "sum"]
        if allow_none:
            valid_pooling_method += ["none"]

        if x not in valid_pooling_method:
            raise ValueError(f"The {var_name} parameter expects one of the following values: "
                             f"{', '.join(valid_pooling_method)}. Found: {x}")

        return x

    @staticmethod
    def check_sigma(x, var_name):
        # Check sigma is a list.
        if not isinstance(x, list):
            x = [x]

        # Check that the sigma values are floating points.
        if not all(isinstance(sigma, float) for sigma in x):
            raise TypeError(f"The {var_name} parameter is expected to consists of floating points with values "
                            f"greater than 0.0. Found: one or more values that were not floating points.")

        if not all(sigma > 0.0 for sigma in x):
            raise ValueError(f"The {var_name} parameter is expected to consists of floating points with values "
                             f"greater than 0.0. Found: one or more values with value 0.0 or less.")

        return x

    @staticmethod
    def check_truncation(x, var_name):

        # Check that the truncation values are floating points.
        if not isinstance(x, float):
            raise TypeError(f"The {var_name} parameter is expected to be a floating point with value "
                            f"greater than 0.0. Found: a value that was not a floating point.")

        if not x > 0.0:
            raise ValueError(f"The {var_name} parameter is expected to be a floating point with value "
                             f"greater than 0.0. Found: a value of 0.0 or less.")

        return x

    @staticmethod
    def check_response(x, var_name):

        valid_response = ["modulus", "abs", "magnitude", "angle", "phase", "argument", "real", "imaginary"]

        # Check that response is correct.
        if x not in valid_response:
            raise ValueError(f"The {var_name} parameter is not correct. Expected one of {', '.join(valid_response)}. "
                             f"Found: {x}")

        return x

    @staticmethod
    def check_separable_wavelet_families(x, var_name):
        # Import pywavelets.
        import pywt

        # Check if list.
        if not isinstance(x, list):
            x = [x]

        available_kernels = pywt.wavelist(kind="discrete")
        valid_kernel = [kernel.lower() in available_kernels for kernel in x]

        if not all(valid_kernel):
            raise ValueError(f"The {var_name} parameter requires wavelet families that match those defined in the "
                             f"pywavelets package. Could not match: "
                             f"{', '.join([kernel for ii, kernel in x if not valid_kernel[ii]])}")

        # Return lowercase values.
        return [xx.lower for xx in x]

    @staticmethod
    def check_nonseparable_wavelet_families(x, var_name):
        # Check if list.
        if not isinstance(x, list):
            x = [x]

        available_kernels = ["simoncelli", "shannon"]
        valid_kernel = [kernel.lower() in available_kernels for kernel in x]

        if not all(valid_kernel):
            raise ValueError(f"The {var_name} parameter expects one or more of the following values: "
                             f"{', '.join(available_kernels)}. Could not match: "
                             f"{', '.join([kernel for ii, kernel in x if not valid_kernel[ii]])}")

        # Return lowercase values.
        return [xx.lower for xx in x]

    @staticmethod
    def check_decomposition_level(x, var_name):
        # Check if list.
        if not isinstance(x, list):
            x = [x]

        if not all(isinstance(xx, int) for xx in x):
            raise TypeError(f"The {var_name} parameter should be one or more integer "
                            f"values of at least 1. Found: one or more values that was not an integer.")

        if not all(xx >= 1 for xx in x):
            raise ValueError(f"The {var_name} parameter should be one or more integer "
                             f"values of at least 1. Found: one or more values that was not an integer.")

        return x

    def check_separable_wavelet_sets(self, x: Union[None, str, List[str]], var_name):
        from itertools import product

        if x is None:
            if self.by_slice:
                x = "hh"
            else:
                x = "hhh"

        # Check if x is a list.
        if not isinstance(x, list):
            x = [x]

        # Generate all potential combinations.
        if self.by_slice:
            possible_combinations = ["".join(combination) for combination in product(["l", "h"], repeat=2)]

        else:
            possible_combinations = ["".join(combination) for combination in product(["l", "h"], repeat=3)]

        # Check for all.
        if any(kernel == "all" for kernel in x):
            x = possible_combinations

        # Check which kernels are valid.
        valid_kernel = [kernel.lower() in possible_combinations for kernel in x]

        if not all(valid_kernel):
            raise ValueError(f"The {var_name} parameter requires combinations of low (l) and high-pass (h) kernels. "
                             f"Two kernels should be specified for 2D, and three for 3D. Found the following invalid "
                             f"combinations: "
                             f"{', '.join([kernel for ii, kernel in enumerate(x) if not valid_kernel[ii]])}")

        # Return lowercase values.
        return [xx.lower for xx in x]

    def check_laws_kernels(self, x: Union[str, List[str]], var_name):
        from itertools import product

        # Set implemented kernels.
        kernels = ['l5', 'e5', 's5', 'w5', 'r5', 'l3', 'e3', 's3']

        # Generate all valid combinations.
        if self.by_slice:
            possible_combinations = ["".join(combination) for combination in product(kernels, repeat=2)]

        else:
            possible_combinations = ["".join(combination) for combination in product(kernels, repeat=3)]

        # Create list.
        if not isinstance(x, list):
            x = [x]

        # Check which kernels are valid.
        valid_kernel = [kernel.lower() in possible_combinations for kernel in x]

        if not all(valid_kernel):
            raise ValueError(f"The {var_name} parameter requires combinations of Laws kernels. The follow kernels are "
                             f"implemented: {', '.join(kernels)}. Two kernels should be specified for 2D, "
                             f"and three for 3D. Found the following illegal combinations: "
                             f"{', '.join([kernel for ii, kernel in enumerate(x) if not valid_kernel[ii]])}")

        # Return lowercase values.
        return [xx.lower for xx in x]

    def check_riesz_filter_order(self, x, var_name):
        from itertools import product

        # Skip if None
        if x is None:
            return x

        # Set number of elements that the filter order should have
        if self.by_slice:
            n_elements = 2

        else:
            n_elements = 3

        # Create filterbank.
        if isinstance(x, int):
            # Check that x is not negative.
            if x < 0:
                raise ValueError(f"The {var_name} parameter cannot be negative.")

            # Set filter order.
            single_filter_order = list(np.arange(0, x+1))

            # Generate all valid combinations.
            x = [list(combination) for combination in product(single_filter_order, repeat=n_elements) if
                 sum(combination) == x]

        if not isinstance(x, list):
            raise TypeError(f"The {var_name} parameter is expected to be a list")

        # Create a nested list,
        if not all(isinstance(xx, list) for xx in x):
            x = [x]

        # Check that all elements of x have the right length, and do not negative orders.
        if not all(len(xx) == n_elements for xx in x):
            raise ValueError(f"The {var_name} parameter is expected to contain filter orders, each consisting of "
                             f"{n_elements} non-negative integer values. One or more filter orders did not have the "
                             f"expected number of elements.")

        if not all(all(isinstance(xxx, int) for xxx in xx) for xx in x):
            raise ValueError(f"The {var_name} parameter is expected to contain filter orders, each consisting of "
                             f"{n_elements} non-negative integer values. One or more filter orders did not fully "
                             f"consist of integer values.")

        if not all(all(xxx >= 0 for xxx in xx) for xx in x):
            raise ValueError(f"The {var_name} parameter is expected to contain filter orders, each consisting of "
                             f"{n_elements} non-negative integer values. One or more filter orders contained negative values.")

        return x

    def has_mean_filter(self, x=None):
        if x is None:
            x = self.spatial_filters
        elif not isinstance(x, list):
            x = [x]

        return x is not None and any(filter_kernel in ["mean"] for filter_kernel in x)

    def has_gaussian_filter(self, x=None):
        if x is None:
            x = self.spatial_filters
        elif not isinstance(x, list):
            x = [x]

        return x is not None and any(filter_kernel in ["gaussian", "riesz_gaussian", "riesz_steered_gaussian"] for
                                     filter_kernel in x)

    def has_laplacian_of_gaussian_filter(self, x=None):
        if x is None:
            x = self.spatial_filters
        elif not isinstance(x, list):
            x = [x]

        return x is not None and any(filter_kernel in ["laplacian_of_gaussian", "log", "riesz_laplacian_of_gaussian", "riesz_log",
                                     "riesz_steered_laplacian_of_gaussian", "riesz_steered_log"] for filter_kernel in x)

    def has_laws_filter(self, x=None):
        if x is None:
            x = self.spatial_filters
        elif not isinstance(x, list):
            x = [x]

        return x is not None and any(filter_kernel in ["laws"] for filter_kernel in x)

    def has_gabor_filter(self, x=None):
        if x is None:
            x = self.spatial_filters
        elif not isinstance(x, list):
            x = [x]

        return x is not None and any(filter_kernel in ["gabor", "riesz_gabor", "riesz_steered_gabor"] for
                                     filter_kernel in x)

    def has_separable_wavelet_filter(self, x=None):
        if x is None:
            x = self.spatial_filters
        elif not isinstance(x, list):
            x = [x]

        return x is not None and any(filter_kernel in ["separable_wavelet"] for filter_kernel in x)

    def has_nonseparable_wavelet_filter(self, x=None):
        if x is None:
            x = self.spatial_filters
        elif not isinstance(x, list):
            x = [x]

        return x is not None and any(filter_kernel in ["nonseparable_wavelet", "riesz_nonseparable_wavelet",
                                     "riesz_steered_nonseparable_wavelet"] for filter_kernel in x)

    def has_riesz_filter(self, x=None):
        if x is None:
            x = self.spatial_filters
        elif not isinstance(x, list):
            x = [x]

        return x is not None and any(filter_kernel.startswith("riesz") for filter_kernel in x)

    def has_steered_riesz_filter(self, x=None):
        if x is None:
            x = self.spatial_filters
        elif not isinstance(x, list):
            x = [x]

        return x is not None and any(filter_kernel.startswith("riesz_steered") for filter_kernel in x)


class SettingsClass:

    def __init__(self,
                 general_settings: GeneralSettingsClass,
                 img_interpolate_settings: ImageInterpolationSettingsClass,
                 roi_interpolate_settings: RoiInterpolationSettingsClass,
                 post_process_settings: ImagePostProcessingClass,
                 perturbation_settings: ImagePerturbationSettingsClass,
                 roi_resegment_settings: ResegmentationSettingsClass,
                 feature_extr_settings: FeatureExtractionSettingsClass,
                 img_transform_settings: ImageTransformationSettingsClass):

        self.general = general_settings
        self.img_interpolate = img_interpolate_settings
        self.roi_interpolate = roi_interpolate_settings
        self.post_process = post_process_settings
        self.perturbation = perturbation_settings
        self.roi_resegment = roi_resegment_settings
        self.feature_extr = feature_extr_settings
        self.img_transform = img_transform_settings


def str2list(strx, data_type, default=None):
    """ Function for splitting strings read from the xml file """

    # Check if strx is none
    if strx is None and default is None:
        return None
    elif strx is None and type(default) in [list, tuple]:
        return default
    elif strx is None and not type(default) in [list, tuple]:
        return [default]

    # If strx is an element, read string
    if type(strx) is ElemTree.Element:
        strx = strx.text

    # Repeat check
    if strx is None and default is None:
        return None
    elif strx is None and type(default) in [list, tuple]:
        return default
    elif strx is None and not type(default) in [list, tuple]:
        return [default]

    contents = strx.split(",")
    content_list = []

    if (len(contents) == 1) and (contents[0] == ""):
        return content_list

    for i in np.arange(0, len(contents)):
        append_data = str2type(contents[i], data_type)

        # Check append data for None
        if append_data is None and type(default) in [list, tuple]:
            return default
        elif append_data is None and not type(default) in [list, tuple]:
            return [default]
        else:
            content_list.append(append_data)

    return content_list


def str2type(strx, data_type, default=None):
    # Check if strx is none
    if strx is None and default is None:
        return None
    elif strx is None:
        return default

    # If strx is an element, read string
    if type(strx) is ElemTree.Element:
        strx = strx.text

    # Test if the requested data type is not a string or path, but is empty
    if data_type not in ["str", "path"] and (strx == "" or strx is None):
        return default
    elif data_type in ["str", "path"] and (strx == "" or strx is None) and default is not None:
        return default

    # Casting of strings to different data types
    if data_type == "int":
        return int(strx)
    if data_type == "bool":
        if strx in ("true", "True", "TRUE", "T", "t", "1"):
            return True
        elif strx in ("false", "False", "FALSE", "F", "f", "0"):
            return False
    if data_type == "float":
        if strx in ("na", "nan", "NA", "NaN"):
            return np.nan
        elif strx in ("-inf", "-Inf", "-INF"):
            return -np.inf
        elif strx in ("inf", "Inf", "INF"):
            return np.inf
        else:
            return float(strx)
    if data_type == "str":
        return strx
    if data_type == "path":
        return strx


def import_configuration_settings(compute_features: bool,
                                  path: Union[None, str] = None,
                                  **kwargs):
    import os.path

    # Make a copy of the kwargs argument to avoid updating by reference.
    kwargs = copy.deepcopy(kwargs)

    # Check if a configuration file is used.
    if path is None:

        # Prevent checking of feature parameters if features are not computed.
        if not compute_features:
            kwargs.update({"base_feature_families": "none",
                           "response_map_feature_families": "none"})

        # Set general settings.
        general_settings = GeneralSettingsClass(**kwargs)

        # Remove by_slice from the keyword arguments to avoid double passing.
        kwargs.pop("by_slice", None)
        kwargs.pop("no_approximation", None)

        # Set image interpolation settings
        image_interpolation_settings = ImageInterpolationSettingsClass(by_slice=general_settings.by_slice,
                                                                       **kwargs)
        # Set ROI interpolation settings
        roi_interpolation_settings = RoiInterpolationSettingsClass(**kwargs)

        # Set post-processing settings
        post_processing_settings = ImagePostProcessingClass(**kwargs)

        # Set perturbation settings
        perturbation_settings = ImagePerturbationSettingsClass(**kwargs)

        # Set resegmentation settings
        resegmentation_settings = ResegmentationSettingsClass(**kwargs)

        # Set feature extraction settings
        feature_extraction_settings = FeatureExtractionSettingsClass(by_slice=general_settings.by_slice,
                                                                     no_approximation=general_settings.no_approximation,
                                                                     **kwargs)

        # Set image transformation settings.
        image_transformation_settings = ImageTransformationSettingsClass(
            by_slice=general_settings.by_slice,
            response_map_feature_settings=feature_extraction_settings,
            **kwargs)

        return [SettingsClass(general_settings=general_settings,
                              img_interpolate_settings=image_interpolation_settings,
                              roi_interpolate_settings=roi_interpolation_settings,
                              post_process_settings=post_processing_settings,
                              perturbation_settings=perturbation_settings,
                              roi_resegment_settings=resegmentation_settings,
                              feature_extr_settings=feature_extraction_settings,
                              img_transform_settings=image_transformation_settings)]

    elif not os.path.exists(path):
        raise FileNotFoundError(f"The settings file could not be found at {path}.")

    # Load xml
    tree = ElemTree.parse(path)
    root = tree.getroot()

    # Empty list for settings
    settings_list = []

    # Set default values for feature families.
    base_feature_families = "all"
    response_map_feature_families = "statistical"

    # Prevent checking of feature parameters if features are not computed.
    if not compute_features:
        base_feature_families = None,
        response_map_feature_families = None

    for branch in root.findall("config"):

        # Process general settings
        general_branch = branch.find("general")

        if general_branch is not None:
            general_settings = GeneralSettingsClass(
                by_slice=str2type(general_branch.find("by_slice"), "str", False),
                config_str=str2type(general_branch.find("config_str"), "str", ""),
                no_approximation=str2type(general_branch.find("no_approximation"), "bool", False))

        else:
            general_settings = GeneralSettingsClass()

        # Process image interpolation settings
        img_interp_branch = branch.find("img_interpolate")

        if img_interp_branch is not None:
            img_interp_settings = ImageInterpolationSettingsClass(
                by_slice=general_settings.by_slice,
                interpolate=str2type(img_interp_branch.find("interpolate"), "bool", False),
                spline_order=str2type(img_interp_branch.find("spline_order"), "int", 3),
                new_spacing=str2list(img_interp_branch.find("new_spacing"), "float", None),
                anti_aliasing=str2type(img_interp_branch.find("anti_aliasing"), "bool", True),
                smoothing_beta=str2type(img_interp_branch.find("smoothing_beta"), "float", 0.98))

        else:
            img_interp_settings = ImageInterpolationSettingsClass(by_slice=general_settings.by_slice)

        # Process roi interpolation settings
        roi_interp_branch = branch.find("roi_interpolate")

        if roi_interp_branch is not None:
            roi_interp_settings = RoiInterpolationSettingsClass(
                roi_spline_order=str2type(roi_interp_branch.find("spline_order"), "int", 1),
                roi_interpolation_mask_inclusion_threshold=str2type(roi_interp_branch.find("incl_threshold"), "float", 0.5))

        else:
            roi_interp_settings = RoiInterpolationSettingsClass()

        # Image post-acquisition processing settings
        post_process_branch = branch.find("post_processing")

        if post_process_branch is not None:
            post_process_settings = ImagePostProcessingClass(
                bias_field_correction=str2type(post_process_branch.find("bias_field_correction"), "bool", False),
                bias_field_correction_n_fitting_levels=str2type(post_process_branch.find("n_fitting_levels"), "int", 3),
                bias_field_correction_n_max_iterations=str2list(post_process_branch.find("n_max_iterations"), "int", 100),
                bias_field_convergence_threshold=str2type(post_process_branch.find("convergence_threshold"), "float", 0.001),
                intensity_normalisation=str2type(post_process_branch.find("intensity_normalisation"), "str", "none"),
                intensity_normalisation_range=str2list(post_process_branch.find("intensity_normalisation_range"), "float", None),
                intensity_normalisation_saturation=str2list(post_process_branch.find("intensity_normalisation_saturation"), "float", None),
                tissue_mask_type=str2type(post_process_branch.find("tissue_mask_type"), "str", "relative_range"),
                tissue_mask_range=str2list(post_process_branch.find("tissue_mask_range"), "float", None))

        else:
            post_process_settings = ImagePostProcessingClass()

        # Image and roi volume adaptation settings
        perturbation_branch = branch.find("vol_adapt")

        if perturbation_branch is not None:
            perturbation_settings = ImagePerturbationSettingsClass(
                crop_around_roi=str2type(read_node(
                    perturbation_branch, ["crop_around_roi", "resect"]), "bool", False),
                crop_distance=str2type(perturbation_branch.find("crop_distance"), "float", 150.0),
                perturbation_noise_repetitions=str2type(perturbation_branch.find("noise_repetitions"), "int", 0),
                perturbation_noise_level=str2type(perturbation_branch.find("noise_level"), "float"),
                perturbation_rotation_angles=str2list(read_node(
                    perturbation_branch, ["rotation_angles", "rot_angles"]), "float", 0.0),
                perturbation_translation_fraction=str2list(read_node(
                    perturbation_branch, ["translation_fraction", "translate_frac"]), "float", 0.0),
                perturbation_roi_adapt_type=str2type(perturbation_branch.find("roi_adapt_type"), "str", "distance"),
                perturbation_roi_adapt_size=str2list(perturbation_branch.find("roi_adapt_size"), "float", 0.0),
                perturbation_roi_adapt_max_erosion=str2type(read_node(
                    perturbation_branch, ["roi_adapt_max_erosion", "eroded_vol_fract"]), "float", 0.8),
                perturbation_randomise_roi_repetitions=str2type(perturbation_branch.find("roi_randomise_repetitions"), "int", 0),
                roi_split_boundary_size=str2list(perturbation_branch.find("roi_boundary_size"), "float", 0.0),
                roi_split_max_erosion=str2type(read_node(
                    perturbation_branch, ["roi_split_max_erosion", "bulk_min_vol_fract"]), "float", 0.6))

        else:
            perturbation_settings = ImagePerturbationSettingsClass()

        # Process roi segmentation settings
        roi_resegment_branch = branch.find("roi_resegment")

        if roi_resegment_branch is not None:
            roi_resegment_settings = ResegmentationSettingsClass(
                resegmentation_method=str2list(roi_resegment_branch.find("method"), "str", "none"),
                resegmentation_intensity_range=str2list(read_node(
                    roi_resegment_branch, ["intensity_range", "g_thresh"]), "float", None),
                resegmentation_sigma=str2type(roi_resegment_branch.find("sigma"), "float", 3.0))

        else:
            roi_resegment_settings = ResegmentationSettingsClass()

        # Process feature extraction settings
        feature_extr_branch = branch.find("feature_extr")

        if feature_extr_branch is not None:
            if feature_extr_branch.find("glcm_merge_method") is not None:
                raise DeprecationWarning(
                    "The glcm_merge_method tag has been deprecated. Use the glcm_spatial_method tag instead. This takes"
                    " the following values: `2d_average`, `2d_slice_merge`, '2.5d_direction_merge', '2.5d_volume_merge',"
                    " '3d_average', and `3d_volume_merge`")

            if feature_extr_branch.find("glrlm_merge_method") is not None:
                raise DeprecationWarning(
                    "The glrlm_merge_method tag has been deprecated. Use the glrlm_spatial_method tag instead. This "
                    "takes the following values: `2d_average`, `2d_slice_merge`, '2.5d_direction_merge', "
                    "'2.5d_volume_merge', '3d_average', and `3d_volume_merge`")

            feature_extr_settings = FeatureExtractionSettingsClass(
                by_slice=general_settings.by_slice,
                no_approximation=general_settings.no_approximation,
                base_feature_families=str2list(read_node(
                    feature_extr_branch, ["feature_families", "families"]), "str", base_feature_families),
                base_discretisation_method=str2list(read_node(
                    feature_extr_branch, ["discretisation_method", "discr_method"]), "str", None),
                base_discretisation_n_bins=str2list(read_node(
                    feature_extr_branch, ["discretisation_n_bins", "discr_n_bins"]), "int", None),
                base_discretisation_bin_width=str2list(read_node(
                    feature_extr_branch, ["discretisation_bin_width", "discr_bin_width"]), "float", None),
                ivh_discretisation_method=str2type(read_node(
                    feature_extr_branch, ["ivh_discretisation_method", "ivh_discr_method"]), "str", "none"),
                ivh_discretisation_n_bins=str2type(read_node(
                    feature_extr_branch, ["ivh_discretisation_n_bins", "ivh_discr_n_bins"]), "int", 1000),
                ivh_discretisation_bin_width=str2type(read_node(
                    feature_extr_branch, ["ivh_discretisation_bin_width", "ivh_discr_bin_width"]), "float", None),
                glcm_distance=str2list(read_node(
                    feature_extr_branch, ["glcm_distance", "glcm_dist"]), "float", 1.0),
                glcm_spatial_method=str2list(feature_extr_branch.find("glcm_spatial_method"), "str", None),
                glrlm_spatial_method=str2list(feature_extr_branch.find("glrlm_spatial_method"), "str", None),
                glszm_spatial_method=str2list(feature_extr_branch.find("glszm_spatial_method"), "str", None),
                gldzm_spatial_method=str2list(feature_extr_branch.find("gldzm_spatial_method"), "str", None),
                ngtdm_spatial_method=str2list(feature_extr_branch.find("ngtdm_spatial_method"), "str", None),
                ngldm_distance=str2list(read_node(
                    feature_extr_branch, ["ngldm_distance", "ngldm_dist"]), "float", 1.0),
                ngldm_difference_level=str2list(read_node(
                    feature_extr_branch, ["ngldm_difference_level", "ngldm_diff_lvl"]), "float", 0.0),
                ngldm_spatial_method=str2list(feature_extr_branch.find("ngldm_spatial_method"), "str", None)
            )

        else:
            # If the section is absent, assume that no features are extracted.
            feature_extr_settings = FeatureExtractionSettingsClass(by_slice=general_settings.by_slice,
                                                                   no_approximation=general_settings.no_approximation,
                                                                   base_feature_families=None)

        # Process image transformation settings
        img_transform_branch = branch.find("img_transform")

        if img_transform_branch is not None:
            if img_transform_branch.find("log_average") is not None:
                raise DeprecationWarning(
                    "The log_average tag has been deprecated. Use the laplacian_of_gaussian_pooling_method tag "
                    "instead with the value `mean` to emulate log_average=True.")

            if img_transform_branch.find("riesz_steered") is not None:
                raise DeprecationWarning(
                    "The riesz_steered tag has been deprecated. Steerable Riesz filter are now identified by the name "
                    "of the filter kernel (filter_kernels parameter).")

            img_transform_settings = ImageTransformationSettingsClass(
                by_slice=general_settings.by_slice,
                response_map_feature_settings=feature_extr_settings,
                response_map_feature_families=str2list(
                    img_transform_branch.find("feature_families"), "str", response_map_feature_families),
                response_map_discretisation_method=str2list(
                    img_transform_branch.find("discretisation_method"), "str", None),
                response_map_discretisation_bin_width=str2list(
                    img_transform_branch.find("discretisation_bin_width"), "float", None),
                response_map_discretisation_n_bins=str2list(
                    img_transform_branch.find("discretisation_n_bins"), "int", None),
                filter_kernels=str2list(read_node(
                    img_transform_branch, ["filter_kernels", "spatial_filters"]), "str", None),
                boundary_condition=str2type(img_transform_branch.find("boundary_condition"), "str", "mirror"),
                separable_wavelet_families=str2list(read_node(
                    img_transform_branch, "separable_wavelet_families", "wavelet_fam"), "str", None),
                separable_wavelet_set=str2list(read_node(
                    img_transform_branch, "separable_wavelet_set", "wavelet_filter_set"), "str", "all"),
                separable_wavelet_stationary=True,
                separable_wavelet_decomposition_level=str2list(read_node(
                    img_transform_branch, "separable_wavelet_decomposition_level", "wavelet_decomposition_level"),
                    "int", 1),
                separable_wavelet_rotation_invariance=str2type(read_node(
                    img_transform_branch, "separable_wavelet_rotation_invariance", "wavelet_rot_invar"), "bool", True),
                separable_wavelet_pooling_method=str2type(read_node(
                    img_transform_branch, "separable_wavelet_pooling_method", "wavelet_pooling_method"), "str", "max"),
                separable_wavelet_boundary_condition=str2type(
                    img_transform_branch.find("separable_wavelet_boundary_condition"), "str", None),
                nonseparable_wavelet_families=str2list(read_node(
                    img_transform_branch, "nonseparable_wavelet_families"), "str", None),
                nonseparable_wavelet_decomposition_level=str2list(read_node(
                    img_transform_branch, "nonseparable_wavelet_decomposition_level", "wavelet_decomposition_level"),
                    "int", 1),
                nonseparable_wavelet_response=str2type(
                    img_transform_branch.find("nonseparable_wavelet_response"), "str", "real"),
                nonseparable_wavelet_boundary_condition=str2type(
                    img_transform_branch.find("nonseparable_wavelet_boundary_condition"), "str", None),
                gaussian_sigma=str2list(read_node(
                    img_transform_branch, ["gaussian_sigma", "gauss_sigma"]), "float", None),
                gaussian_kernel_truncate=str2type(read_node(
                    img_transform_branch, ["gaussian_kernel_truncate", "gaussian_sigma_truncate"]), "float", 4.0),
                gaussian_kernel_boundary_condition=str2type(
                    img_transform_branch.find("gaussian_kernel_boundary_condition"), "str", None),
                laplacian_of_gaussian_sigma=str2list(read_node(
                    img_transform_branch, ["laplacian_of_gaussian_sigma", "log_sigma"]), "float", None),
                laplacian_of_gaussian_kernel_truncate=str2type(read_node(
                    img_transform_branch, ["laplacian_of_gaussian_kernel_truncate", "log_sigma_truncate"]),
                    "float", 4.0),
                laplacian_of_gaussian_pooling_method=str2type(
                    img_transform_branch.find("laplacian_of_gaussian_pooling_method"), "str", "none"),
                laplacian_of_gaussian_boundary_condition=str2type(
                    img_transform_branch.find("laplacian_of_gaussian_boundary_condition"), "str", None),
                laws_kernel=str2list(img_transform_branch.find("laws_kernel"), "str", None),
                laws_compute_energy=str2type(
                    img_transform_branch.find("laws_calculate_energy"), "bool", True),
                laws_delta=str2list(img_transform_branch.find("laws_delta"), "int", 7),
                laws_rotation_invariance=str2type(read_node(
                    img_transform_branch, ["laws_rotation_invariance", "laws_rot_invar"]), "bool", True),
                laws_pooling_method=str2type(img_transform_branch.find("laws_pooling_method"), "str", "max"),
                laws_boundary_condition=str2type(img_transform_branch.find('laws_boundary_condition'), "str", None),
                gabor_sigma=str2list(img_transform_branch.find("gabor_sigma"), "float", None),
                gabor_lambda=str2list(img_transform_branch.find("gabor_lambda"), "float", None),
                gabor_kernel_truncate=str2type(read_node(
                    img_transform_branch, ["gabor_kernel_truncate", "gabor_sigma_truncate"]), "float", 10.0),
                gabor_gamma=str2list(img_transform_branch.find("gabor_gamma"), "float", 1.0),
                gabor_theta=str2list(read_node(
                    img_transform_branch, "gabor_theta", "gabor_theta_initial"), "float", 0.0),
                gabor_theta_step=str2type(img_transform_branch.find("gabor_theta_step"), "float", None),
                gabor_response=str2type(img_transform_branch.find("gabor_response"), "str", "modulus"),
                gabor_rotation_invariance=str2type(read_node(
                    img_transform_branch, ["gabor_rotation_invariance", "gabor_rot_invar"]), "bool", True),
                gabor_pooling_method=str2type(img_transform_branch.find("gabor_pooling_method"), "str", "max"),
                gabor_boundary_condition=str2type(img_transform_branch.find('gabor_boundary_condition'), "str", None),
                mean_filter_kernel_size=str2list(read_node(
                    img_transform_branch, ["mean_filter_kernel_size", "mean_filter_size"]), "int", None),
                mean_filter_boundary_condition=str2type(
                    img_transform_branch.find('mean_filter_boundary_condition'), "str", None),
                riesz_filter_order=str2list(read_node(
                    img_transform_branch, ["riesz_filter_order", "riesz_order"]), "int", None),
                riesz_filter_tensor_sigma=str2list(img_transform_branch.find("riesz_filter_tensor_sigma"), "float", None)
            )

        else:
            img_transform_settings = ImageTransformationSettingsClass(by_slice=general_settings.by_slice,
                                                                      response_map_feature_settings=feature_extr_settings,
                                                                      response_map_feature_families=None)

        # Deep learning branch
        deep_learning_branch = branch.find("deep_learning")

        if deep_learning_branch is not None:
            raise DeprecationWarning("deep_learning parameter branch has been deprecated. Parameters for image  "
                                     "processing for deep learning can now be set directly using the  "
                                     "extract_images_for_deep_learning function.")

        # Parse to settings
        settings_list += [SettingsClass(general_settings=general_settings,
                                        img_interpolate_settings=img_interp_settings,
                                        roi_interpolate_settings=roi_interp_settings,
                                        post_process_settings=post_process_settings,
                                        perturbation_settings=perturbation_settings,
                                        roi_resegment_settings=roi_resegment_settings,
                                        feature_extr_settings=feature_extr_settings,
                                        img_transform_settings=img_transform_settings)]

    return settings_list


def import_data_settings(path,
                         config_settings,
                         compute_features=False,
                         extract_images=False,
                         plot_images=False,
                         keep_images_in_memory=False,
                         file_structure=False,
                         **kwargs):
    from mirp.experimentClass import ExperimentClass
    import os
    import logging

    def find_sub_directories(dir_path):
        sub_dir = []
        for dir_file in os.listdir(dir_path):
            if os.path.isdir(os.path.join(dir_path, dir_file)):
                sub_dir.append(dir_file)
        return sub_dir

    def find_imaging_files(dir_path):
        file_found = False
        for dir_file in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, dir_file)):
                if dir_file.lower().endswith((".dcm", ".ima", ".nii", ".nii.gz", ".nifti", ".nifti.gz", ".nrrd")):
                    file_found = True
                    break
        return file_found

    # Configure logger
    logging.basicConfig(
        format="%(levelname)s\t: %(processName)s \t %(asctime)s \t %(message)s",
        level=logging.INFO,
        stream=sys.stdout)

    # Load xml
    tree = ElemTree.parse(path)
    root = tree.getroot()

    # Empty list for iteratively storing data objects
    data_obj_list = []

    # Iterate over configurations
    for branch in root.findall("config"):

        # Read data from xml file
        paths_branch = branch.find("paths")
        project_path = os.path.normpath(str2type(paths_branch.find("project_folder"), "path"))
        write_path = os.path.normpath(str2type(paths_branch.find("write_folder"), "path"))
        excl_subj = str2list(paths_branch.find("subject_exclude"), "str")
        incl_subj = str2list(paths_branch.find("subject_include"), "str")
        provide_diagnostics = str2type(paths_branch.find("provide_diagnostics"), "bool")

        # Read cohort name or ID
        cohort_id = str2type(paths_branch.find("cohort"), "str", "NA")

        # Identify subject folders
        folder_list = find_sub_directories(project_path)

        # Check if there are multiple data branches with the same modality, because then we need to update the subject.
        image_data_identifier_list = []
        for data_branch in branch.findall("data"):
            # Collect modality, image_folder and image_file_name_pattern
            image_data_identifier_list += [pd.DataFrame.from_dict(dict({
                "modality": [str2type(data_branch.find("modality"), "str")],
                "folder": [str2type(data_branch.find("image_folder"), "path")],
                "image_file_name_pattern": [str2type(data_branch.find("image_filename_pattern"), "str")]
            }))]

        # Concatenate to single data frame.
        image_data_identifier_list = pd.concat(image_data_identifier_list,
                                               ignore_index=True)

        # Populate image data identifiers aside from subject/
        n_unique_sets = image_data_identifier_list.shape[0]
        if image_data_identifier_list.drop_duplicates(subset=["modality"], inplace=False).shape[0] == n_unique_sets:
            image_data_identifiers = ["modality"]

        elif image_data_identifier_list.drop_duplicates(subset=["modality", "folder"], inplace=False).shape[0] == \
                n_unique_sets:
            image_data_identifiers = ["modality", "folder"]

        elif image_data_identifier_list.drop_duplicates(subset=["modality", "image_file_name_pattern"],
                                                        inplace=False).shape[0] == n_unique_sets:
            image_data_identifiers = ["modality", "file_name"]

        else:
            image_data_identifiers = ["modality", "folder", "file_name"]

        # Iterate over data branches
        for data_branch in branch.findall("data"):

            # Read current data branch
            image_modality = str2type(data_branch.find("modality"), "str")
            image_folder = str2type(data_branch.find("image_folder"), "path")
            roi_folder = str2type(data_branch.find("roi_folder"), "path")
            roi_reg_img_folder = str2type(data_branch.find("registration_image_folder"), "str")
            image_file_name_pattern = str2type(data_branch.find("image_filename_pattern"), "str")
            registration_image_file_name_pattern = str2type(data_branch.find("registration_image_filename_pattern"),
                                                            "str")
            roi_names = str2list(data_branch.find("roi_names"), "str")
            roi_list_path: str = str2type(data_branch.find("roi_list_path"), "str")
            divide_disconnected_roi = str2type(data_branch.find("divide_disconnected_roi"), "str", "combine")
            extraction_config = str2list(data_branch.find("extraction_config"), "str")

            # Check if extraction config has been set -- this allows setting configurations for mixed modalities
            if extraction_config is not None and config_settings is not None:
                new_config_settings = []

                # Iterate over configuration names mentioned in the data and compare those to the configuration strings in the settings
                # If a match is found, the configuration is set to the new configuration list.
                for config_name in extraction_config:
                    for ii in np.arange(len(config_settings)):
                        if config_settings[ii].general.config_str == config_name:
                            new_config_settings.append(config_settings[ii])

                if len(new_config_settings) == 0:
                    raise ValueError("No matching configuration strings were found in the settings file.")

            elif type(config_settings) is list:
                new_config_settings = config_settings
            else:
                new_config_settings = [config_settings]

            if not file_structure:
                # Check if image_folder has been set
                if image_folder is None:
                    logging.warning(
                        "No image folder was set. If images are located directly in the patient folder, use subject_dir as tag")
                    continue

                # Set correct paths in case folders are tagged with subject_dir. This tag indicates that the data is directly in the subject folder
                if image_folder == "subject_dir":
                    image_folder = ""
                if roi_folder == "subject_dir":
                    roi_folder = ""

                # Perform consistency check for roi folder, roi_names
                if roi_folder is None:
                    logging.info("No roi folder was configured. The roi folder reverts to the image folder.")
                    roi_folder = image_folder

                if roi_folder is not None and roi_names is None and roi_list_path is None:
                    logging.warning("No roi names were provided with the configuration.")

                if roi_reg_img_folder is None:
                    roi_reg_img_folder = image_folder

            else:
                image_folder = roi_folder = roi_reg_img_folder = ""

            # A separate file with roi names per sample may be provided in case there is no standardized name
            if roi_list_path is not None:
                roi_list = pd.read_csv(roi_list_path, sep=None, engine="python")

                # Convert sample_id to string
                roi_list["sample_id"] = roi_list["sample_id"].astype(str)

                # Check if column names are correct
                if not all([ii in roi_list.columns.values for ii in ["sample_id", "roi_name"]]):
                    raise ValueError(
                        "Column names in the provided roi list do not match \"sample_id\" or \"roi_name\".")
            else:
                roi_list = None

            # Iterate over subjects
            for curr_subj in folder_list:

                # Check if the current subject is included in the analysis
                if incl_subj is not None:
                    if curr_subj not in incl_subj:
                        logging.info("%s was excluded as per configuration.", curr_subj)
                        continue

                # Check if the current subject is excluded from the analysis
                if excl_subj is not None:
                    if curr_subj in excl_subj:
                        logging.info("%s was excluded as per configuration.", curr_subj)
                        continue

                # Set image folder and roi folder paths
                image_dir_subj = os.path.normpath(os.path.join(project_path, curr_subj, image_folder))
                roi_dir_subj = os.path.normpath(os.path.join(project_path, curr_subj, roi_folder))
                roi_reg_img_subj = os.path.normpath(os.path.join(project_path, curr_subj, roi_reg_img_folder))

                # Check if image and roi folders exist on the path
                if not os.path.isdir(image_dir_subj):
                    logging.info("%s was excluded as the image folder %s was not found.", curr_subj,
                                 os.path.join(image_dir_subj))
                    continue

                if not os.path.isdir(roi_dir_subj):
                    logging.info("%s was excluded as the roi folder %s was not found.", curr_subj,
                                 os.path.join(roi_dir_subj))
                    continue

                # Check if the image and roi folders contain files
                if not find_imaging_files(image_dir_subj) and not file_structure:
                    logging.info("%s was excluded as the image folder did not contain image files.", curr_subj)
                    continue

                if not find_imaging_files(roi_dir_subj) and not file_structure:
                    logging.info("%s was excluded as the roi folder did not contain image files.", curr_subj)
                    continue

                # Get rois for the current sample in case a roi_list was provided.
                if roi_list is not None:
                    roi_names = roi_list.loc[roi_list.sample_id == curr_subj, "roi_name"].values

                # Create data class object and add to list
                # For better parallellisation performance a data object only contains a single configurations file
                for curr_config_setting in new_config_settings:

                    # Set divide_disconnected_roi setting
                    if curr_config_setting is not None:
                        curr_config_setting.general.divide_disconnected_roi = divide_disconnected_roi

                    # Identify data string.
                    data_string = []
                    for data_identifier in image_data_identifiers:
                        if data_identifier == "modality":
                            data_string += [image_modality]

                        elif data_identifier == "folder":
                            if image_folder is not None:
                                image_folder_str = image_folder.replace("/", "_").replace("\\", "_")
                                data_string += [image_folder_str]

                        elif data_identifier == "file_name":
                            if image_file_name_pattern is not None:
                                data_string += [image_file_name_pattern]
                        else:
                            raise ValueError(f"Encountered an unexpected data_identifier: {data_identifier}")

                    data_obj = ExperimentClass(modality=image_modality,
                                               subject=curr_subj,
                                               cohort=cohort_id,
                                               write_path=write_path,
                                               image_folder=image_dir_subj,
                                               roi_folder=roi_dir_subj,
                                               roi_reg_img_folder=roi_reg_img_subj,
                                               image_file_name_pattern=image_file_name_pattern,
                                               registration_image_file_name_pattern=registration_image_file_name_pattern,
                                               roi_names=roi_names,
                                               data_str=data_string,
                                               provide_diagnostics=provide_diagnostics,
                                               settings=curr_config_setting,
                                               compute_features=compute_features,
                                               extract_images=extract_images,
                                               plot_images=plot_images,
                                               keep_images_in_memory=keep_images_in_memory)

                    data_obj_list.append(data_obj)

    return data_obj_list


def read_node(tree: xml.etree.ElementTree.Element,
              node: Union[str, List[str]],
              deprecated_node: Union[None, str, List[str]] = None):
    """
    :param tree: Tree element
    :param node: Name or list of names for each tree element.
    :param deprecated_node: Deprecated name.
    :return:
    """

    # Turn node into a list.
    if not isinstance(node, list):
        node = [node]

    # Throw deprecation warnings if necessary.
    if deprecated_node is not None:
        if not isinstance(deprecated_node, list):
            deprecated_node = [deprecated_node]

        for current_node in deprecated_node:
            if tree.find(current_node) is not None:
                raise DeprecationWarning(f"The {current_node} has been deprecated. Use {', '.join(node)} instead.")

        # Append deprecated nodes to node.
        node += deprecated_node

    # Cycle over node, and return first instance without None.
    for current_node in node:
        node_contents = tree.find(current_node)
        if node_contents is not None:
            return node_contents

    return None
