import copy
from typing import Any
from dataclasses import dataclass

import numpy as np

from mirp.settings.settingsFeatureExtraction import FeatureExtractionSettingsClass
from mirp.settings.utilities import setting_def


@dataclass
class ImageTransformationSettingsClass:
    """
    Parameters related to image transformation using filters. Many parameters are conditional on the selected image
    filter (``filter_kernels``). By default, only statistical features are computed from filtered images.

    .. note::
        Many feature extraction parameters are copied from
        :class:`~mirp.settings.settingsFeatureExtraction.FeatureExtractionSettingsClass`, except
        ``response_map_feature_families``, ``response_map_discretisation_method`` and
        ``response_map_discretisation_n_bins``. If other parameters need to be changed from their default settings,
        first create an object of the current class (
        :class:`~mirp.settings.settingsImageTransformation.ImageTransformationSettingsClass`), and then update the
        attributes.

    Parameters
    ----------
    by_slice: str or bool, optional, default: False
        Defines whether calculations should be performed in 2D (True) or 3D (False), or alternatively only in the
        largest slice ("largest"). See :class:`~mirp.settings.settingsGeneral.GeneralSettingsClass`.

    ibsi_compliant: bool, optional, default: True
        Limits use of filters to those that exist in the IBSI reference standard.

    response_map_feature_families: str or list of str, optional, default: "statistics"
        Determines the feature families for which features are computed from response maps (filtered images). Radiomics
        features are implemented as defined in the IBSI reference manual. The following feature families can be
        computed from response maps:

        * Local intensity features: "li", "loc.int", "loc_int", "local_int", and "local_intensity".
        * Intensity-based statistical features: "st", "stat", "stats", "statistics", and "statistical".
        * Intensity histogram features: "ih", "int_hist", "int_histogram", and "intensity_histogram".
        * Intensity-volume histogram features: "ivh", "int_vol_hist", and "intensity_volume_histogram".
        * Grey level co-occurrence matrix (GLCM) features: "cm", "glcm", "grey_level_cooccurrence_matrix",
          and "cooccurrence_matrix".
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

        In addition, the following tags can be used:

        * "none": no features are computed.
        * "all": all features are computed.

        A list of tags may be provided to select multiple feature families. Morphological features are not computed
        from response maps, because these are mask-based and are invariant to filtering.

    response_map_discretisation_method: {"fixed_bin_number", "fixed_bin_size", "fixed_bin_size_pyradiomics", "none"}, optional, default: "fixed_bin_number"
        Method used for discretising intensities. Used to compute intensity histogram as well as texture features.
        The setting is ignored if none of these feature families are being computed. The following options are
        available:

        * "fixed_bin_number": The intensity range within the mask is divided into a fixed number of bins,
          defined by the ``base_discretisation_bin_width`` parameter.
        * "fixed_bin_size": The intensity range is divided into bins with a fixed width, defined using the
          ``base_discretisation_bin_width`` parameter. The lower bound of the range is determined from the lower
          bound of the mask resegmentation range, see the ``resegmentation_intensity_range`` in
          :class:`~mirp.settings.settingsMaskResegmentation.ResegmentationSettingsClass`. Other images,
          including MRI, normalised CT and PET images and filtered images, do not have a default value, and bins are
          created from using the minimum intensity as lower bound.
        * "fixed_bin_size_pyradiomics": The intensity range is divided into bins with a fixed width. This follows the
          non-IBSI compliant implementation in the pyradiomics package.
        * "none": The intensity range is not discretised into bins. This method can only be used if the image
          intensities are integer and strictly positive.

        Multiple discretisation methods can be specified as a list to yield features according to each method.

        .. note::
            Use of the "fixed_bin_size", "fixed_bin_size_pyradiomics", and "none" discretisation methods is discouraged
            for transformed images. Due to transformation, a direct link to any meaningful quantity represented by the
            intensity of the original image (e.g. Hounsfield Units for CT, Standardised Uptake Value for PET) is lost.

    response_map_discretisation_n_bins: int or list of int, optional, default: 16
        Number of bins used for the "fixed_bin_number" discretisation method. Multiple values can be specified in a
        list to yield features according to each number of bins.

    response_map_discretisation_bin_width: float or list of float, optional
        Width of each bin in the "fixed_bin_size" and "fixed_bin_size_pyradiomics" discretisation methods. Multiple
        values can be specified in a list to yield features according to each bin width.

    filter_kernels: str or list of str, optional, default: None
        Names of the filters applied to the original image to create response maps (filtered images). Filter
        implementation follows the IBSI reference manual. The following filters are supported:

        * Mean filters: "mean"
        * Gaussian filters: "gaussian", "riesz_gaussian", and "riesz_steered_gaussian"
        * Laplacian-of-Gaussian filters: "laplacian_of_gaussian", "log", "riesz_laplacian_of_gaussian",
          "riesz_log", "riesz_steered_laplacian_of_gaussian", and "riesz_steered_log".
        * Laws kernels: "laws"
        * Gabor kernels: "gabor", "riesz_gabor", and "riesz_steered_gabor"
        * Separable wavelets: "separable_wavelet"
        * Non-separable wavelets: "nonseparable_wavelet", "riesz_nonseparable_wavelet",
          and "riesz_steered_nonseparable_wavelet"
        * Function transformations: "pyradiomics_square", "pyradiomics_square_root",
          and "pyradiomics_logarithm", "pyradiomics_exponential"

        Filters with names that preceded by "riesz" undergo a Riesz transformation. If the filter name is preceded by
        "riesz_steered", a steerable riesz filter is used.

        More than one filter name can be provided. By default, no filters are selected, and image transformation is
        skipped.

        .. note::
            There is no IBSI reference standard for Gaussian filters. However, the filter implementation is relatively
            straightforward, and most likely reproducible.

        .. warning::
            Riesz transformation and steerable riesz transformations are experimental. The implementation of these
            filter transformations is complex. Since there is no corresponding IBSI reference standard, any feature
            derived from response maps of Riesz transformations is unlikely to be reproducible.

        .. warning::
            Function transformations (square, square root, logarithm, exponential) do not have an IBSI reference
            standard. These transformations follow the definition in pyradiomics, and have been implemented for
            validation purposes.

    boundary_condition: {"reflect", "constant", "nearest", "mirror", "wrap"}, optional, default: "mirror"
        Sets the boundary condition, which determines how filters behave at the edge of an image. MIRP uses
        the same nomenclature for boundary conditions as scipy.ndimage. See the ``mode`` parameter of
        `scipy.ndimage.convolve
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html#scipy.ndimage.convolve>`_

    separable_wavelet_families: str or list str
        Name of separable wavelet kernel as implemented in the ``pywavelets`` package. See `pywt.wavelist(
        kind="discrete") <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#built-in-wavelets-wavelist>`_
        for options.

    separable_wavelet_set: str or list of str, optional
        Filter orientation of separable wavelets. Allows for specifying combinations for high and low-pass filters.
        For 2D (``by_slice=True``) filters, the following sets are possible: "hh", "hl", "lh", "ll" (y-x directions).
        For 3D (``by_slice=False``) filters, the set of possibilities is larger: "hhh", "hhl", "hlh", "lhh", "hll",
        "lhl", "llh", "lll". More than one  orientation may be set. Default: "hh" (2d) or "hhh (3d).

    separable_wavelet_stationary: bool, optional, default: True
        Determines if wavelets are stationary or not. Stationary wavelets maintain the image dimensions after
        decomposition.

    separable_wavelet_decomposition_level: int or list of int, optional, default: 1
        Sets the wavelet decomposition level. For the first decomposition level, the base image is used as input to
        generate a  response map. For decomposition levels greater than 1, the low-pass image from the previous level
        is used as input. More than 1 value may be specified in a list.

    separable_wavelet_rotation_invariance: bool, optional, default: True
        Determines whether separable filters are applied in a pseudo-rotational invariant manner. This generates
        permutations of the filter and, as a consequence, additional response maps. These maps are then merged using
        the pooling method (``separable_wavelet_pooling_method``).

    separable_wavelet_pooling_method: {"max", "min", "mean", "sum"}, optional, default: "max"
        Response maps are pooled to create a rotationally invariant response map. This sets the method for
        pooling.

        * "max": Each voxel of the pooled response map represents the maximum value for that voxel in the underlying
          response maps.
        * "min": Each voxel of the pooled response map represents the minimum value for that voxel in the underlying
          response maps.
        * "mean": Each voxel of the pooled response map represents the mean value for that voxel in the underlying
          response maps. For band-pass and high-pass filters, this will likely result in values close to 0.0,
          and "max" or "min" pooling methods should be used instead.
        * "sum": Each voxel of the pooled response map is the sum of intensities for that voxel in the underlying
          response maps. Similar to the "mean" pooling method, but without the normalisation.

    separable_wavelet_boundary_condition: str, optional, default: "mirror"
        Sets the boundary condition for separable wavelets. This supersedes any value set by the general
        ``boundary_condition`` parameter. See the ``boundary_condition`` parameter above for all valid options.

    nonseparable_wavelet_families: {"shannon", "simoncelli"}
        Name of non-separable wavelet kernels used for image transformation. Shannon and Simoncelli wavelets are
        implemented.

    nonseparable_wavelet_decomposition_level: int or list of int, optional, default: 1
        Sets the wavelet decomposition level. Unlike the decomposition level in separable wavelets, decomposition of
        non-separable wavelets is purely a filter-based operation.

    nonseparable_wavelet_response: {"modulus", "abs", "magnitude", "angle", "phase", "argument", "real", "imaginary"}, optional, default: "real"
        Nonseparable wavelets produce response maps with complex numbers. The complex-valued response map is
        converted to a real-valued response map using the specified method. "modulus", "abs", "magnitude" are
        synonymous, as are "angle", "phase", and "argument". "real" selects the real component of the complex values,
        and "imaginary" selects the imaginary component.

    nonseparable_wavelet_boundary_condition: str, optional, default: "mirror"
        Sets the boundary condition for non-separable wavelets. This supersedes any value set by the general
        ``boundary_condition`` parameter. See the ``boundary_condition`` parameter above for all valid options.

    gaussian_sigma: float or list of float, optional
         Width of the Gaussian filter in physical dimensions (e.g. mm). Multiple values can be specified.

    gaussian_kernel_truncate: float, optional, default: 4.0
        Width, in units of sigma, at which the filter is truncated.

    gaussian_kernel_boundary_condition: str, optional, default: "mirror"
        Sets the boundary condition for Gaussian filters. This supersedes any value set by the general
        ``boundary_condition`` parameter. See the ``boundary_condition`` parameter above for all valid options.

    laplacian_of_gaussian_sigma: float or list of float, optional
        Width of the Gaussian filter in physical dimensions (e.g. mm). Multiple values can be specified.

    laplacian_of_gaussian_kernel_truncate: float, optional, default: 4.0
        Width, in sigma, at which the filter is truncated.

    laplacian_of_gaussian_pooling_method: {"max", "min", "mean", "sum", "none"}, optional, default: "none"
        Determines whether and how response maps for filters with different widths (``laplacian_of_gaussian_sigma``)
        are pooled.

        * "max": Each voxel of the pooled response map represents the maximum value for that voxel in the underlying
          response maps.
        * "min": Each voxel of the pooled response map represents the minimum value for that voxel in the underlying
          response maps.
        * "mean": Each voxel of the pooled response map represents the mean value for that voxel in the underlying
          response maps. For band-pass and high-pass filters, this will likely result in values close to 0.0,
          and "max" or "min" pooling methods should be used instead.
        * "sum": Each voxel of the pooled response map is the sum of intensities for that voxel in the underlying
          response maps. Similar to the "mean" pooling method, but without the normalisation.
        * "none": Each Laplacian-of-Gaussian response map is treated separately, without pooling.

    laplacian_of_gaussian_boundary_condition: str, optional, default: "mirror"
        Sets the boundary condition for Laplacian-of-Gaussian filters. This supersedes any value set by the general
        ``boundary_condition`` parameter. See the ``boundary_condition`` parameter above for all valid options.

    laws_kernel: str or list of str, optional
        Compute specific Laws kernels these typically are specific combinations of kernels such as L5S5E5,
        E5E5E5. The following kernels are available: 'l5', 'e5', 's5', 'w5', 'r5', 'l3', 'e3', 's3'. A combination of
        two kernels is expected for 2D (``by_slice=True``), whereas a kernel triplet is expected for 3D filters (
        ``by_slice=False``).

    laws_compute_energy: bool, optional, default: True
        Determine whether an energy image should be computed, or just the response map.

    laws_delta: int or list of int, optional, default: 7
        Delta for chebyshev distance between center voxel and neighbourhood boundary used to calculate energy maps.

    laws_rotation_invariance: bool, optional, default: True
        Determines whether separable filters are applied in a pseudo-rotational invariant manner. This generates
        permutations of the filter and, as a consequence, additional response maps. These maps are then merged using
        the pooling method (``laws_pooling_method``).

    laws_pooling_method:  {"max", "min", "mean", "sum"}, optional, default: "max"
        Response maps are pooled to create a rotationally invariant response map. This sets the method for
        pooling.

        * "max": Each voxel of the pooled response map represents the maximum value for that voxel in the underlying
          response maps.
        * "min": Each voxel of the pooled response map represents the minimum value for that voxel in the underlying
          response maps.
        * "mean": Each voxel of the pooled response map represents the mean value for that voxel in the underlying
          response maps. For band-pass and high-pass filters, this will likely result in values close to 0.0,
          and "max" or "min" pooling methods should be used instead.
        * "sum": Each voxel of the pooled response map is the sum of intensities for that voxel in the underlying
          response maps. Similar to the "mean" pooling method, but without the normalisation.

    laws_boundary_condition: str, optional, default: "mirror"
        Sets the boundary condition for Laws filters. This supersedes any value set by the general
        ``boundary_condition`` parameter. See the ``boundary_condition`` parameter above for all valid options.

    gabor_sigma: float or list of float, optional
        Width of the Gaussian envelope in physical dimensions (e.g. mm). Multiple values can be specified.

    gabor_lambda: float or list of float, optional
        Wavelength of the oscillator component of the Gabor filter, in physical dimensions (e.g. mm).

    gabor_gamma: float or list of float, optional, default: 1.0
        Eccentricity parameter of the Gaussian envelope of the Gabor kernel. Defines width of y-axis relative to
        x-axis for 0-angle Gabor kernel. Default: 1.0

    gabor_theta: float or list of flaot, optional, default: 0.0
        Initial angle of the Gabor filter in degrees (not radians). Multiple angles can be provided.

    gabor_theta_step: float, optional, default: None
        Angle step size in degrees for in-plane rotational invariance. A value of 0.0 or None (default) disables
        stepping.

    gabor_response: {"modulus", "abs", "magnitude", "angle", "phase", "argument", "real", "imaginary"}, optional, default: "modulus"
        Type of response map created by Gabor filters. Gabor kernels consist of complex numbers, and the response map
        will be complex as well. The complex-valued response map is converted to a real-valued response map using the
        specified method.

    gabor_rotation_invariance: bool, optional, default: False
        Determines whether (2D) Gabor filters are applied in a pseudo-rotational invariant manner. If True,
        Gabor filters are applied in each of the orthogonal planes.

    gabor_pooling_method: {"max", "min", "mean", "sum"}, optional, default: "max"
        Response maps are pooled to create a rotationally invariant response map. This sets the method for
        pooling.

        * "max": Each voxel of the pooled response map represents the maximum value for that voxel in the underlying
          response maps.
        * "min": Each voxel of the pooled response map represents the minimum value for that voxel in the underlying
          response maps.
        * "mean": Each voxel of the pooled response map represents the mean value for that voxel in the underlying
          response maps. For band-pass and high-pass filters, this will likely result in values close to 0.0,
          and "max" or "min" pooling methods should be used instead.
        * "sum": Each voxel of the pooled response map is the sum of intensities for that voxel in the underlying
          response maps. Similar to the "mean" pooling method, but without the normalisation.

    gabor_boundary_condition: str, optional, default: "mirror"
        Sets the boundary condition for Gabor filters. This supersedes any value set by the general
        ``boundary_condition`` parameter. See the ``boundary_condition`` parameter above for all valid options.

    mean_filter_kernel_size: int or list of int, optional
        Length of the kernel in pixels. Multiple values can be specified to create multiple response maps.

    mean_filter_boundary_condition: str, optional, default: "mirror"
        Sets the boundary condition for mean filters. This supersedes any value set by the general
        ``boundary_condition`` parameter. See the ``boundary_condition`` parameter above for all valid options.

    riesz_filter_order: float, list of float or list of list of float, optional
        Riesz-transformation order. If required, should be a 2 (2D filter), or 3-element (3D filter) integer
        vector, e.g. [0,0,1]. Multiple sets can be provided by nesting the list, e.g. [[0, 0, 1],
        [0, 1, 0]]. If an integer is provided, a set of filters is created. For example when
        riesz_filter_order = 2 and a 2D filter is used, the following Riesz-transformations are performed: [2,
        0], [1, 1] and [0, 2].

        .. note::
            Riesz filter order uses the numpy coordinate ordering and represents (z, y, x) directions.

    riesz_filter_tensor_sigma: float or list of float, optional
        Determines width of Gaussian filter used with Riesz filter banks.

    **kwargs: dict, optional
        Unused keyword arguments.

    """

    def __init__(
            self,
            by_slice: bool,
            ibsi_compliant: bool = True,
            response_map_feature_settings: FeatureExtractionSettingsClass | None = None,
            response_map_feature_families: None | str | list[str] = "statistical",
            response_map_discretisation_method: None | str | list[str] = "fixed_bin_number",
            response_map_discretisation_n_bins: None | int | list[int] = 16,
            response_map_discretisation_bin_width: None | int | list[float] = None,
            filter_kernels: None | str | list[str] = None,
            boundary_condition: None | str = "mirror",
            separable_wavelet_families: None | str | list[str] = None,
            separable_wavelet_set: None | str | list[str] = None,
            separable_wavelet_stationary: bool = True,
            separable_wavelet_decomposition_level: None | int | list[int] = 1,
            separable_wavelet_rotation_invariance: bool = True,
            separable_wavelet_pooling_method: str = "max",
            separable_wavelet_boundary_condition: None | str = None,
            nonseparable_wavelet_families: None | str | list[str] = None,
            nonseparable_wavelet_decomposition_level: None | int | list[int] = 1,
            nonseparable_wavelet_response: None | str = "real",
            nonseparable_wavelet_boundary_condition: None | str = None,
            gaussian_sigma: None | float | list[float] = None,
            gaussian_kernel_truncate: None | float = 4.0,
            gaussian_kernel_boundary_condition: None | str = None,
            laplacian_of_gaussian_sigma: None | float | list[float] = None,
            laplacian_of_gaussian_kernel_truncate: None | float = 4.0,
            laplacian_of_gaussian_pooling_method: str = "none",
            laplacian_of_gaussian_boundary_condition: None | str = None,
            laws_kernel: None | str | list[str] = None,
            laws_delta: int | list[int] = 7,
            laws_compute_energy: bool = True,
            laws_rotation_invariance: bool = True,
            laws_pooling_method: str = "max",
            laws_boundary_condition: None | str = None,
            gabor_sigma: None | float | list[float] = None,
            gabor_lambda: None | float | list[float] = None,
            gabor_gamma: None | float | list[float] = 1.0,
            gabor_theta: None | float | list[float] = 0.0,
            gabor_theta_step: None | float = None,
            gabor_response: str = "modulus",
            gabor_rotation_invariance: bool = False,
            gabor_pooling_method: str = "max",
            gabor_boundary_condition: None | str = None,
            mean_filter_kernel_size: None | int | list[int] = None,
            mean_filter_boundary_condition: None | str = None,
            riesz_filter_order: None | int | list[int] = None,
            riesz_filter_tensor_sigma: None | float | list[float] = None,
            **kwargs
    ):
        # Set by slice
        self.by_slice: bool = by_slice

        # Set IBSI-compliance flag.
        self.ibsi_compliant: bool = ibsi_compliant

        # Check filter kernels
        if not isinstance(filter_kernels, list):
            filter_kernels = [filter_kernels]

        if any(filter_kernel is None for filter_kernel in filter_kernels):
            filter_kernels = None

        if filter_kernels is not None:
            # Check validity of the filter kernel names.
            valid_kernels: list[bool] = [ii in self.get_available_image_filters() for ii in filter_kernels]

            if not all(valid_kernels):
                raise ValueError(
                    f"One or more kernels are not implemented, or were spelled incorrectly: "
                    f"{', '.join([filter_kernel for ii, filter_kernel in enumerate(filter_kernels) if not valid_kernels[ii]])}")

        self.spatial_filters: None | list[str] = filter_kernels

        # Check families.
        if response_map_feature_families is None:
            response_map_feature_families = "none"

        if not isinstance(response_map_feature_families, list):
            response_map_feature_families = [response_map_feature_families]

        # Check which entries are valid.
        valid_families: list[bool] = [ii in [
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
            raise ValueError(
                f"One or more families in the base_feature_families parameter were not recognised: "
                f"{', '.join([response_map_feature_families[ii] for ii, is_valid in enumerate(valid_families) if not is_valid])}"
            )

        # Create a temporary feature settings object. If response_map_feature_settings is not present, this object is
        # used. Otherwise, response_map_feature_settings is copied, and then updated.
        if response_map_feature_settings is None:

            kwargs = copy.deepcopy(kwargs)
            kwargs.update({
                "base_feature_families": response_map_feature_families,
                "base_discretisation_method": response_map_discretisation_method,
                "base_discretisation_bin_width": response_map_discretisation_bin_width,
                "base_discretisation_n_bins": response_map_discretisation_n_bins
            })

            response_map_feature_settings = FeatureExtractionSettingsClass(
                by_slice=by_slice,
                no_approximation=False,
                ibsi_compliant=ibsi_compliant,
                **kwargs
            )

        # Set feature settings.
        self.feature_settings: FeatureExtractionSettingsClass = response_map_feature_settings

        # Check boundary condition.
        self.boundary_condition = boundary_condition
        self.boundary_condition: str = self.check_boundary_condition(
            boundary_condition,
            "boundary_condition")

        # Check mean filter settings
        if self.has_mean_filter():
            # Check filter size.
            if not isinstance(mean_filter_kernel_size, list):
                mean_filter_kernel_size = [mean_filter_kernel_size]

            if not all(isinstance(kernel_size, int) for kernel_size in mean_filter_kernel_size):
                raise TypeError(
                    f"All kernel sizes for the mean filter are expected to be integer values equal or "
                    f"greater than 1. Found: one or more kernel sizes that were not integers.")

            if not all(kernel_size >= 1 for kernel_size in mean_filter_kernel_size):
                raise ValueError(
                    f"All kernel sizes for the mean filter are expected to be integer values equal or "
                    f"greater than 1. Found: one or more kernel sizes less then 1.")

            # Check boundary condition
            mean_filter_boundary_condition = self.check_boundary_condition(
                mean_filter_boundary_condition,
                "mean_filter_boundary_condition")

        else:
            mean_filter_kernel_size = None
            mean_filter_boundary_condition = None

        self.mean_filter_size: None | list[int] = mean_filter_kernel_size
        self.mean_filter_boundary_condition: None | str = mean_filter_boundary_condition

        # Check Gaussian kernel settings.
        if self.has_gaussian_filter():
            # Check sigma.
            gaussian_sigma = self.check_sigma(
                gaussian_sigma,
                "gaussian_sigma")

            # Check filter truncation.
            gaussian_kernel_truncate = self.check_truncation(
                gaussian_kernel_truncate,
                "gaussian_kernel_truncate")

            # Check boundary condition
            gaussian_kernel_boundary_condition = self.check_boundary_condition(
                gaussian_kernel_boundary_condition,
                "gaussian_kernel_boundary_condition")

        else:
            gaussian_sigma = None
            gaussian_kernel_truncate = None
            gaussian_kernel_boundary_condition = None

        self.gaussian_sigma: None | list[float] = gaussian_sigma
        self.gaussian_sigma_truncate: None | float = gaussian_kernel_truncate
        self.gaussian_boundary_condition: None | str = gaussian_kernel_boundary_condition

        # Check laplacian-of-gaussian filter settings
        if self.has_laplacian_of_gaussian_filter():
            # Check sigma.
            laplacian_of_gaussian_sigma = self.check_sigma(
                laplacian_of_gaussian_sigma,
                "laplacian_of_gaussian_sigma")

            # Check filter truncation.
            laplacian_of_gaussian_kernel_truncate = self.check_truncation(
                laplacian_of_gaussian_kernel_truncate,
                "laplacian_of_gaussian_kernel_truncate")

            # Check pooling method.
            laplacian_of_gaussian_pooling_method = self.check_pooling_method(
                laplacian_of_gaussian_pooling_method,
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

        self.log_sigma: None | list[float] = laplacian_of_gaussian_sigma
        self.log_sigma_truncate: None | float = laplacian_of_gaussian_kernel_truncate
        self.log_pooling_method: None | str = laplacian_of_gaussian_pooling_method
        self.log_boundary_condition: None | str = laplacian_of_gaussian_boundary_condition

        # Check Laws kernel filter settings
        if self.has_laws_filter():
            # Check kernel.
            laws_kernel = self.check_laws_kernels(laws_kernel, "laws_kernel")

            # Check energy computation.
            if not isinstance(laws_compute_energy, bool):
                raise TypeError("The laws_compute_energy parameter is expected to be a boolean value.")

            if laws_compute_energy:

                # Check delta.
                if not isinstance(laws_delta, list):
                    laws_delta = [laws_delta]

                if not all(isinstance(delta, int) for delta in laws_delta):
                    raise TypeError(
                        "The laws_delta parameter is expected to be one or more integers with value 0 or "
                        "greater. Found: one or more values that are not integer.")

                if not all(delta >= 0 for delta in laws_delta):
                    raise ValueError(
                        "The laws_delta parameter is expected to be one or more integers with value 0 or "
                        "greater. Found: one or more values that are less than 0.")

            else:
                laws_delta = None

            # Check invariance.
            if not isinstance(laws_rotation_invariance, bool):
                raise TypeError("The laws_rotation_invariance parameter is expected to be a boolean value.")

            # Check pooling method.
            laws_pooling_method = self.check_pooling_method(laws_pooling_method, "laws_pooling_method")

            # Check boundary condition
            laws_boundary_condition = self.check_boundary_condition(laws_boundary_condition, "laws_boundary_condition")

        else:
            laws_kernel = None
            laws_compute_energy = None,
            laws_delta = None
            laws_rotation_invariance = None
            laws_pooling_method = None
            laws_boundary_condition = None

        self.laws_calculate_energy: None | bool = laws_compute_energy
        self.laws_kernel: None | list[str] = laws_kernel
        self.laws_delta: None | bool = laws_delta
        self.laws_rotation_invariance: None | bool = laws_rotation_invariance
        self.laws_pooling_method: None | str = laws_pooling_method
        self.laws_boundary_condition: None | str = laws_boundary_condition

        # Check Gabor filter settings.
        if self.has_gabor_filter():
            # Check sigma.
            gabor_sigma = self.check_sigma(gabor_sigma, "gabor_sigma")

            # Check gamma. Gamma behaves like sigma.
            gabor_gamma = self.check_sigma(gabor_gamma, "gabor_gamma")

            # Check lambda. Lambda behaves like sigma
            gabor_lambda = self.check_sigma(gabor_lambda, "gabor_lambda")

            # Check theta step.
            if gabor_theta_step is not None:
                if not isinstance(gabor_theta_step, (float, int)):
                    raise TypeError(
                        "The gabor_theta_step parameter is expected to be an angle, in degrees. Found a "
                        "value that was not a number.")

                if gabor_theta_step == 0.0:
                    gabor_theta_step = None

            if gabor_theta_step is not None:
                # Check that the step would divide the 360-degree circle into an integer number of steps.
                if not (360.0 / gabor_theta_step).is_integer():
                    raise ValueError(
                        f"The gabor_theta_step parameter should divide a circle into equal portions. "
                        f"The current settings would create {360.0 / gabor_theta_step} portions.")

            # Check theta.
            gabor_pool_theta = gabor_theta_step is not None

            if not isinstance(gabor_theta, list):
                gabor_theta = [gabor_theta]

            if gabor_theta_step is not None and len(gabor_theta) > 1:
                raise ValueError(
                    f"The gabor_theta parameter cannot have more than one value when used in conjunction"
                    f" with the gabor_theta_step parameter")

            if not all(isinstance(theta, (float, int)) for theta in gabor_theta):
                raise TypeError(
                    f"The gabor_theta parameter is expected to be one or more values indicating angles in"
                    f" degrees. Found: one or more values that were not numeric.")

            if gabor_theta_step is not None:
                gabor_theta = [gabor_theta[0] + ii for ii in np.arange(0.0, 360.0, gabor_theta_step)]

            # Check filter response.
            gabor_response = self.check_response(gabor_response, "gabor_response")

            # Check rotation invariance
            if not isinstance(gabor_rotation_invariance, bool):
                raise TypeError("The gabor_rotation_invariance parameter is expected to be a boolean value.")

            # Check pooling method
            gabor_pooling_method = self.check_pooling_method(gabor_pooling_method, "gabor_pooling_method")

            # Check boundary condition
            gabor_boundary_condition = self.check_boundary_condition(
                gabor_boundary_condition, "gabor_boundary_condition")

        else:
            gabor_sigma = None
            gabor_gamma = None
            gabor_lambda = None
            gabor_theta = None
            gabor_pool_theta = None
            gabor_response = None
            gabor_rotation_invariance = None
            gabor_pooling_method = None
            gabor_boundary_condition = None

        self.gabor_sigma: None | list[float] = gabor_sigma
        self.gabor_gamma: None | list[float] = gabor_gamma
        self.gabor_lambda: None | list[float] = gabor_lambda
        self.gabor_theta: None | list[float] | list[int] = gabor_theta
        self.gabor_pool_theta: None | bool = gabor_pool_theta
        self.gabor_response: None | str = gabor_response
        self.gabor_rotation_invariance: None | str = gabor_rotation_invariance
        self.gabor_pooling_method: None | str = gabor_pooling_method
        self.gabor_boundary_condition: None | str = gabor_boundary_condition

        # Check separable wavelet settings.
        if self.has_separable_wavelet_filter():
            # Check wavelet families.
            separable_wavelet_families = self.check_separable_wavelet_families(
                separable_wavelet_families, "separable_wavelet_families")

            # Check wavelet filter sets.
            separable_wavelet_set = self.check_separable_wavelet_sets(separable_wavelet_set, "separable_wavelet_set")

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
            separable_wavelet_pooling_method = self.check_pooling_method(
                separable_wavelet_pooling_method, "separable_wavelet_pooling_method")

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

        self.separable_wavelet_families: None | list[str] = separable_wavelet_families
        self.separable_wavelet_filter_set: None | list[str] = separable_wavelet_set
        self.separable_wavelet_stationary: None | bool = separable_wavelet_stationary
        self.separable_wavelet_decomposition_level: None | list[int] = separable_wavelet_decomposition_level
        self.separable_wavelet_rotation_invariance: None | bool = separable_wavelet_rotation_invariance
        self.separable_wavelet_pooling_method: None | str = separable_wavelet_pooling_method
        self.separable_wavelet_boundary_condition: None | str = separable_wavelet_boundary_condition

        # Set parameters for non-separable wavelets.
        if self.has_nonseparable_wavelet_filter():
            # Check wavelet families.
            nonseparable_wavelet_families = self.check_nonseparable_wavelet_families(
                nonseparable_wavelet_families, "nonseparable_wavelet_families")

            # Check decomposition level.
            nonseparable_wavelet_decomposition_level = self.check_decomposition_level(
                nonseparable_wavelet_decomposition_level, "nonseparable_wavelet_decomposition_level")

            # Check filter response.
            nonseparable_wavelet_response = self.check_response(
                nonseparable_wavelet_response, "nonseparable_wavelet_response")

            # Check boundary condition.
            nonseparable_wavelet_boundary_condition = self.check_boundary_condition(
                nonseparable_wavelet_boundary_condition, "nonseparable_wavelet_boundary_condition")

        else:
            nonseparable_wavelet_families = None
            nonseparable_wavelet_decomposition_level = None
            nonseparable_wavelet_response = None
            nonseparable_wavelet_boundary_condition = None

        self.nonseparable_wavelet_families: None | list[str] = nonseparable_wavelet_families
        self.nonseparable_wavelet_decomposition_level: None | list[int] = nonseparable_wavelet_decomposition_level
        self.nonseparable_wavelet_response: None | str = nonseparable_wavelet_response
        self.nonseparable_wavelet_boundary_condition: None | str = nonseparable_wavelet_boundary_condition

        # Check Riesz filter orders.
        if self.has_riesz_filter():
            riesz_filter_order = self.check_riesz_filter_order(riesz_filter_order, "riesz_filter_order")

        else:
            riesz_filter_order = None

        if self.has_steered_riesz_filter() and self.ibsi_compliant:
            raise ValueError(
                "The steered riesz filters are not part of the IBSI reference standard. If you are sure that you want "
                "to use this method, use ibsi_compliant = False."
            )

        elif self.has_riesz_filter():
            riesz_filter_tensor_sigma = self.check_sigma(riesz_filter_tensor_sigma, "riesz_filter_tensor_sigma")

        else:
            riesz_filter_tensor_sigma = None

        self.riesz_order: None | list[list[int]] = riesz_filter_order
        self.riesz_filter_tensor_sigma: None | list[float] = riesz_filter_tensor_sigma

        if ibsi_compliant and self.has_square_transform_filter():
            raise ValueError(
                "The square transformation filter is not part of the IBSI reference standard. If you are sure that "
                "you want to use this method, use ibsi_compliant = False."
            )
        if ibsi_compliant and self.has_square_root_transform_filter():
            raise ValueError(
                "The square root transformation filter is not part of the IBSI reference standard. If you are sure "
                "that you want to use this method, use ibsi_compliant = False."
            )
        if ibsi_compliant and self.has_logarithm_transform_filter():
            raise ValueError(
                "The logarithmic transformation filter is not part of the IBSI reference standard. If you are sure "
                "that you want to use this method, use ibsi_compliant = False."
            )
        if ibsi_compliant and self.has_exponential_transform_filter():
            raise ValueError(
                "The exponential transformation filter is not part of the IBSI reference standard. If you are sure "
                "that you want to use this method, use ibsi_compliant = False."
            )

    @staticmethod
    def get_available_image_filters():
        return [
            "separable_wavelet", "nonseparable_wavelet", "riesz_nonseparable_wavelet",
            "riesz_steered_nonseparable_wavelet", "gaussian", "riesz_gaussian", "riesz_steered_gaussian",
            "laplacian_of_gaussian", "log", "riesz_laplacian_of_gaussian", "riesz_steered_laplacian_of_gaussian",
            "riesz_log", "riesz_steered_log", "laws", "gabor", "riesz_gabor", "riesz_steered_gabor", "mean",
            "pyradiomics_square", "pyradiomics_square_root", "pyradiomics_logarithm", "pyradiomics_exponential"
        ]

    def check_boundary_condition(self, x, var_name):
        if x is None:
            if self.boundary_condition is not None:
                # Avoid updating by reference.
                x = copy.deepcopy(self.boundary_condition)

            else:
                raise ValueError(f"No value for the {var_name} parameter could be set, due to a lack of a default.")

        # Check value
        if x not in ["reflect", "constant", "nearest", "mirror", "wrap"]:
            raise ValueError(
                f"The provided value for the {var_name} is not valid. One of 'reflect', 'constant', "
                f"'nearest', 'mirror' or 'wrap' was expected. Found: {x}")

        return x

    @staticmethod
    def check_pooling_method(x, var_name, allow_none=False):

        valid_pooling_method = ["max", "min", "mean", "sum"]
        if allow_none:
            valid_pooling_method += ["none"]

        if x not in valid_pooling_method:
            raise ValueError(
                f"The {var_name} parameter expects one of the following values: "
                f"{', '.join(valid_pooling_method)}. Found: {x}")

        return x

    @staticmethod
    def check_sigma(x, var_name):
        # Check sigma is a list.
        if not isinstance(x, list):
            x = [x]

        # Check that the sigma values are floating points.
        if not all(isinstance(sigma, float) for sigma in x):
            raise TypeError(
                f"The {var_name} parameter is expected to consists of floating points with values "
                f"greater than 0.0. Found: one or more values that were not floating points.")

        if not all(sigma > 0.0 for sigma in x):
            raise ValueError(
                f"The {var_name} parameter is expected to consists of floating points with values "
                f"greater than 0.0. Found: one or more values with value 0.0 or less.")

        return x

    @staticmethod
    def check_truncation(x, var_name):

        # Check that the truncation values are floating points.
        if not isinstance(x, float):
            raise TypeError(
                f"The {var_name} parameter is expected to be a floating point with value "
                f"greater than 0.0. Found: a value that was not a floating point.")

        if not x > 0.0:
            raise ValueError(
                f"The {var_name} parameter is expected to be a floating point with value "
                f"greater than 0.0. Found: a value of 0.0 or less.")

        return x

    @staticmethod
    def check_response(x, var_name):

        valid_response = ["modulus", "abs", "magnitude", "angle", "phase", "argument", "real", "imaginary"]

        # Check that response is correct.
        if x not in valid_response:
            raise ValueError(
                f"The {var_name} parameter is not correct. Expected one of {', '.join(valid_response)}. "
                f"Found: {x}")

        return x

    @staticmethod
    def check_separable_wavelet_families(x, var_name):
        # Import pywavelets.
        import pywt

        if not isinstance(x, list):
            x = [x]

        available_kernels = pywt.wavelist(kind="discrete")
        valid_kernel = [kernel.lower() in available_kernels for kernel in x]

        if not all(valid_kernel):
            raise ValueError(
                f"The {var_name} parameter requires wavelet families that match those defined in the "
                f"pywavelets package. Could not match: "
                f"{', '.join([kernel for ii, kernel in x if not valid_kernel[ii]])}")

        # Return lowercase values.
        return [xx.lower() for xx in x]

    @staticmethod
    def check_nonseparable_wavelet_families(x, var_name):
        if not isinstance(x, list):
            x = [x]

        available_kernels = ["simoncelli", "shannon"]
        valid_kernel = [kernel.lower() in available_kernels for kernel in x]

        if not all(valid_kernel):
            raise ValueError(
                f"The {var_name} parameter expects one or more of the following values: "
                f"{', '.join(available_kernels)}. Could not match: "
                f"{', '.join([kernel for ii, kernel in x if not valid_kernel[ii]])}")

        # Return lowercase values.
        return [xx.lower() for xx in x]

    @staticmethod
    def check_decomposition_level(x, var_name):
        if not isinstance(x, list):
            x = [x]

        if not all(isinstance(xx, int) for xx in x):
            raise TypeError(
                f"The {var_name} parameter should be one or more integer "
                f"values of at least 1. Found: one or more values that was not an integer.")

        if not all(xx >= 1 for xx in x):
            raise ValueError(
                f"The {var_name} parameter should be one or more integer "
                f"values of at least 1. Found: one or more values that was not an integer.")

        return x

    def check_separable_wavelet_sets(self, x: None | str | list[str], var_name: str):
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
            raise ValueError(
                f"The {var_name} parameter requires combinations of low (l) and high-pass (h) kernels. "
                f"Two kernels should be specified for 2D, and three for 3D. Found the following invalid "
                f"combinations: "
                f"{', '.join([kernel for ii, kernel in enumerate(x) if not valid_kernel[ii]])}")

        # Return lowercase values.
        return [xx.lower() for xx in x]

    def check_laws_kernels(self, x: str | list[str], var_name: str):
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
            raise ValueError(
                f"The {var_name} parameter requires combinations of Laws kernels. The follow kernels are "
                f"implemented: {', '.join(kernels)}. Two kernels should be specified for 2D, "
                f"and three for 3D. Found the following illegal combinations: "
                f"{', '.join([kernel for ii, kernel in enumerate(x) if not valid_kernel[ii]])}")

        # Return lowercase values.
        return [xx.lower() for xx in x]

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
            single_filter_order = list(range(x+1))

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
            raise ValueError(
                f"The {var_name} parameter is expected to contain filter orders, each consisting of "
                f"{n_elements} non-negative integer values. One or more filter orders did not have the "
                f"expected number of elements.")

        if not all(all(isinstance(xxx, int) for xxx in xx) for xx in x):
            raise ValueError(
                f"The {var_name} parameter is expected to contain filter orders, each consisting of "
                f"{n_elements} non-negative integer values. One or more filter orders did not fully "
                f"consist of integer values.")

        if not all(all(xxx >= 0 for xxx in xx) for xx in x):
            raise ValueError(
                f"The {var_name} parameter is expected to contain filter orders, each consisting of "
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

        return x is not None and any(
            filter_kernel in ["gaussian", "riesz_gaussian", "riesz_steered_gaussian"] for filter_kernel in x)

    def has_laplacian_of_gaussian_filter(self, x=None):
        if x is None:
            x = self.spatial_filters
        elif not isinstance(x, list):
            x = [x]

        return x is not None and any(
            filter_kernel in [
                "laplacian_of_gaussian", "log", "riesz_laplacian_of_gaussian", "riesz_log",
                "riesz_steered_laplacian_of_gaussian", "riesz_steered_log"
            ] for filter_kernel in x)

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

        return x is not None and any(
            filter_kernel in ["gabor", "riesz_gabor", "riesz_steered_gabor"] for filter_kernel in x)

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

        return x is not None and any(
            filter_kernel in [
                "nonseparable_wavelet", "riesz_nonseparable_wavelet", "riesz_steered_nonseparable_wavelet"
            ] for filter_kernel in x)

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

    def has_square_transform_filter(self, x=None):
        if x is None:
            x = self.spatial_filters
        elif not isinstance(x, list):
            x = [x]

        return x is not None and any(filter_kernel == "pyradiomics_square" for filter_kernel in x)

    def has_square_root_transform_filter(self, x=None):
        if x is None:
            x = self.spatial_filters
        elif not isinstance(x, list):
            x = [x]

        return x is not None and any(filter_kernel == "pyradiomics_square_root" for filter_kernel in x)

    def has_logarithm_transform_filter(self, x=None):
        if x is None:
            x = self.spatial_filters
        elif not isinstance(x, list):
            x = [x]

        return x is not None and any(filter_kernel == "pyradiomics_logarithm" for filter_kernel in x)

    def has_exponential_transform_filter(self, x=None):
        if x is None:
            x = self.spatial_filters
        elif not isinstance(x, list):
            x = [x]

        return x is not None and any(filter_kernel == "pyradiomics_exponential" for filter_kernel in x)


def get_image_transformation_settings() -> list[dict[str, Any]]:
    return [
        setting_def(
            "response_map_feature_families", "str", to_list=True, xml_key="feature_families",
            class_key="families", test=["statistical", "glcm"]
        ),
        setting_def(
            "response_map_discretisation_method", "str", to_list=True, xml_key="discretisation_method",
            class_key="discretisation_method", test=["fixed_bin_size", "fixed_bin_number"]
        ),
        setting_def(
            "response_map_discretisation_n_bins", "int", to_list=True, xml_key="discretisation_n_bins",
            class_key="discretisation_n_bins", test=[10, 33]
        ),
        setting_def(
            "response_map_discretisation_bin_width", "float", to_list=True, xml_key="discretisation_bin_width",
            class_key="discretisation_bin_width", test=[10.0, 34.0]
        ),
        setting_def(
            "filter_kernels", "str", to_list=True, xml_key=["filter_kernels", "spatial_filters"],
            class_key="spatial_filters", test=[
                "separable_wavelet", "nonseparable_wavelet", "riesz_nonseparable_wavelet",
                "riesz_steered_nonseparable_wavelet", "gaussian", "riesz_gaussian", "riesz_steered_gaussian",
                "laplacian_of_gaussian", "log", "riesz_laplacian_of_gaussian", "riesz_steered_laplacian_of_gaussian",
                "riesz_log", "riesz_steered_log", "laws", "gabor", "riesz_gabor", "riesz_steered_gabor", "mean"
            ]
        ),
        setting_def("boundary_condition", "str", test="nearest"),
        setting_def("separable_wavelet_families", "str", to_list=True, test=["coif4", "coif5"]),
        setting_def(
            "separable_wavelet_set", "str", to_list=True, class_key="separable_wavelet_filter_set",
            test=["hhh", "lll"]
        ),
        setting_def("separable_wavelet_stationary", "bool", test=False),
        setting_def("separable_wavelet_decomposition_level", "int", to_list=True, test=[1, 2]),
        setting_def("separable_wavelet_rotation_invariance", "bool", test=False),
        setting_def("separable_wavelet_pooling_method", "str", test="mean"),
        setting_def("separable_wavelet_boundary_condition", "str", test="constant"),
        setting_def("nonseparable_wavelet_families", "str", to_list=True, test=["simoncelli", "shannon"]),
        setting_def("nonseparable_wavelet_decomposition_level", "int", to_list=True, test=[1, 2]),
        setting_def("nonseparable_wavelet_response", "str", test="magnitude"),
        setting_def("nonseparable_wavelet_boundary_condition", "str", test="constant"),
        setting_def("gaussian_sigma", "float", to_list=True, test=[1.0, 3.0]),
        setting_def("gaussian_kernel_truncate", "float", class_key="gaussian_sigma_truncate", test=10.0),
        setting_def(
            "gaussian_kernel_boundary_condition", "str", class_key="gaussian_boundary_condition", test="constant"
        ),
        setting_def(
            "laplacian_of_gaussian_sigma", "float", to_list=True,
            xml_key=["laplacian_of_gaussian_sigma", "log_sigma"], class_key="log_sigma", test=[1.0, 3.0]
        ),
        setting_def(
            "laplacian_of_gaussian_kernel_truncate", "float",
            xml_key=["laplacian_of_gaussian_kernel_truncate", "log_sigma_truncate"], class_key="log_sigma_truncate",
            test=10.0
        ),
        setting_def("laplacian_of_gaussian_pooling_method", "str", class_key="log_pooling_method", test="mean"),
        setting_def(
            "laplacian_of_gaussian_boundary_condition", "str", class_key="log_boundary_condition", test="constant"
        ),
        setting_def("laws_kernel", "str", to_list=True, test=["l5e5s5", "w5r5l3"]),
        setting_def(
            "laws_compute_energy", "bool", xml_key="laws_calculate_energy",
            class_key="laws_calculate_energy", test=True
        ),
        setting_def("laws_delta", "int", to_list=True, test=[3, 5]),
        setting_def(
            "laws_rotation_invariance", "bool", xml_key=["laws_rotation_invariance", "laws_rot_invar"], test=False
        ),
        setting_def("laws_pooling_method", "str", test="mean"),
        setting_def("laws_boundary_condition", "str", test="constant"),
        setting_def("gabor_sigma", "float", to_list=True, test=[1.0, 3.0]),
        setting_def("gabor_lambda", "float", to_list=True, test=[0.5, 2.0]),
        setting_def("gabor_gamma", "float", to_list=True, test=[0.5, 0.75]),
        setting_def("gabor_theta", "float", to_list=True, test=[5.0, 15.0]),
        setting_def("gabor_theta_step", "float", test=None),
        setting_def("gabor_response", "str", test="magnitude"),
        setting_def(
            "gabor_rotation_invariance", "bool", xml_key=["gabor_rotation_invariance", "gabor_rot_invar"], test=False
        ),
        setting_def("gabor_pooling_method", "str", test="mean"),
        setting_def("gabor_boundary_condition", "str", test="constant"),
        setting_def(
            "mean_filter_kernel_size", "int", to_list=True, xml_key=["mean_filter_kernel_size", "mean_filter_size"],
            class_key="mean_filter_size", test=[3, 7]
        ),
        setting_def("mean_filter_boundary_condition", "str", test="constant"),
        setting_def(
            "riesz_filter_order", "int", to_list=True, xml_key=["riesz_filter_order", "riesz_order"],
            class_key="riesz_order", test=[2, 1, 0]
        ),
        setting_def("riesz_filter_tensor_sigma", "float", to_list=True, test=[3.0, 5.0])
    ]
