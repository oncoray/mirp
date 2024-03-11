from typing import Any
from dataclasses import dataclass
from mirp.settings.utilities import setting_def


@dataclass
class FeatureExtractionSettingsClass:
    """
    Parameters related to feature computation. Many are conditional on the type of features that will be computed (
    ``base_feature_families``).

    Parameters
    ----------
    by_slice: str or bool, optional, default: False
        Defines whether calculations should be performed in 2D (True) or 3D (False), or alternatively only in the
        largest slice ("largest"). See :class:`~mirp.settings.settingsGeneral.GeneralSettingsClass`.

    no_approximation: bool, optional, default: False
        Disables approximation of features, such as Geary's c-measure. Can be True or False (default). See
        :class:`~mirp.settings.settingsGeneral.GeneralSettingsClass`.

    ibsi_compliant: bool, optional, default: True
        Limits computation of features to those features that have a reference value in the IBSI reference standard.

    base_feature_families: str or list of str, optional, default: "none"
        Determines the feature families for which features are computed. Radiomics features are implemented as
        defined in the IBSI reference manual. The following feature families are currently present, and can be added
        using the following tags:

        * Morphological features: "mrp", "morph", "morphology", and "morphological".
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

        A list of strings may be provided to select multiple feature families.

        .. note::
            Even though ``"none"`` is the internal default, the :func:`~mirp.extractFeaturesAndImages.extract_features`
            function overrides this, and sets the default to ``"all"``.

    base_discretisation_method: {"fixed_bin_number", "fixed_bin_size", "fixed_bin_size_pyradiomics", "none"}
        Method used for discretising intensities. Used to compute intensity histogram as well as texture features.
        The setting is ignored if none of these feature families are being computed. The following options are
        available:

        * "fixed_bin_number": The intensity range within the mask is divided into a fixed number of bins,
          defined by the ``base_discretisation_bin_width`` parameter.
        * "fixed_bin_size": The intensity range is divided into bins with a fixed width, defined using the
          ``base_discretisation_bin_width`` parameter. The lower bound of the range is determined from the lower
          bound of the mask resegmentation range, see the ``resegmentation_intensity_range`` in
          :class:`~mirp.settings.settingsMaskResegmentation.ResegmentationSettingsClass`. CT images have a default
          lower bound of the initial bin at -1000.0 and PET images have a default lower bound at 0.0. Other images,
          including MRI, normalised CT and PET images and filtered images, do not have a default value.
        * "fixed_bin_size_pyradiomics": The intensity range is divided into bins with a fixed width. This follows the
          non-IBSI compliant implementation in the pyradiomics package.
        * "none": The intensity range is not discretised into bins. This method can only be used if the image
          intensities are integer and strictly positive.

         There is no default method. Multiple methods can be specified as a list to yield features according to each
         method.

        .. warning::
            The "fixed_bin_size_pyradiomics" is not IBSI compliant, and should only be used when
            reproducing results from studies that used pyradiomics.

    base_discretisation_n_bins: int or list of int
        Number of bins used for the "fixed_bin_number" discretisation method. No default value. Multiple values can
        be specified in a list to yield features according to each number of bins.

    base_discretisation_bin_width: float or list of float
        Width of each bin in the "fixed_bin_size" discretisation method. No default value. Multiple values can be
        specified in a list to yield features according to each bin width.

    ivh_discretisation_method: {"fixed_bin_number", "fixed_bin_size", "none"}, optional, default: "none"
        Method used for discretising intensities for computing intensity-volume histograms. The discretisation
        methods follow those in ``base_discretisation_method``. The "none" method changes to "fixed_bin_number" if
        the underlying data are not suitable.

    ivh_discretisation_n_bins: int, optional, default: 1000
        Number of bins used for the "fixed_bin_number" discretisation method.

    ivh_discretisation_bin_width: float, optional
        Width of each bin in the "fixed_bin_size" discretisation method. No default value.

    glcm_distance: float or list of float, optional, default: 1.0
        Distance (in voxels) for GLCM for determining the neighbourhood. Chebyshev, or checkerboard, distance is
        used. A value of 1.0 will therefore consider all (diagonally) adjacent voxels as its neighbourhood. A list of
        values can be provided to compute GLCM features at different scales.

    glcm_spatial_method: {"2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge", "3d_average", "3d_volume_merge"}, optional
        Determines how co-occurrence matrices are formed and aggregated. One of the following:

        * "2d_average": features are computed from all matrices then averaged [IBSI:BTW3].
        * "2d_slice_merge": matrices in the same slice are merged, features computed and then averaged [IBSI:SUJT].
        * "2.5d_direction_merge": matrices for the same direction are merged, features computed and then averaged
          [IBSI:JJUI].
        * "2.5d_volume_merge": all matrices are merged and a single feature is calculated [IBSI:ZW7Z].
        * "3d_average": features are computed from all matrices then averaged [IBSI:ITBB].
        * "3d_volume_merge": all matrices are merged and a single feature is computed from the merged matrix
          [IBSI:IAZD].

        A list of values may be provided to extract features for multiple spatial methods. Default: "2d_slice_merge"
        (``by_slice = False``) or "3d_volume_merge" (``by_slice = True``).

    glrlm_spatial_method: {"2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge", "3d_average", "3d_volume_merge"}, optional
        Determines how run length matrices are formed and aggregated. One of the following:

        * "2d_average": features are calculated from all matrices then averaged [IBSI:BTW3].
        * "2d_slice_merge": matrices in the same slice are merged, features computed and then averaged [IBSI:SUJT].
        * "2.5d_direction_merge": matrices for the same direction are merged, features computed and then averaged
          [IBSI:JJUI].
        * "2.5d_volume_merge": all matrices are merged and a single feature is computed [IBSI:ZW7Z].
        * "3d_average": features are computed from all matrices then averaged [IBSI:ITBB].
        * "3d_volume_merge": all matrices are merged and a single feature is computed from the merged matrix
          [IBSI:IAZD].

        A list of values may be provided to extract features for multiple spatial methods. Default:
        "2d_slice_merge" (``by_slice = False``) or "3d_volume_merge" (``by_slice = True``).

    glszm_spatial_method: {"2d", "2.5d", "3d"}, optional
        Determines how the size zone matrices are formed and aggregated. One of the following:

        * "2d": features are computed from individual matrices and subsequently averaged [IBSI:8QNN].
        * "2.5d": all 2D matrices are merged and features are computed from this single matrix [IBSI:62GR].
        * "3d": features are computed from a single 3D matrix [IBSI:KOBO].

        A list of values may be provided to extract features for multiple spatial methods. Default: "2d"
        (``by_slice = False``) or "3d" (``by_slice = True``).

    gldzm_spatial_method: {"2d", "2.5d", "3d"}, optional
        Determines how the distance zone matrices are formed and aggregated. One of the following:

        * "2d": features are computed from individual matrices and subsequently averaged [IBSI:8QNN].
        * "2.5d": all 2D matrices are merged and features are computed from this single matrix [IBSI:62GR].
        * "3d": features are computed from a single 3D matrix [IBSI:KOBO].

        A list of values may be provided to extract features for multiple spatial methods. Default: "2d"
        (``by_slice = False``) or "3d" (``by_slice = True``).

    ngtdm_spatial_method: {"2d", "2.5d", "3d"}, optional
        Determines how the neighbourhood grey tone difference matrices are formed and aggregated. One of the
        following:

        * "2d": features are computed from individual matrices and subsequently averaged [IBSI:8QNN].
        * "2.5d": all 2D matrices are merged and features are computed from this single matrix [IBSI:62GR].
        * "3d": features are computed from a single 3D matrix [IBSI:KOBO].

        A list of values may be provided to extract features for multiple spatial methods. Default: "2d"
        (``by_slice = False``) or "3d" (``by_slice = True``).

    ngldm_distance: float or list of float, optional, default: 1.0
        Distance (in voxels) for NGLDM for determining the neighbourhood. Chebyshev, or checkerboard, distance is
        used. A value of 1.0 will therefore consider all (diagonally) adjacent voxels as its neighbourhood. A list of
        values can be provided to compute NGLDM features at different scales.

    ngldm_difference_level: float or list of float, optional, default: 0.0
        Difference level (alpha) for NGLDM. Determines which bins are grouped together in the matrix.

    ngldm_spatial_method: {"2d", "2.5d", "3d"}, optional
        Determines how the neighbourhood grey level dependence matrices are formed and aggregated. One of the
        following:

        * "2d": features are computed from individual matrices and subsequently averaged [IBSI:8QNN].
        * "2.5d": all 2D matrices are merged and features are computed from this single matrix [IBSI:62GR].
        * "3d": features are computed from a single 3D matrix [IBSI:KOBO].

        A list of values may be provided to extract features for multiple spatial methods. Default: "2d"
        (``by_slice = False``) or "3d" (``by_slice = True``).

    **kwargs: dict, optional
        Unused keyword arguments.
    """

    def __init__(
            self,
            by_slice: bool = False,
            no_approximation: bool = False,
            ibsi_compliant: bool = True,
            base_feature_families: None | str | list[str] = "none",
            base_discretisation_method: None | str | list[str] = None,
            base_discretisation_n_bins: None | int | list[int] = None,
            base_discretisation_bin_width: None | float | list[float] = None,
            ivh_discretisation_method: str = "none",
            ivh_discretisation_n_bins: None | int = 1000,
            ivh_discretisation_bin_width: None | float = None,
            glcm_distance: float | list[float] = 1.0,
            glcm_spatial_method: None | str | list[str] = None,
            glrlm_spatial_method: None | str | list[str] = None,
            glszm_spatial_method: None | str | list[str] = None,
            gldzm_spatial_method: None | str | list[str] = None,
            ngtdm_spatial_method: None | str | list[str] = None,
            ngldm_distance: float | list[float] = 1.0,
            ngldm_difference_level: float | list[float] = 0.0,
            ngldm_spatial_method: None | str | list[str] = None,
            **kwargs
    ):
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
        valid_families: list[bool] = [ii in self.get_available_families() for ii in base_feature_families]

        if not all(valid_families):
            raise ValueError(
                f"One or more families in the base_feature_families parameter were not recognised: "
                f"{', '.join([base_feature_families[ii] for ii, is_valid in enumerate(valid_families) if not is_valid])}")

        # Set families.
        self.families: list[str] = base_feature_families

        if not self.has_any_feature_family():
            self.families = ["none"]

        if self.has_discretised_family():
            # Check if discretisation_method is None.
            if base_discretisation_method is None:
                raise ValueError("The base_discretisation_method parameter has no default and must be set.")

            if not isinstance(base_discretisation_method, list):
                base_discretisation_method = [base_discretisation_method]

            if not all(discretisation_method in [
                "fixed_bin_size", "fixed_bin_number", "fixed_bin_size_pyradiomics", "none"
            ] for discretisation_method in base_discretisation_method):
                raise ValueError(
                    "Available values for the base_discretisation_method parameter are "
                    "'fixed_bin_number', 'fixed_bin_size', 'fixed_bin_size_pyradiomics' and 'none'. "
                    "One or more values were not recognised.")

            # Check discretisation_n_bins
            if "fixed_bin_number" in base_discretisation_method:
                if base_discretisation_n_bins is None:
                    raise ValueError("The base_discretisation_n_bins parameter has no default and must be set")

                if not isinstance(base_discretisation_n_bins, list):
                    base_discretisation_n_bins = [base_discretisation_n_bins]

                if not all(isinstance(n_bins, int) for n_bins in base_discretisation_n_bins):
                    raise TypeError(
                        "The base_discretisation_n_bins parameter is expected to contain integers with "
                        "value 2 or larger. Found one or more values that were not integers.")

                if not all(n_bins >= 2 for n_bins in base_discretisation_n_bins):
                    raise ValueError(
                        "The base_discretisation_n_bins parameter is expected to contain integers with "
                        "value 2 or larger. Found one or more values that were less than 2.")

            else:
                base_discretisation_n_bins = None

            # Check discretisation_bin_width
            if "fixed_bin_size" in base_discretisation_method or "fixed_bin_size_pyradiomics" in base_discretisation_method:
                if base_discretisation_bin_width is None:
                    raise ValueError(
                        "The base_discretisation_bin_width parameter has no default value and must be set.")

                if not isinstance(base_discretisation_bin_width, list):
                    base_discretisation_bin_width = [base_discretisation_bin_width]

                if not all(isinstance(bin_size, float) for bin_size in base_discretisation_bin_width):
                    raise TypeError(
                        "The base_discretisation_bin_width parameter is expected to contain floating "
                        "point values greater than 0.0. Found one or more values that were not floating "
                        "points.")

                if not all(bin_size > 0.0 for bin_size in base_discretisation_bin_width):
                    raise ValueError(
                        "The base_discretisation_bin_width parameter is expected to contain floating "
                        "point values greater than 0.0. Found one or more values that were 0.0 or less.")

            else:
                base_discretisation_bin_width = None

        else:
            base_discretisation_method = None
            base_discretisation_n_bins = None
            base_discretisation_bin_width = None

        # Set discretisation method-related parameters.
        self.discretisation_method: None | list[str] = base_discretisation_method
        self.discretisation_n_bins: None | list[int] = base_discretisation_n_bins
        self.discretisation_bin_width: None | list[float] = base_discretisation_bin_width

        if self.has_ivh_family():
            if ivh_discretisation_method not in ["fixed_bin_size", "fixed_bin_number", "none"]:
                raise ValueError(
                    "Available values for the ivh_discretisation_method parameter are 'fixed_bin_size', "
                    "'fixed_bin_number', and 'none'. One or more values were not recognised.")

            # Check discretisation_n_bins
            if "fixed_bin_number" in ivh_discretisation_method:

                if not isinstance(ivh_discretisation_n_bins, int):
                    raise TypeError(
                        "The ivh_discretisation_n_bins parameter is expected to be an integer with "
                        "value 2 or greater. Found: a value that was not an integer.")

                if not ivh_discretisation_n_bins >= 2:
                    raise ValueError(
                        "The ivh_discretisation_n_bins parameter is expected to be an integer with "
                        f"value 2 or greater. Found: {ivh_discretisation_n_bins}")

            else:
                ivh_discretisation_n_bins = None

            # Check discretisation_bin_width
            if "fixed_bin_size" in ivh_discretisation_method:

                if not isinstance(ivh_discretisation_bin_width, float):
                    raise TypeError(
                        "The ivh_discretisation_bin_width parameter is expected to be a floating "
                        "point value greater than 0.0. Found a value that was not a floating point.")

                if not ivh_discretisation_bin_width > 0.0:
                    raise ValueError(
                        "The ivh_discretisation_bin_width parameter is expected to  be a floating "
                        f"point value greater than 0.0. Found: {ivh_discretisation_bin_width}")

            else:
                ivh_discretisation_bin_width = None

        else:
            ivh_discretisation_method = None
            ivh_discretisation_n_bins = None
            ivh_discretisation_bin_width = None

        # Set parameters
        self.ivh_discretisation_method: None | str = ivh_discretisation_method
        self.ivh_discretisation_n_bins: None | int = ivh_discretisation_n_bins
        self.ivh_discretisation_bin_width: None | float = ivh_discretisation_bin_width

        # Set GLCM attributes.
        if self.has_glcm_family():
            # Check distance parameter.
            if not isinstance(glcm_distance, list):
                glcm_distance = [glcm_distance]

            if not all(isinstance(distance, float) for distance in glcm_distance):
                raise TypeError(
                    "The glcm_distance parameter is expected to contain floating point values of 1.0 "
                    "or greater. Found one or more values that were not floating points.")

            if not all(distance >= 1.0 for distance in glcm_distance):
                raise ValueError(
                    "The glcm_distance parameter is expected to contain floating point values of 1.0 "
                    "or greater. Found one or more values that were less than 1.0.")

            # Check spatial method.
            glcm_spatial_method = self.check_valid_directional_spatial_method(
                glcm_spatial_method,
                "glcm_spatial_method")

        else:
            glcm_distance = None
            glcm_spatial_method = None

        self.glcm_distance: None | list[float] = glcm_distance
        self.glcm_spatial_method: None | list[str] = glcm_spatial_method

        # Set GLRLM attributes.
        if self.has_glrlm_family():
            # Check spatial method.
            glrlm_spatial_method = self.check_valid_directional_spatial_method(
                glrlm_spatial_method, "glrlm_spatial_method")

        else:
            glrlm_spatial_method = None

        self.glrlm_spatial_method: None | list[str] = glrlm_spatial_method

        # Set GLSZM attributes.
        if self.has_glszm_family():
            # Check spatial method.
            glszm_spatial_method = self.check_valid_omnidirectional_spatial_method(
                glszm_spatial_method, "glszm_spatial_method")
        else:
            glszm_spatial_method = None

        self.glszm_spatial_method: None | list[str] = glszm_spatial_method

        # Set GLDZM attributes.
        if self.has_gldzm_family():
            # Check spatial method.
            gldzm_spatial_method = self.check_valid_omnidirectional_spatial_method(
                gldzm_spatial_method, "gldzm_spatial_method")

        else:
            gldzm_spatial_method = None

        self.gldzm_spatial_method: None | list[str] = gldzm_spatial_method

        # Set NGTDM attributes.
        if self.has_ngtdm_family():
            # Check spatial method
            ngtdm_spatial_method = self.check_valid_omnidirectional_spatial_method(
                ngtdm_spatial_method, "ngtdm_spatial_method")

        else:
            ngtdm_spatial_method = None

        self.ngtdm_spatial_method: None | list[str] = ngtdm_spatial_method

        # Set NGLDM attributes
        if self.has_ngldm_family():

            # Check distance.
            if not isinstance(ngldm_distance, list):
                ngldm_distance = [ngldm_distance]

            if not all(isinstance(distance, float) for distance in ngldm_distance):
                raise TypeError(
                    "The ngldm_distance parameter is expected to contain floating point values of 1.0 "
                    "or greater. Found one or more values that were not floating points.")

            if not all(distance >= 1.0 for distance in ngldm_distance):
                raise ValueError(
                    "The ngldm_distance parameter is expected to contain floating point values of 1.0 "
                    "or greater. Found one or more values that were less than 1.0.")

            # Check spatial method
            ngldm_spatial_method = self.check_valid_omnidirectional_spatial_method(
                ngldm_spatial_method, "ngldm_spatial_method")

            # Check difference level.
            if not isinstance(ngldm_difference_level, list):
                ngldm_difference_level = [ngldm_difference_level]

            if not all(isinstance(difference, float) for difference in ngldm_difference_level):
                raise TypeError(
                    "The ngldm_difference_level parameter is expected to contain floating point values of 0.0 "
                    "or greater. Found one or more values that were not floating points.")

            if not all(difference >= 0.0 for difference in ngldm_difference_level):
                raise ValueError(
                    "The ngldm_difference_level parameter is expected to contain floating point values "
                    "of 0.0 or greater. Found one or more values that were less than 0.0.")

        else:
            ngldm_spatial_method = None
            ngldm_distance = None
            ngldm_difference_level = None

        self.ngldm_dist: None | list[float] = ngldm_distance
        self.ngldm_diff_lvl: None | list[float] = ngldm_difference_level
        self.ngldm_spatial_method: None | list[str] = ngldm_spatial_method

    @staticmethod
    def get_available_families():
        return [
            "mrp", "morph", "morphology", "morphological", "li", "loc.int", "loc_int", "local_int", "local_intensity",
            "st", "stat", "stats", "statistics", "statistical", "ih", "int_hist", "int_histogram", "intensity_histogram",
            "ivh", "int_vol_hist", "intensity_volume_histogram", "cm", "glcm", "grey_level_cooccurrence_matrix",
            "cooccurrence_matrix", "rlm", "glrlm", "grey_level_run_length_matrix", "run_length_matrix",
            "szm", "glszm", "grey_level_size_zone_matrix", "size_zone_matrix", "dzm", "gldzm",
            "grey_level_distance_zone_matrix", "distance_zone_matrix", "tdm", "ngtdm",
            "neighbourhood_grey_tone_difference_matrix", "grey_tone_difference_matrix", "ldm", "ngldm",
            "neighbouring_grey_level_dependence_matrix", "grey_level_dependence_matrix", "all", "none"
        ]

    def has_any_feature_family(self):
        return not any(family == "none" for family in self.families)

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
            raise TypeError(
                f"The {var_name} parameter expects one or more of the following values: "
                f"{', '.join(all_spatial_method)}. Found: one or more values that were not strings.")

        # Check spatial method.
        valid_spatial_method = [spatial_method in all_spatial_method for spatial_method in x]

        if not all(valid_spatial_method):
            raise ValueError(
                f"The {var_name} parameter expects one or more of the following values: "
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
            raise TypeError(
                f"The {var_name} parameter expects one or more of the following values: "
                f"{', '.join(all_spatial_method)}. Found: one or more values that were not strings.")

        # Check spatial method.
        valid_spatial_method = [spatial_method in all_spatial_method for spatial_method in x]

        if not all(valid_spatial_method):
            raise ValueError(
                f"The {var_name} parameter expects one or more of the following values: "
                f"{', '.join(all_spatial_method)}. Found: "
                f"{', '.join([spatial_method for spatial_method in x if spatial_method in all_spatial_method])}")

        return x


def get_feature_extraction_settings() -> list[dict[str, Any]]:
    return [
        setting_def("ibsi_compliant", "bool", test=True),
        setting_def(
            "base_feature_families", "str", to_list=True, xml_key=["feature_families", "families"],
            class_key="families", test=["all"]
        ),
        setting_def(
            "base_discretisation_method", "str", to_list=True, xml_key=["discretisation_method", "discr_method"],
            class_key="discretisation_method", test=["fixed_bin_size", "fixed_bin_number"]
        ),
        setting_def(
            "base_discretisation_n_bins", "int", to_list=True, xml_key=["discretisation_n_bins", "discr_n_bins"],
            class_key="discretisation_n_bins", test=[10, 33]
        ),
        setting_def(
            "base_discretisation_bin_width", "float", to_list=True,
            xml_key=["discretisation_bin_width", "discr_bin_width"], class_key="discretisation_bin_width",
            test=[10.0, 34.0]
        ),
        setting_def(
            "ivh_discretisation_method", "str", xml_key=["ivh_discretisation_method", "ivh_discr_method"],
            class_key="ivh_discretisation_method", test="fixed_bin_size"
        ),
        setting_def(
            "ivh_discretisation_n_bins", "int", xml_key=["ivh_discretisation_n_bins", "ivh_discr_n_bins"],
            test=20
        ),
        setting_def(
            "ivh_discretisation_bin_width", "float", xml_key=["ivh_discretisation_bin_width", "ivh_discr_bin_width"],
            test=30.0
        ),
        setting_def("glcm_distance", "float", to_list=True, xml_key=["glcm_distance", "glcm_dist"], test=[2.0, 3.0]),
        setting_def("glcm_spatial_method", "str", to_list=True, test=["2d_average", "2d_slice_merge"]),
        setting_def("glrlm_spatial_method", "str", to_list=True, test=["2d_average", "2d_slice_merge"]),
        setting_def("glszm_spatial_method", "str", to_list=True, test=["2d", "2.5d"]),
        setting_def("gldzm_spatial_method", "str", to_list=True, test=["2d", "2.5d"]),
        setting_def("ngtdm_spatial_method", "str", to_list=True, test=["2d", "2.5d"]),
        setting_def(
            "ngldm_distance", "float", to_list=True, xml_key=["ngldm_distance", "ngldm_dist"],
            class_key="ngldm_dist", test=[2.5, 3.5]
        ),
        setting_def(
            "ngldm_difference_level", "float", to_list=True, xml_key=["ngldm_difference_level", "ngldm_diff_lvl"],
            class_key="ngldm_diff_lvl", test=[1.0, 1.9]
        ),
        setting_def("ngldm_spatial_method", "str", to_list=True, test=["2d", "2.5d"])
    ]
