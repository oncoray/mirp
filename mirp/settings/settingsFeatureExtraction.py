from typing import Union, List


class FeatureExtractionSettingsClass:

    def __init__(
            self,
            by_slice: bool,
            no_approximation: bool,
            ibsi_compliant: bool = True,
            base_feature_families: Union[None, str, List[str]] = "all",
            base_discretisation_method: Union[None, str, List[str]] = None,
            base_discretisation_n_bins: Union[None, int, List[int]] = None,
            base_discretisation_bin_width: Union[None, float, List[float]] = None,
            ivh_discretisation_method: str = "none",
            ivh_discretisation_n_bins: Union[None, int] = 1000,
            ivh_discretisation_bin_width: Union[None, float] = None,
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
        valid_families: List[bool] = [ii in self.get_available_families() for ii in base_feature_families]

        if not all(valid_families):
            raise ValueError(
                f"One or more families in the base_feature_families parameter were not recognised: "
                f"{', '.join([base_feature_families[ii] for ii, is_valid in enumerate(valid_families) if not is_valid])}")

        # Set families.
        self.families: List[str] = base_feature_families

        if not self.has_any_feature_family():
            self.families = ["none"]

        if self.has_discretised_family():
            # Check if discretisation_method is None.
            if base_discretisation_method is None:
                raise ValueError("The base_discretisation_method parameter has no default and must be set.")

            if not isinstance(base_discretisation_method, list):
                base_discretisation_method = [base_discretisation_method]

            if not all(discretisation_method in ["fixed_bin_size", "fixed_bin_number", "none"] for
                       discretisation_method in base_discretisation_method):
                raise ValueError(
                    "Available values for the base_discretisation_method parameter are 'fixed_bin_size', "
                    "'fixed_bin_number', and 'none'. One or more values were not recognised.")

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
            if "fixed_bin_size" in base_discretisation_method:
                if base_discretisation_bin_width is None:
                    raise ValueError("The base_discretisation_bin_width parameter has no default and must be set")

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
        self.discretisation_method: Union[None, List[str]] = base_discretisation_method
        self.discretisation_n_bins: Union[None, List[int]] = base_discretisation_n_bins
        self.discretisation_bin_width: Union[None, List[float]] = base_discretisation_bin_width

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
        self.ivh_discretisation_method: Union[None, str] = ivh_discretisation_method
        self.ivh_discretisation_n_bins: Union[None, int] = ivh_discretisation_n_bins
        self.ivh_discretisation_bin_width: Union[None, float] = ivh_discretisation_bin_width

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

        self.glcm_distance: Union[None, List[float]] = glcm_distance
        self.glcm_spatial_method: Union[None, List[str]] = glcm_spatial_method

        # Set GLRLM attributes.
        if self.has_glrlm_family():
            # Check spatial method.
            glrlm_spatial_method = self.check_valid_directional_spatial_method(
                glrlm_spatial_method, "glrlm_spatial_method")

        else:
            glrlm_spatial_method = None

        self.glrlm_spatial_method: Union[None, List[str]] = glrlm_spatial_method

        # Set GLSZM attributes.
        if self.has_glszm_family():
            # Check spatial method.
            glszm_spatial_method = self.check_valid_omnidirectional_spatial_method(
                glszm_spatial_method, "glszm_spatial_method")
        else:
            glszm_spatial_method = None

        self.glszm_spatial_method: Union[None, List[str]] = glszm_spatial_method

        # Set GLDZM attributes.
        if self.has_gldzm_family():
            # Check spatial method.
            gldzm_spatial_method = self.check_valid_omnidirectional_spatial_method(
                gldzm_spatial_method, "gldzm_spatial_method")

        else:
            gldzm_spatial_method = None

        self.gldzm_spatial_method: Union[None, List[str]] = gldzm_spatial_method

        # Set NGTDM attributes.
        if self.has_ngtdm_family():
            # Check spatial method
            ngtdm_spatial_method = self.check_valid_omnidirectional_spatial_method(
                ngtdm_spatial_method, "ngtdm_spatial_method")

        else:
            ngtdm_spatial_method = None

        self.ngtdm_spatial_method: Union[None, List[str]] = ngtdm_spatial_method

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

        self.ngldm_dist: Union[None, List[float]] = ngldm_distance
        self.ngldm_diff_lvl: Union[None, List[float]] = ngldm_difference_level
        self.ngldm_spatial_method: Union[None, List[str]] = ngldm_spatial_method

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
