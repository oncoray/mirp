from typing import Union, List


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
