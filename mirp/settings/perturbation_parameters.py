from typing import Any
from dataclasses import dataclass
from mirp.settings.utilities import setting_def


@dataclass
class ImagePerturbationSettingsClass:
    """
    Parameters related to image and mask perturbation / augmentation. By default images and masks are not perturbed or
    augmented.

    Parameters
    ----------

    crop_around_roi: bool, optional, default: False
        Determines whether the image may be cropped around the regions of interest. Setting
        this to True may speed up computation and save memory.

    crop_distance: float, optional, default: 150.0
        Physical distance around the mask that should be maintained when cropping the image. When using convolutional
        kernels for filtering an image, we recommend to leave some distance to prevent boundary effects. A crop
        distance of 0.0 crops the image tightly around the mask.

    perturbation_noise_repetitions: int, optional, default: 0
        Number of repetitions where noise is randomly added to the image. A value of 0 means that no noise will be
        added.

    perturbation_noise_level: float, optional, default: None
        Set the noise level in intensity units. This determines the width of the normal distribution used to generate
        random noise. If None (default), noise is determined from the image itself.

    perturbation_rotation_angles: float or list of float, optional, default: 0.0
        Angles (in degrees) over which the image and mask are rotated. This rotation is only in the x-y (axial)
        plane. Multiple angles can be provided to create images with different rotations.

    perturbation_translation_fraction: float or list of float, optional, default: 0.0
        Sub-voxel translation distance fractions of the interpolation grid. This forces the interpolation grid to
        shift slightly and interpolate at different points. Multiple values can be provided. All values should be
        between 0.0 and 1.0.

    perturbation_roi_adapt_type: {"fraction", "distance"}, optional, default: "distance"
        Determines how the mask is grown or shrunk. Can be either "fraction" or "distance". "fraction" is used to
        grow or shrink the mask by a certain fraction (see the ``perturbation_roi_adapt_size`` parameter).
        "distance" is used to grow or shrink the mask by a certain physical distance, defined using the
        ``perturbation_roi_adapt_size`` parameter.

    perturbation_roi_adapt_size: float or list of float, optional, default: 0.0
        Determines the extent of growth/shrinkage of the ROI mask. The use of this parameter depends on the
        growth/shrinkage type (``perturbation_roi_adapt_type``), For "distance", this parameter defines
        growth/shrinkage in physical units, typically mm. For "fraction", this parameter defines growth/shrinkage in
        volume fraction (e.g. a value of 0.2 grows the mask by 20%). For either type, positive values indicate growing
        the mask, whereas negative values indicate its shrinkage. Multiple values can be provided to perturb the
        volume of the mask.

    perturbation_roi_adapt_max_erosion: float, optional, default: 0.8
        Limits shrinkage of the mask by distance-based adaptations to avoid forming empty masks. Defined as fraction of
        the original volume, e.g. a value of 0.8 prevents shrinking the mask below 80% of its original volume. Only
        used when ``perturbation_roi_adapt_type=="distance"``.

    perturbation_randomise_roi_repetitions: int, optional, default: 0.0
        Number of repetitions where the mask is randomised using supervoxel-based randomisation.

    roi_split_boundary_size: float or list of float, optional, default: 0.0
        Width of the rim used for splitting the mask into bulk and rim masks, in physical dimensions. Multiple values
        can be provided to generate rims of different widths.

    roi_split_max_erosion: float, optional, default: 0.6
        Determines the minimum volume of the bulk mask when splitting the original mask into bulk and rim sections.
        Fraction of the original volume, e.g. 0.6 means that the bulk contains at least 60% of the original mask.

    **kwargs: dict, optional
        Unused keyword arguments.
    """

    def __init__(
            self,
            crop_around_roi: bool = False,
            crop_distance: float = 150.0,
            perturbation_noise_repetitions: int = 0,
            perturbation_noise_level: None | float = None,
            perturbation_rotation_angles: None | float | list[float] = 0.0,
            perturbation_translation_fraction: None | float | list[float] = 0.0,
            perturbation_roi_adapt_type: str = "distance",
            perturbation_roi_adapt_size: None | float | list[float] = 0.0,
            perturbation_roi_adapt_max_erosion: float = 0.8,
            perturbation_randomise_roi_repetitions: int = 0,
            roi_split_boundary_size: None | float | list[float] = 0.0,
            roi_split_max_erosion: float = 0.6,
            **kwargs
    ):

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
        self.noise_level: None | float = perturbation_noise_level

        # Convert perturbation_rotation_angles to list, if necessary.
        if not isinstance(perturbation_rotation_angles, list):
            perturbation_rotation_angles = [perturbation_rotation_angles]

        # Check that the rotation angles are floating points.
        if not all(isinstance(ii, float) for ii in perturbation_rotation_angles):
            raise TypeError(f"Not all values for perturbation_rotation_angles are floating point values.")

        # Set rotation_angles.
        self.rotation_angles: list[float] = perturbation_rotation_angles

        # Convert perturbation_translation_fraction to list, if necessary.
        if not isinstance(perturbation_translation_fraction, list):
            perturbation_translation_fraction = [perturbation_translation_fraction]

        # Check that the translation fractions are floating points.
        if not all(isinstance(ii, float) for ii in perturbation_translation_fraction):
            raise TypeError(f"Not all values for perturbation_translation_fraction are floating point values.")

        # Check that the translation fractions lie between 0.0 and 1.0.
        if not all(0.0 <= ii < 1.0 for ii in perturbation_translation_fraction):
            raise ValueError(
                "Not all values for perturbation_translation_fraction lie between 0.0 and 1.0, not including 1.0."
            )

        # Set translation_fraction.
        self.translation_fraction: list[float] = perturbation_translation_fraction

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
        self.roi_adapt_size: list[float] = perturbation_roi_adapt_size

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
        self.roi_boundary_size: list[float] = roi_split_boundary_size

        # Initially local variables
        self.translate_x: None | float = None
        self.translate_y: None | float = None
        self.translate_z: None | float = None


def get_perturbation_settings() -> list[dict[str, Any]]:
    return [
        setting_def("crop_around_roi", "bool", xml_key=["crop_around_roi", "resect"], test=True),
        setting_def("crop_distance", "float", test=10.0),
        setting_def(
            "perturbation_noise_repetitions", "int", xml_key="noise_repetitions",
            class_key="noise_repetitions", test=10
        ),
        setting_def(
            "perturbation_noise_level", "float", xml_key="noise_level", class_key="noise_level", test=0.75
        ),
        setting_def(
            "perturbation_rotation_angles", "float", to_list=True, xml_key=["rotation_angles", "rot_angles"],
            class_key="rotation_angles", test=[-33.0, 33.0]
        ),
        setting_def(
            "perturbation_translation_fraction", "float", to_list=True,
            xml_key=["translation_fraction", "translate_frac"], class_key="translation_fraction", test=[0.25, 0.75]
        ),
        setting_def(
            "perturbation_roi_adapt_type", "str", xml_key="roi_adapt_type", class_key="roi_adapt_type",
            test="fraction"
        ),
        setting_def(
            "perturbation_roi_adapt_size", "float", to_list=True, xml_key="roi_adapt_size",
            class_key="roi_adapt_size", test=[0.8, 1.0, 1.2]
        ),
        setting_def(
            "perturbation_roi_adapt_max_erosion", "float", xml_key=["roi_adapt_max_erosion", "eroded_vol_fract"],
            class_key="max_volume_erosion", test=0.2
        ),
        setting_def(
            "perturbation_randomise_roi_repetitions", "int", xml_key="roi_random_rep",
            class_key="roi_random_rep", test=100
        ),
        setting_def(
            "roi_split_boundary_size", "float", to_list=True, xml_key="roi_boundary_size",
            class_key="roi_boundary_size", test=[2.0, 5.0]
        ),
        setting_def(
            "roi_split_max_erosion", "float", xml_key=["roi_split_max_erosion", "bulk_min_vol_fract"],
            class_key="max_bulk_volume_erosion", test=0.2
        )
    ]

