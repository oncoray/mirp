from typing import Union, List, Iterable


class ImageInterpolationSettingsClass:
    """
    Set of parameters related to image interpolating.

    Parameters
    ----------
    by_slice: str or bool, optional, default: False
        Defines whether calculations should be performed in 2D (True) or 3D (False), or alternatively only in the
        largest slice ("largest"). See :class:`~mirp.settings.settingsGeneral.GeneralSettingsClass`.

    interpolate: bool, optional, default: False
        Controls whether interpolation of images to a common grid is performed at all.

    spline_order: int, optional, default: 3
        Sets the spline order used for spline interpolation. mirp uses `scipy.ndimage.map_coordinates
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage
        .map_coordinates>`_ internally. Spline orders 0, 1, and 3 refer to nearest neighbour, linear interpolation
        and cubic interpolation, respectively.

    new_spacing: float or list of float or list of list of float:
        Sets voxel spacing after interpolation. A single value represents the spacing that will be applied in all
        directions. Non-uniform voxel spacing may also be provided, but requires 3 values for z, y, and x directions
        (if `by_slice = False`) or 2 values for y and x directions (otherwise).

        Multiple spacings may be defined by creating a nested list, e.g. [[1.0], [1.5], [2.0]] to resample the
        same image multiple times to different (here: isotropic) voxel spacings, namely 1.0, 1.5 and 2.0. Units
        are defined by the headers of the image files. These are typically millimeters for radiological images.

    anti_aliasing: bool, optional, default: true
        Determines whether to perform anti-aliasing, which is done to mitigate aliasing artifacts when downsampling.

    smoothing_beta: float, optional, default: 0.98
        Determines the smoothness of the Gaussian filter used for anti-aliasing. A value of 1.00 equates to no
        anti-aliasing, with lower values producing increasingly smooth imaging. Values above 0.90 are recommended.

    **kwargs: dict, optional
        Unused keyword arguments.
    """

    def __init__(
            self,
            by_slice: bool,
            interpolate: bool = False,
            spline_order: int = 3,
            new_spacing: Union[float, int, List[int], List[float], None] = None,
            anti_aliasing: bool = True,
            smoothing_beta: float = 0.98,
            **kwargs
    ):

        # Set interpolate parameter.
        self.interpolate: bool = interpolate

        # Check if the spline order is valid.
        if spline_order < 0 or spline_order > 5:
            raise ValueError(
                f"The interpolation spline order should be an integer between 0 and 5. Found: {spline_order}")

        # Set spline order for the interpolating spline.
        self.spline_order: int = spline_order

        # Check
        if not interpolate:
            new_spacing = None

        else:
            # When interpolation is desired, check that the desired spacing is set.
            if new_spacing is None:
                raise ValueError(
                    "The desired voxel spacing for resampling is required if interpolation=True. "
                    "However, no new spacing was defined.")

            # Parse value to list of floating point values to facilitate checks.
            if isinstance(new_spacing, (int, float)):
                new_spacing = [new_spacing]

            # Check if nested list elements are present.
            if any(isinstance(ii, Iterable) for ii in new_spacing):
                new_spacing = [
                    self._check_new_sample_spacing(by_slice=by_slice, new_spacing=new_spacing_element)
                    for new_spacing_element in new_spacing
                ]

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
                raise ValueError(
                    f"The value of the smoothing_beta parameter should lie between 0.0 and 1.0, "
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
                raise ValueError(
                    f"The desired voxel spacing for in-slice resampling should consist of two "
                    f"elements. Found: {len(new_spacing)} elements.")
        else:
            if len(new_spacing) == 1:
                # This creates isotropic spacing.
                new_spacing = [new_spacing[0], new_spacing[0], new_spacing[0]]

            elif len(new_spacing) == 3:
                # Do nothing.
                pass

            else:
                raise ValueError(
                    f"The desired voxel spacing for volumetric resampling should consist of three "
                    f"elements. Found: {len(new_spacing)} elements.")

        return new_spacing


class MaskInterpolationSettingsClass:
    """
    Set of parameters related to mask interpolation. mirp registers the mask to an interpolated image. Therefore,
    parameters such as `new_spacing` are missing.

    Parameters
    ----------
    roi_spline_order: int, optional, default: 1
        Sets the spline order used for spline interpolation. mirp uses `scipy.ndimage.map_coordinates
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage
        .map_coordinates>`_ internally. Spline orders 0, 1, and 3 refer to nearest neighbour, linear interpolation
        and cubic interpolation, respectively.

    roi_interpolation_mask_inclusion_threshold: float, optional, default: 0.5
        Threshold for partially masked voxels after interpolation. All voxels with a value equal to or greater than
        this threshold are assigned to the mask.

    **kwargs: dict, optional
        Unused keyword arguments.
    """

    def __init__(
            self,
            roi_spline_order: int = 1,
            roi_interpolation_mask_inclusion_threshold: float = 0.5,
            **kwargs):


        # Check if the spline order is valid.
        if roi_spline_order < 0 or roi_spline_order > 5:
            raise ValueError(
                f"The interpolation spline order for the ROI should be an integer between 0 and 5. Found:"
                f" {roi_spline_order}")

        # Set spline order.
        self.spline_order = roi_spline_order

        # Check if the inclusion threshold is between 0 and 1.
        if roi_interpolation_mask_inclusion_threshold <= 0.0 or roi_interpolation_mask_inclusion_threshold > 1.0:
            raise ValueError(
                f"The inclusion threshold for the ROI mask should be between 0.0 and 1.0, excluding 0.0. "
                f"Found: {roi_interpolation_mask_inclusion_threshold}")

        self.incl_threshold = roi_interpolation_mask_inclusion_threshold
