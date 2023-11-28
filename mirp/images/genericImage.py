import copy
import numpy as np
from typing import Any, Self

import pandas as pd

from mirp.images.baseImage import BaseImage
from mirp.settings.settingsGeneric import SettingsClass


class GenericImage(BaseImage):

    def __init__(
            self,
            image_data: None | np.ndarray,
            separate_slices: None | bool = None,
            translation: None | tuple[float, ...] = None,
            rotation_angle: None | float = None,
            noise_iteration_id: None | int = None,
            noise_level: None | float = None,
            interpolated: bool = False,
            interpolation_algorithm: None | str = None,
            discretisation_method: None | str = None,
            discretisation_bin_number: None | int = None,
            discretisation_bin_width: None | float = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Image data. Note that image_data is explicitly defined as this prevents IDE warnings. However,
        # the attribute itself is set using set_voxel_grid. Also, note that we explicitly copy image_data because
        # otherwise we may end up changing objects by reference, which is not the expected behaviour.
        self.image_data = None
        self.set_voxel_grid(copy.deepcopy(image_data)) if image_data is not None else None

        # Determines whether slices in the stack should be treated separately.
        self.separate_slices = separate_slices

        # Perturbation-related settings that are set during interpolate.
        self.translation = translation
        self.rotation_angle = rotation_angle
        self.noise_iteration_id = noise_iteration_id
        self.noise_level = noise_level

        # Interpolation-related settings
        self.interpolated = interpolated
        self.interpolation_algorithm = interpolation_algorithm

        # Discretisation-related settings
        self.discretisation_method = discretisation_method
        self.discretisation_bin_number = discretisation_bin_number
        self.discretisation_bin_width = discretisation_bin_width

        # Slice identifiers.
        self.slice_id: None | int = None

    def copy(self, drop_image=False) -> Self:
        image = copy.deepcopy(self)

        if drop_image:
            image.drop_image()

        return image

    def update_from_template(self, template):
        if not isinstance(template, GenericImage):
            raise TypeError(
                f"The new class object should inherit from the template provided by the \"image\" object, which is "
                f"expected to inherit from GenericImage. Found: {type(template)}")

        # NOTE: image_data is not set automatically.

        # Attributes from BaseImage
        self.modality = template.modality
        self.image_origin = copy.deepcopy(template.image_origin)
        self.image_orientation = copy.deepcopy(template.image_orientation)
        self.image_spacing = copy.deepcopy(template.image_spacing)
        self.image_dimension = copy.deepcopy(template.image_dimension)
        self.sample_name = copy.deepcopy(template.sample_name)

        # Attributes from GenericImage
        self.separate_slices = template.separate_slices
        self.translation = copy.deepcopy(template.translation)
        self.rotation_angle = copy.deepcopy(template.rotation_angle)
        self.noise_iteration_id = template.noise_iteration_id
        self.noise_level = template.noise_level
        self.interpolated = template.interpolated
        self.interpolation_algorithm = template.interpolation_algorithm
        self.discretisation_method = template.discretisation_method
        self.discretisation_bin_number = template.discretisation_bin_number
        self.discretisation_bin_width = template.discretisation_bin_width

    def promote(self):
        from mirp.images.ctImage import CTImage
        from mirp.images.petImage import PETImage
        from mirp.images.mrImage import MRImage

        if self.modality == "ct":
            image = CTImage(image_data=self.image_data)
        elif self.modality in ["pet", "pt"]:
            image = PETImage(image_data=self.image_data)
        elif self.modality in ["mr", "mri"]:
            image = MRImage(image_data=self.image_data)
        elif self.modality == "generic":
            return self
        else:
            raise ValueError(f"The imaging modality was not recognised: {self.modality}")

        # Set remaining attributes.
        image.update_from_template(template=self)

        return image

    def get_slices(
            self,
            slice_number: None | int | list[int] = None
    ) -> None | Self | list[Self]:

        image_list = []
        return_list = True

        if slice_number is None:
            slice_number = list(range(self.image_dimension[0]))
        elif isinstance(slice_number, int):
            return_list = False
            slice_number = [slice_number]

        for current_slice_id in slice_number:
            # Check if slice is not empty and falls within the allowed range.
            if self.is_empty():
                continue
            if not 0 <= current_slice_id < self.image_dimension[0]:
                continue

            slice_image = self.copy(drop_image=True)
            slice_image.image_origin = tuple(self.to_world_coordinates(x=np.array([current_slice_id, 0, 0])))
            slice_image.slice_id = current_slice_id
            slice_image.set_voxel_grid(voxel_grid=self.get_voxel_grid()[current_slice_id:current_slice_id + 1, :, :])

            image_list += [slice_image]

        if len(image_list) == 0:
            return None
        elif return_list:
            return image_list
        else:
            return image_list[0]

    def drop_image(self):
        self.image_data = None

    def is_empty(self) -> bool:
        return self.image_data is None

    def set_voxel_grid(self, voxel_grid: np.ndarray):
        self.image_data = voxel_grid
        self.image_dimension = tuple(voxel_grid.shape)

    def get_voxel_grid(self) -> None | np.ndarray:
        return self.image_data

    def update_image_data(self):
        pass

    def show(self, mask=None):
        if self.is_empty():
            return

        import matplotlib.pyplot as plt
        from mirp.images.utilities import InteractivePlot

        figure, axes = plt.subplots()

        # Create an index tracked object
        tracker = InteractivePlot(
            axes=axes,
            image=self,
            mask=mask)

        figure.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()

    @staticmethod
    def get_colour_map():
        return "gist_gray"

    @staticmethod
    def get_default_lowest_intensity():
        return None

    @staticmethod
    def get_default_upper_intensity():
        return None

    def interpolate(
            self,
            by_slice: None | bool = None,
            interpolate: None | bool = None,
            new_spacing: None | tuple[float, ...] = None,
            translation: None | float | tuple[float, ...] = (0.0, 0.0, 0.0),
            rotation: None | float = 0.0,
            spline_order: None | int = None,
            anti_aliasing: None | bool = None,
            anti_aliasing_smoothing_beta: None | float = None,
            settings: None | SettingsClass = None
    ):

        if self.separate_slices is not None:
            by_slice = self.separate_slices

        if (by_slice is None or interpolate is None or spline_order is None or anti_aliasing is
                None or anti_aliasing_smoothing_beta is None) and settings is None:
            raise ValueError("None of the parameters for interpolation can be set.")

        if by_slice is None:
            by_slice = settings.general.by_slice
        if interpolate is None:
            interpolate = settings.img_interpolate.interpolate
        if new_spacing is None and settings is not None:
            new_spacing = settings.img_interpolate.new_spacing
        if translation is None:
            if settings is not None:
                translation_x = settings.perturbation.translate_x if settings.perturbation.translate_x is not None else 0.0
                translation_y = settings.perturbation.translate_y if settings.perturbation.translate_y is not None else 0.0
                translation_z = settings.perturbation.translate_z if settings.perturbation.translate_z is not None else 0.0
                translation = tuple([translation_z, translation_y, translation_x])
            else:
                translation = (0.0, 0.0, 0.0)
        elif isinstance(translation, float):
            translation = tuple([translation] * 3)

        if rotation is None and settings is not None:
            rotation = settings.perturbation.rotation_angles[0]
        if spline_order is None:
            spline_order = self.get_interpolation_spline_order(settings=settings)
        if anti_aliasing is None:
            anti_aliasing = settings.img_interpolate.anti_aliasing
        if anti_aliasing_smoothing_beta is None:
            anti_aliasing_smoothing_beta = settings.img_interpolate.smoothing_beta

        # Set spacing
        if new_spacing is None or not interpolate:
            # Use original spacing.
            new_spacing = self.image_spacing

        elif by_slice:
            # Use provided spacing, in 2D. Spacing for interpolation across slices is set to the original spacing in
            # case interpolation is only conducted within the slice.
            new_spacing = list(new_spacing)
            new_spacing[0] = self.image_spacing[0]

        else:
            # Use provided spacing, in 3D
            new_spacing = list(new_spacing)

        # Set translation. Check that translation is specified for every direction.
        translation = list(translation)
        if len(translation) == 1:
            translation *= 3
        for ii in range(len(translation)):
            if translation[ii] is None:
                translation[ii] = 0.0

        if by_slice:
            translation[0] = 0.0

        translation: tuple[float, ...] = tuple(translation)

        return self._interpolate(
            by_slice=by_slice,
            interpolate=interpolate,
            new_spacing=tuple(new_spacing),
            translation=translation,
            rotation=rotation,
            spline_order=spline_order,
            anti_aliasing=anti_aliasing,
            anti_aliasing_smoothing_beta=anti_aliasing_smoothing_beta
        )

    @staticmethod
    def get_interpolation_spline_order(settings: SettingsClass):
        return settings.img_interpolate.spline_order

    def _interpolate(
            self,
            by_slice: bool,
            interpolate: bool,
            new_spacing: tuple[float, ...],
            translation: tuple[float, ...],
            rotation: float,
            spline_order: int,
            anti_aliasing: bool,
            anti_aliasing_smoothing_beta: float
    ):
        """Performs interpolation of the image volume"""
        from mirp.imageProcess.antiAliasing import gaussian_preprocess_filter
        from scipy.ndimage import map_coordinates

        # Skip for missing images.
        if self.is_empty() is None:
            return

        if self.separate_slices is not None:
            by_slice = self.separate_slices

        # Translate tuples to np.array
        new_spacing = np.array(new_spacing).astype(float)
        translation = np.array(translation).astype(float)

        # Convert translation to [0.0, 1.0 range].
        translation = translation - np.floor(translation)

        # Set translation
        self.translation = translation

        # Rotation around the z-axis (initial axis in numpy)
        rotation_angle = np.radians(rotation)

        # Set rotation
        self.rotation_angle = rotation

        # Skip if nor interpolation, nor affine transformation are required.
        if not interpolate and np.allclose(translation, 0.0) and np.isclose(rotation_angle, 0.0):
            return

        # Check if pre-processing is required
        if anti_aliasing:
            self.set_voxel_grid(
                voxel_grid=gaussian_preprocess_filter(
                    orig_vox=self.get_voxel_grid(),
                    orig_spacing=np.array(self.image_spacing),
                    sample_spacing=new_spacing,
                    param_beta=anti_aliasing_smoothing_beta,
                    mode="nearest",
                    by_slice=by_slice
                )
            )

        # Determine dimensions of voxel grid after resampling.
        sample_dim = np.ceil(
            np.multiply(np.array(self.image_dimension), np.array(self.image_spacing) / new_spacing)
        ).astype(int)

        # Set grid spacing (i.e. a fractional spacing in input voxel dimensions)
        grid_spacing = new_spacing / np.array(self.image_spacing)

        # Set grid origin according to IBSI reference manual, and update with the translation.
        grid_origin = translation * grid_spacing + 0.5 * (np.array(self.image_dimension) - 1.0) - 0.5 * (np.array(sample_dim) - 1.0) * grid_spacing

        # Set grid origin in world coordinates.
        world_grid_origin = self.to_world_coordinates(x=grid_origin, trim_result=False)

        # Establish the rotation center.
        world_rotation_center = self.to_world_coordinates(x=grid_origin + 0.5 * (sample_dim - 1) * grid_spacing)

        # Compute affine and inverse affine matrix.
        affine_matrix = self.get_affine_matrix()
        inverse_affine_matrix = self.get_affine_matrix(inverse=True)

        # Create affine matrix for rotating around the rotation center.
        #   W' = T * R * T^-1 * W
        #      = A_geometric * W
        # Here T^-1 moves the center of the world space to the rotation center, R performs the rotation and T^-1
        # moves the center of world space to the origin. Since all matrices are square and have the same dimensions,
        # we will first compute T * R * T^-1, and then multiply this with grid coordinates in world coordinates.

        # Set up the rotation matrix for rotation in the y-x plane.
        rotation_matrix = np.identity(4, dtype=float)
        rotation_matrix[1:3, 1:3] = np.array(
            [[np.cos(rotation_angle), np.sin(rotation_angle)],
             [-np.sin(rotation_angle), np.cos(rotation_angle)]]
        )

        # Create the entire affine matrix to perform the rotation around the rotation center.
        geometric_affine_matrix = np.matmul(
            rotation_matrix, self._get_origin_matrix(origin=world_rotation_center, inverse=True))
        geometric_affine_matrix = np.matmul(
            self._get_origin_matrix(origin=world_rotation_center), geometric_affine_matrix)

        # Multiply all matrices so: A_affine ^-1 * A_geometric * A_affine
        matrix = np.matmul(geometric_affine_matrix, affine_matrix)
        matrix = np.matmul(inverse_affine_matrix, matrix)

        # Generate interpolation grid in voxel space.
        voxel_map_z, voxel_map_y, voxel_map_x = np.mgrid[:sample_dim[0], :sample_dim[1], :sample_dim[2]]
        voxel_map_z = voxel_map_z.flatten() * grid_spacing[0] + grid_origin[0]
        voxel_map_y = voxel_map_y.flatten() * grid_spacing[1] + grid_origin[1]
        voxel_map_x = voxel_map_x.flatten() * grid_spacing[2] + grid_origin[2]

        # Set up voxel map coordinates as a 4 x n matrix.
        voxel_map_coordinates = np.array(
            [voxel_map_z, voxel_map_y, voxel_map_x, np.ones(voxel_map_x.shape, dtype=float)]
        )

        # Free up voxel map variables.
        del voxel_map_x, voxel_map_y, voxel_map_z

        # Rotate and translate the voxel grid.
        voxel_map_coordinates = np.matmul(matrix, voxel_map_coordinates)[0:3, :]

        # Interpolate orig_vox on interpolation grid
        sample_voxel_grid = map_coordinates(
            input=self.get_voxel_grid(),
            coordinates=voxel_map_coordinates,
            order=spline_order,
            mode="nearest"
        )

        # Shape map_vox to the correct dimensions.
        sample_voxel_grid = np.reshape(sample_voxel_grid, sample_dim)

        # Orientation changes through rotation.
        self.image_orientation = np.matmul(rotation_matrix, self._get_orientation_matrix())[0:3, 0:3]

        # Update origin
        self.image_origin = tuple(
            np.squeeze(np.matmul(geometric_affine_matrix, world_grid_origin[:, np.newaxis]), axis=1)[0:3]
        )

        # Update spacing and affine matrix.
        self.image_spacing = tuple(new_spacing)

        # Set interpolation
        self.interpolated = True

        # Set interpolation algorithm
        if spline_order == 0:
            self.interpolation_algorithm = "nnb"
        elif spline_order == 1:
            self.interpolation_algorithm = "lin"
        elif spline_order > 1:
            self.interpolation_algorithm = "si" + str(spline_order)

        # Set voxel grid
        self.set_voxel_grid(voxel_grid=sample_voxel_grid)
        self.update_image_data()

    def register(
            self,
            image,
            spline_order: None | int = None,
            anti_aliasing: None | bool = None,
            anti_aliasing_smoothing_beta: None | float = None,
            settings: None | SettingsClass = None
    ):
        if (spline_order is None or anti_aliasing is None or anti_aliasing is None) and settings is None:
            raise ValueError("None of the parameters for registration can be set.")

        if spline_order is None:
            spline_order = self.get_interpolation_spline_order(settings=settings)
        if anti_aliasing is None:
            anti_aliasing = settings.img_interpolate.anti_aliasing
        if anti_aliasing_smoothing_beta is None:
            anti_aliasing_smoothing_beta = settings.img_interpolate.smoothing_beta

        return self._register(
            image=image,
            spline_order=spline_order,
            anti_aliasing=anti_aliasing,
            anti_aliasing_smoothing_beta=anti_aliasing_smoothing_beta
        )

    def _register(
            self,
            image,
            spline_order: int,
            anti_aliasing: bool,
            anti_aliasing_smoothing_beta: float
    ):
        """Register this image with another image."""

        from scipy.ndimage import map_coordinates
        from mirp.imageProcess.antiAliasing import gaussian_preprocess_filter

        # This is just for type hinting. Use typing.Self once this is supported by the codestack.
        image: GenericImage = image

        # Skip if either internal or external image data are missing.
        if self.is_empty() or image.is_empty():
            return

        # Check whether registration is required
        registration_required = False

        # Mismatch in grid dimension
        if not np.array_equal(self.image_dimension, image.image_dimension):
            registration_required = True

        # Mismatch in origin
        if not np.allclose(self.image_origin, image.image_origin):
            registration_required = True

        # Mismatch in spacing
        if not np.allclose(self.image_spacing, image.image_spacing):
            registration_required = True

        # Mismatch in orientation
        if not np.allclose(self.image_orientation, image.image_orientation):
            registration_required = True

        if not registration_required:
            return

        # Apply anti-aliasing.
        if anti_aliasing:
            self.set_voxel_grid(
                voxel_grid=gaussian_preprocess_filter(
                    orig_vox=self.get_voxel_grid(),
                    orig_spacing=np.array(self.image_spacing),
                    sample_spacing=np.array(image.image_spacing),
                    param_beta=anti_aliasing_smoothing_beta,
                    mode="nearest",
                    by_slice=False
                )
            )

        # Create grid coordinates in world space using the image object.
        grid_coordinates = image.world_coordinates()

        # Translate grid coordinates into voxel space of the current image.
        grid_coordinates = self.to_voxel_coordinates(x=grid_coordinates)

        # Interpolate at the grid coordinates.
        new_mask = map_coordinates(
            input=self.get_voxel_grid().astype(float),
            coordinates=grid_coordinates,
            order=spline_order,
            mode="nearest"
        )

        # Restore form.
        new_mask = np.reshape(new_mask, image.image_dimension)

        # Update positional and affine parameters of the image.
        self.image_orientation = copy.deepcopy(image.image_orientation)
        self.image_origin = copy.deepcopy(image.image_origin)
        self.image_spacing = copy.deepcopy(image.image_spacing)

        # Update translation and rotation using the image object.
        self.translation = image.translation
        self.rotation_angle = image.rotation_angle

        # Set and update image after registration.
        self.set_voxel_grid(voxel_grid=new_mask)
        self.update_image_data()

    def add_noise(self, noise_level, noise_iteration_id):
        """
         Adds Gaussian noise to the image volume
         noise_level: standard deviation of image noise present """

        # Add noise iteration number
        self.noise_iteration_id = noise_iteration_id

        # Skip for missing images
        if self.image_data is None:
            return

        # Skip for invalid noise levels
        if noise_level is None:
            return
        if np.isnan(noise_level) or noise_level < 0.0:
            return

        # Add Gaussian noise to image
        voxel_grid = self.get_voxel_grid()
        voxel_grid += np.random.normal(loc=0.0, scale=noise_level, size=self.image_dimension)

        # Set noise level in image
        self.noise_level = noise_level

        # Update image.
        self.set_voxel_grid(voxel_grid=voxel_grid)
        self.update_image_data()

    def estimate_noise(self, method="chang"):

        import scipy.ndimage as ndi

        # Skip if the image is missing
        if self.is_empty():
            return

        if method == "rank":
            """ Estimate image noise level using the method by Rank, Lendl and Unbehauen, Estimation of 
            image noise variance, IEEE Proc. Vis. Image Signal Process (1999) 146:80-84"""

            # Step 1: filter with a cascading difference filter to suppress original image
            difference_filter = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])

            # Filter voxel volume
            response_map = ndi.convolve1d(self.get_voxel_grid(), weights=difference_filter, axis=1)
            response_map = ndi.convolve1d(response_map, weights=difference_filter, axis=2)

            # Step 2: compute histogram of local standard deviation and calculate histogram
            # Calculate local means
            local_means = ndi.uniform_filter(response_map, size=[1, 3, 3])

            # Calculate local sum of squares
            sum_filter = np.array([1.0, 1.0, 1.0]) / 3.0
            local_sum_square = ndi.convolve1d(np.power(response_map, 2.0), weights=sum_filter, axis=1)
            local_sum_square = ndi.convolve1d(local_sum_square, weights=sum_filter, axis=2)

            # Calculate local variance
            local_variance = 1.0 / 8.0 * (local_sum_square - 9.0 * np.power(local_means, 2.0))

            # Step 3: calculate median noise - this differs from the original
            # Set local variances below 0 (due to floating point rounding) to 0.
            local_variance = np.ravel(local_variance)
            local_variance[local_variance < 0.0] = 0.0

            # Select robust range (within IQR)
            local_variance = local_variance[
                np.percentile(local_variance, 25) <= local_variance <= np.percentile(local_variance, 75)
            ]

            # Calculate Gaussian noise
            estimated_noise = np.sqrt(np.mean(local_variance))

            del local_variance

        elif method == "ikeda":
            """ Estimate image noise level using a method by Ikeda, Makino, Imai et al., A method for estimating noise
             variance of CT image, Comp Med Imaging Graph (2010) 34:642-650"""

            # Step 1: filter with a cascading difference filter to suppress original image volume
            diff_filter = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])

            # Filter voxel volume
            response_map = ndi.convolve1d(self.get_voxel_grid(), weights=diff_filter, axis=1)
            response_map = ndi.convolve1d(response_map, weights=diff_filter, axis=2)

            # Step 2: calculate median noise
            estimated_noise = np.median(np.abs(response_map)) / 0.6754

        elif method == "chang":
            """ Noise estimation based on wavelets used in Chang, Yu and Vetterli, Adaptive wavelet thresholding for image
            denoising and compression. IEEE Trans Image Proc (2000) 9:1532-1546"""

            import pywt

            # Step 1: calculate HH subband of the wavelet transformation
            # Generate digital wavelet filter
            hi_filt = np.array(pywt.Wavelet("coif1").dec_hi)

            # Calculate HH subband image
            response_map = ndi.convolve1d(self.get_voxel_grid(), weights=hi_filt, axis=1)
            response_map = ndi.convolve1d(response_map, weights=hi_filt, axis=2)

            # Step 2: calculate median noise
            estimated_noise = np.median(np.abs(response_map)) / 0.6754

        elif method == "immerkaer":
            """ Noise estimation based on laplacian filtering, described in Immerkaer, Fast noise variance estimation.
            Comput Vis Image Underst (1995) 64:300-302"""

            # Step 1: construct filter and filter voxel volume
            # Create filter
            noise_filt = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], ndmin=3)

            # Apply filter
            response_map = ndi.convolve(self.get_voxel_grid(), weights=noise_filt)

            # Step 2: calculate noise level
            estimated_noise = np.sqrt(np.mean(np.power(response_map, 2.0))) / 36.0

        elif method == "zwanenburg":
            """ Noise estimation based on blob detection for weighting immerkaer filtering """

            # Step 1: construct laplacian filter and filter voxel volume
            # Create filter
            noise_filt = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], ndmin=3)

            # Apply filter
            response_map = ndi.convolve(self.get_voxel_grid(), weights=noise_filt)
            response_map = np.power(response_map, 2.0)

            # Step 2: construct blob weighting
            # Spacing for gaussian
            gauss_filt_spacing = np.full(shape=3, fill_value=np.min(self.image_spacing))
            gauss_filt_spacing = np.divide(gauss_filt_spacing, np.array(self.image_spacing))

            # Difference of gaussians
            weight_vox = ndi.gaussian_filter(
                self.get_voxel_grid(),
                sigma=1.0 * gauss_filt_spacing
            ) - ndi.gaussian_filter(
                self.get_voxel_grid(),
                sigma=4.0 * gauss_filt_spacing
            )

            # Smooth edge detection
            weight_vox = ndi.gaussian_filter(np.abs(weight_vox), sigma=2.0 * gauss_filt_spacing)

            # Convert to weighting scale
            weight_vox = 1.0 - weight_vox / np.max(weight_vox)

            # Decrease weight of edge voxels
            weight_vox = np.power(weight_vox, 2.0)

            # Step 3: estimate noise level
            estimated_noise = np.sqrt(np.sum(np.multiply(response_map, weight_vox)) / (36.0 * np.sum(weight_vox)))

        else:
            raise ValueError(
                "The provided noise estimation method is not implemented. Use one of \"chang\" (default), "
                "\"rank\", \"ikeda\", \"immerkaer\" or \"zwanenburg\"."
            )

        return estimated_noise

    def saturate(self, intensity_range, fill_value=None):
        """
        Saturate image intensities using an intensity range
        :param intensity_range: range of intensity values
        :param fill_value: fill value for out-of-range intensities. If None, the upper and lower ranges are used
        :return:
        """
        # Skip for missing images
        if self.image_data is None or intensity_range is None:
            return

        intensity_range = np.array(copy.deepcopy(intensity_range))

        if np.any(~np.isnan(intensity_range)):
            # Set updated image data.
            image_data = self.get_voxel_grid()

            # Saturate values below lower boundary.
            if not np.isnan(intensity_range[0]):
                if fill_value is None:
                    image_data[image_data < intensity_range[0]] = intensity_range[0]
                else:
                    image_data[image_data < intensity_range[0]] = fill_value[0]

            # Saturate values above upper boundary.
            if not np.isnan(intensity_range[1]):
                if fill_value is None:
                    image_data[image_data > intensity_range[1]] = intensity_range[1]
                else:
                    image_data[image_data > intensity_range[1]] = fill_value[1]

            # Set the updated image data.
            self.set_voxel_grid(voxel_grid=image_data)
            self.update_image_data()

    def normalise_intensities(
            self,
            normalisation_method: None | str = "none",
            intensity_range: None | tuple[Any, Any] = None,
            saturation_range: None | tuple[Any, Any] = None,
            mask: None | np.ndarray = None
    ) -> Self:
        """
        Normalises image intensities
        :param normalisation_method: string defining the normalisation method. Should be one of "none", "range",
        "relative_range", "quantile_range", "standardisation".
        :param intensity_range: range of intensities for normalisation.
        :param saturation_range: range of allowed intensity values.
        :param mask: sets area that should be considered for determining normalisation parameters.
        :return:
        """

        # Skip for missing images
        if self.is_empty():
            return self

        if intensity_range is None:
            intensity_range = [np.nan, np.nan]

        if mask is None:
            mask = np.ones(self.image_dimension, dtype=bool)
        else:
            mask = mask.astype(bool)

        if np.sum(mask) == 0:
            mask = np.ones(self.image_dimension, dtype=bool)

        if saturation_range is None:
            saturation_range = [np.nan, np.nan]

        if normalisation_method is None or normalisation_method == "none":
            return self

        elif normalisation_method == "range":
            # Normalisation to [0, 1] range using fixed intensities.

            # Get image data
            image_data = self.get_voxel_grid()

            # Find maximum and minimum intensities
            if np.isnan(intensity_range[0]):
                min_int = np.min(image_data[mask])
            else:
                min_int = intensity_range[0]

            if np.isnan(intensity_range[1]):
                max_int = np.max(image_data[mask])
            else:
                max_int = intensity_range[1]

            # Normalise by range
            if not max_int == min_int:
                image_data = (image_data - min_int) / (max_int - min_int)
            else:
                image_data = image_data - min_int

            # Update image data
            self.set_voxel_grid(voxel_grid=image_data)

        elif normalisation_method == "relative_range":
            # Normalisation to [0, 1]-ish range using relative intensities.

            # Get image data
            image_data = self.get_voxel_grid()

            min_int_rel = 0.0
            if not np.isnan(intensity_range[0]):
                min_int_rel = intensity_range[0]

            max_int_rel = 1.0
            if not np.isnan(intensity_range[1]):
                max_int_rel = intensity_range[1]

            # Compute minimum and maximum intensities.
            value_range = [np.min(image_data[mask]), np.max(image_data[mask])]
            min_int = value_range[0] + min_int_rel * (value_range[1] - value_range[0])
            max_int = value_range[0] + max_int_rel * (value_range[1] - value_range[0])

            # Normalise by range
            if not max_int == min_int:
                image_data = (image_data - min_int) / (max_int - min_int)
            else:
                image_data = image_data - min_int

            # Update image data
            self.set_voxel_grid(voxel_grid=image_data)

        elif normalisation_method == "quantile_range":
            # Normalisation to [0, 1]-ish range based on quantiles.

            # Get image data
            image_data = self.get_voxel_grid()

            min_quantile = 0.0
            if not np.isnan(intensity_range[0]):
                min_quantile = intensity_range[0]

            max_quantile = 1.0
            if not np.isnan(intensity_range[1]):
                max_quantile = intensity_range[1]

            # Compute quantiles from voxel grid.
            min_int = np.quantile(image_data[mask], q=min_quantile)
            max_int = np.quantile(image_data[mask], q=max_quantile)

            # Normalise by range
            if not max_int == min_int:
                image_data = (image_data - min_int) / (max_int - min_int)
            else:
                image_data -= min_int

            # Update image data
            self.set_voxel_grid(voxel_grid=image_data)

        elif normalisation_method == "standardisation":
            # Normalisation to mean 0 and standard deviation 1.

            # Get image data
            image_data = self.get_voxel_grid()

            # Determine mean and standard deviation of the voxel intensities
            mean_int = np.mean(image_data[mask])
            sd_int = np.std(image_data[mask])

            # Protect against invariance.
            if sd_int == 0.0:
                sd_int = 1.0

            # Normalise
            image_data = (image_data - mean_int) / sd_int

            # Update image data
            self.set_voxel_grid(voxel_grid=image_data)
        else:
            raise ValueError(f"{normalisation_method} is not a valid method for normalising intensity values.")

        self.saturate(intensity_range=saturation_range)

        return self

    def decimate(self, by_slice: bool):
        """
        Decimates image by removing every second element
        :param by_slice: Whether the analysis is conducted in 2D or 3D.
        :return:
        """

        # Skip for missing images
        if self.is_empty():
            return

        if self.separate_slices is not None:
            by_slice = self.separate_slices

        # Get the voxel grid
        image_data = self.get_voxel_grid()
        image_spacing = np.array(self.image_spacing)

        # Update the voxel grid
        if by_slice:
            # Drop every second pixel
            image_data = image_data[:, slice(None, None, 2), slice(None, None, 2)]
            self.image_spacing = tuple(image_spacing[[1, 2]] * 2.0)

        else:
            # Drop every second voxel
            image_data = image_data[slice(None, None, 2), slice(None, None, 2), slice(None, None, 2)]

            # Update voxel spacing
            self.image_spacing = tuple(image_spacing * 2.0)

        # Update voxel grid. This also updates the size attribute.
        self.set_voxel_grid(voxel_grid=image_data)

    def crop(
            self,
            ind_ext_z=None,
            ind_ext_y=None,
            ind_ext_x=None,
            xy_only=False,
            z_only=False):
        """Crop image to the provided map extent."""

        # Skip for missing images
        if self.is_empty():
            return

        # Determine corresponding voxel indices
        max_ind = np.ceil(np.array((np.max(ind_ext_z), np.max(ind_ext_y), np.max(ind_ext_x)))).astype(int)
        min_ind = np.floor(np.array((np.min(ind_ext_z), np.min(ind_ext_y), np.min(ind_ext_x)))).astype(int)

        # Set bounding indices
        max_bound_ind = np.minimum(max_ind, np.array(self.image_dimension)).astype(int)
        min_bound_ind = np.maximum(min_ind, np.array([0, 0, 0])).astype(int)

        # Get image data.
        image_data = self.get_voxel_grid()

        # Create corresponding image volumes by slicing original volume
        if z_only:
            image_data = image_data[
                min_bound_ind[0]:max_bound_ind[0] + 1,
                :,
                :
            ]
            min_bound_ind[1] = 0
            min_bound_ind[2] = 0
        elif xy_only:
            image_data = image_data[
                :,
                min_bound_ind[1]:max_bound_ind[1] + 1,
                min_bound_ind[2]:max_bound_ind[2] + 1
            ]
            min_bound_ind[0] = 0
            max_bound_ind[0] = np.array(self.image_dimension)[0].astype(int)
        else:
            image_data = image_data[
                min_bound_ind[0]:max_bound_ind[0] + 1,
                min_bound_ind[1]:max_bound_ind[1] + 1,
                min_bound_ind[2]:max_bound_ind[2] + 1
            ]

        # Update origin
        self.image_origin = tuple(self.to_world_coordinates(x=np.array(min_bound_ind)))

        # Update voxel grid
        self.set_voxel_grid(voxel_grid=image_data)

    def crop_to_size(self, center, crop_size):
        """Crop images to the exact size"""

        # Skip for missing images
        if self.is_empty():
            return

        # Make local copies.
        crop_size = copy.deepcopy(crop_size)
        center = copy.deepcopy(center)

        if len(crop_size) == 2:
            crop_size = [None, crop_size[0], crop_size[1]]
        if len(center) == 2:
            center = [None, center[0], center[1]]

        # Set origin for the new grid.
        new_grid_index = [
            int(0.5 * (center[ii] - crop_size[ii]))
            if center[ii] is not None and crop_size[ii] is not None else 0
            for ii in range(len(center))
        ]

        # Update crop size.
        crop_size = [
            crop_size[ii]
            if crop_size[ii] is not None else self.image_dimension[ii]
            for ii in range(len(crop_size))
        ]
        crop_size = np.array(crop_size)
        new_grid_index = np.array(new_grid_index)

        # Determine coordinates of the box that can be copied in the original space
        max_index_original = new_grid_index + crop_size
        min_index_original = new_grid_index

        # Update coordinates based on boundaries in the original images
        max_index_original = np.minimum(max_index_original, np.array(self.image_dimension))
        min_index_original = np.maximum(min_index_original, np.array([0, 0, 0]))

        # Determine coordinates where this box should land, i.e. perform the coordinate transformation to grid index space.
        max_index_crop = max_index_original - new_grid_index
        min_index_crop = min_index_original - new_grid_index

        # Create an empty voxel_grid to copy to
        cropped_image = np.full(crop_size, fill_value=np.nan)

        # Put the voxel grid slice into the cropped grid
        cropped_image[
            min_index_crop[0]:max_index_crop[0],
            min_index_crop[1]:max_index_crop[1],
            min_index_crop[2]:max_index_crop[2]
        ] = self.get_voxel_grid()[
            min_index_original[0]:max_index_original[0],
            min_index_original[1]:max_index_original[1],
            min_index_original[2]:max_index_original[2]
        ]

        # Replace any remaining NaN values in the grid by the lowest intensity in voxel_grid
        cropped_image[np.isnan(cropped_image)] = np.min(self.get_voxel_grid())

        # Restore the original dtype in case it got lost
        cropped_image = cropped_image.astype(self.get_voxel_grid().dtype)

        # Update origin
        self.image_origin = tuple(self.to_world_coordinates(x=np.array(new_grid_index)))

        # Set voxel grid
        self.set_voxel_grid(voxel_grid=cropped_image)

    def get_supervoxels(
            self,
            intensity_range: tuple[float]
    ):
        """Extracts supervoxels from an image"""

        from skimage.segmentation import slic
        from mirp.imageProcess.utilities import extend_intensity_range
        from mirp.imageProcess.utilities import set_intensity_range

        if self.is_empty():
            return None

        # Update or set intensity range, and extend it by around 10% on either side.
        intensity_range = set_intensity_range(image=self, intensity_range=intensity_range)
        intensity_range = extend_intensity_range(intensity_range=intensity_range, extend_fraction=0.1)

        # Get image data
        image_data = copy.deepcopy(self.get_voxel_grid())

        # Apply threshold
        image_data[image_data < intensity_range[0]] = intensity_range[0]
        image_data[image_data > intensity_range[1]] = intensity_range[1]

        # Slic constants - sigma
        sigma = 1.0 * np.min(self.image_spacing)

        # Slic constants - number of segments.
        min_n_voxels = np.max([20.0, 500.0 / np.prod(self.image_spacing)])
        n_segments = int(np.prod(self.image_dimension) / min_n_voxels)

        # Convert to float with range [0.0, 1.0]
        image_data -= intensity_range[0]
        image_data *= 1.0 / (intensity_range[1] - intensity_range[0])

        if image_data.dtype not in ["float", "float64"]:
            image_data = image_data.astype(float)

        # Create a slic segmentation of the image stack
        image_segments = slic(
            image=image_data,
            n_segments=n_segments,
            sigma=sigma,
            spacing=self.image_spacing,
            compactness=0.05,
            convert2lab=False,
            enforce_connectivity=True,
            channel_axis=None
        )

        image_segments += 1

        # Release image_data
        del image_data

        return image_segments

    def bias_field_correction(self, in_place=True, **kwargs):
        if not in_place:
            return self.copy()

    def write(
            self,
            dir_path: str,
            file_name: None | str = None,
            file_format: str = "nifti"
    ):
        """ Writes the image to a file """
        import os
        import itk

        if self.is_empty():
            return

        # Check if path exists
        dir_path = os.path.normpath(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Generate filename, if necessary.
        if file_name is None:
            file_name = "_".join(self.get_file_name_descriptor())

        # Add extension.
        if file_format == "nifti":
            file_name += ".nii.gz"
        elif file_format == "numpy":
            file_name += ".npy"
        else:
            raise ValueError(f"The provided file format {file_format} is not available.")

        # Add file and file name
        file_path = os.path.join(dir_path, file_name)

        if file_format == "nifti":
            image_data = self.get_voxel_grid()
            if np.issubdtype(self.image_data.dtype, bool):
                cast_type = np.uint8
            elif np.issubdtype(self.image_data.dtype, float):
                cast_type = float
            else:
                cast_type = self.image_data.dtype
            image_data = itk.GetImageFromArray(image_data.astype(cast_type))
            image_data.SetOrigin(np.array(self.image_origin)[::-1])
            image_data.SetSpacing(np.array(self.image_spacing)[::-1])
            image_data.SetDirection(itk.matrix_from_array(np.reshape(np.ravel(self.image_orientation)[::-1], [3, 3])))

            itk.imwrite(image_data, file_path)

        elif file_format == "numpy":
            image_data = self.get_voxel_grid()
            np.save(file_path, image_data)

    def get_file_name_descriptor(self) -> list[str]:
        descriptors = []

        # Sample name
        if self.sample_name is not None:
            descriptors += [self.sample_name]

        # Slice id
        if self.slice_id is not None:
            descriptors += [self.slice_id]

        # Modality
        if self.modality is not None:
            descriptors += [self.modality]

        # Interpolation
        if self.interpolated:
            descriptors += [
                self.interpolation_algorithm,
                "x", str(self.image_spacing[2])[:5],
                "y", str(self.image_spacing[1])[:5],
                "z", str(self.image_spacing[0])[:5]
            ]

        # Rotation
        if self.rotation_angle is not None and self.rotation_angle != 0.0:
            descriptors += ["rot", str(self.rotation_angle)[:5]]

        # Translation
        if self.translation is not None and not np.all(np.array(self.translation) == 0.0):
            descriptors += [
                "trans",
                "x", str(self.translation[2])[:5],
                "y", str(self.translation[1])[:5],
                "z", str(self.translation[0])[:5]
            ]

        # Noise
        if self.noise_level is not None and self.noise_level > 0.0:
            descriptors += ["noise", str(self.noise_level)[:5], "id", str(self.noise_iteration_id)]

        return descriptors

    def export(
            self,
            export_format: str = "dict"
    ) -> None | np.ndarray | dict[str, Any] | Self:

        if self.is_empty():
            return None

        if export_format == "dict":
            attributes = self.get_export_attributes()
            attributes.update({"image": self.get_voxel_grid()})

            return attributes
        elif export_format == "numpy":
            return self.get_voxel_grid()

        elif export_format == "native":
            return self.copy()

        else:
            raise ValueError(f"The current value of export_format was not recognised: {export_format}")

    def get_export_attributes(self) -> dict[str, Any]:
        attributes = []

        # Sample name
        if self.sample_name is not None:
            attributes += [("sample_name", self.sample_name)]

        # Slice id.
        if self.slice_id is not None:
            attributes += [("slice_id", self.slice_id)]

        # Modality
        if self.modality is not None:
            attributes += [("modality", self.modality)]

        # Rotation
        if self.rotation_angle is None:
            attributes += [("rotation", 0.0)]
        else:
            attributes += [("rotation", self.rotation_angle)]

        # Translation
        if self.translation is None:
            attributes += [("translation", (0.0, 0.0, 0.0))]
        else:
            attributes += [("translation", self.translation)]

        # Noise
        if self.noise_level is not None and self.noise_level > 0.0:
            attributes += [("noise_level", self.noise_level), ("noise_id", self.noise_iteration_id)]

        # Image spacing, origin and orientation.
        attributes += [
            ("image_spacing", self.image_spacing),
            ("image_origin", self.image_origin),
            ("image_orientation", self.image_orientation)
        ]

        return dict(attributes)

    def parse_feature_names(self, x: None | pd.DataFrame) -> pd.DataFrame:
        feature_name_suffix = []
        if self.discretisation_method is not None:
            if self.discretisation_method == "none":
                pass
            elif self.discretisation_method == "fixed_bin_number":
                feature_name_suffix += ["fbn"]
                feature_name_suffix += ["n" + str(self.discretisation_bin_number)]
            elif self.discretisation_method == "fixed_bin_size":
                feature_name_suffix += ["fbs"]
                feature_name_suffix += ["w" + str(self.discretisation_bin_width)]
            elif self.discretisation_method == "fixed_bin_size_pyradiomics":
                feature_name_suffix += ["fbsp"]
                feature_name_suffix += ["w" + str(self.discretisation_bin_width)]
            else:
                raise ValueError(f"Unknown discretisation method: {self.discretisation_method}")

        if len(feature_name_suffix) > 0:
            feature_name_suffix = "_".join(feature_name_suffix)
            feature_name_suffix = "_" + feature_name_suffix
            x.columns += feature_name_suffix

        return x
