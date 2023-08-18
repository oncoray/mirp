import copy
import numpy as np
from typing import Optional, Union

from mirp.images.baseImage import BaseImage
from mirp.importSettings import SettingsClass


class GenericImage(BaseImage):

    def __init__(
            self,
            image_data: Optional[np.ndarray],
            **kwargs
    ):

        super().__init__(**kwargs)

        self.image_data = image_data

        # These are set elsewhere.
        self.translation = 0.0
        self.rotation_angle = 0.0
        self.noise_iteration_id: Optional[int] = None
        self.noise_level = 0.0

        # Interpolation-related settings
        self.interpolated = False
        self.interpolation_algorithm: Optional[str] = None

        # Normalisation-related settings
        self.normalised = False

    def copy(self, drop_image=False):
        image = copy.deepcopy(self)

        if drop_image:
            image.drop_image()

        return image

    def drop_image(self):
        self.image_data = None

    def is_empty(self):
        return self.image_data is None

    def set_voxel_grid(self, voxel_grid: np.ndarray):
        self.image_data = voxel_grid
        self.image_dimension = tuple(voxel_grid.shape)

    def get_voxel_grid(self) -> Union[None, np.ndarray]:
        return self.image_data

    def update_image_data(self):
        pass

    def decimate(self, by_slice):
        """
        Decimates image by removing every second element
        :param by_slice: Whether the analysis is conducted in 2D or 3D.
        :return:
        """

        # Skip for missing images
        if self.image_data is None:
            return

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

    def interpolate(self, by_slice, settings: SettingsClass):
        """Performs interpolation of the image volume"""
        from mirp.imageProcess import gaussian_preprocess_filter
        from scipy.ndimage import map_coordinates

        # Skip for missing images
        if self.image_data is None:
            return

        # Read interpolation flag.
        interpolate_flag = settings.img_interpolate.interpolate

        # Local interpolation constants
        if settings.img_interpolate.new_spacing is None or not interpolate_flag:
            # Use original spacing.
            new_spacing = np.array(self.image_spacing)

        else:
            # Use provided spacing.
            new_spacing = settings.img_interpolate.new_spacing
            new_spacing = np.array(new_spacing)

        # Read order of multidimensional spline filter (0=nearest neighbours, 1=linear, 3=cubic)
        order = settings.img_interpolate.spline_order

        # Set spacing for interpolation across slices to the original spacing in case interpolation is only conducted
        # within the slice.
        if by_slice:
            new_spacing[0] = self.image_spacing[0]
        new_spacing = new_spacing.astype(float)

        # Image translation
        translation = np.array([
            settings.perturbation.translate_z,
            settings.perturbation.translate_y,
            settings.perturbation.translate_x
        ])

        # Convert translation to [0.0, 1.0 range].
        translation = translation - np.floor(translation)

        # Set translation
        self.translation = translation

        # Rotation around the z-axis (initial axis in numpy)
        rotation_angle = np.radians(settings.perturbation.rotation_angles)

        # Set rotation
        self.rotation_angle = settings.perturbation.rotation_angles

        # Skip if nor interpolation, nor affine transformation are required.
        if not interpolate_flag and np.allclose(translation, 0.0) and np.isclose(rotation_angle, 0.0):
            return

        # Check if pre-processing is required
        if settings.img_interpolate.anti_aliasing:
            self.set_voxel_grid(
                voxel_grid=gaussian_preprocess_filter(
                    orig_vox=self.get_voxel_grid(),
                    orig_spacing=np.array(self.image_spacing),
                    sample_spacing=new_spacing,
                    param_beta=settings.img_interpolate.smoothing_beta,
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
            order=order,
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
        if order == 0:
            self.interpolation_algorithm = "nnb"
        elif order == 1:
            self.interpolation_algorithm = "lin"
        elif order > 1:
            self.interpolation_algorithm = "si" + str(order)

        # Set voxel grid
        self.set_voxel_grid(voxel_grid=sample_voxel_grid)
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

    def saturate(self, intensity_range, fill_value=None):
        """
        Saturate image intensities using an intensity range
        :param intensity_range: range of intensity values
        :param fill_value: fill value for out-of-range intensities. If None, the upper and lower ranges are used
        :return:
        """
        # Skip for missing images
        if self.image_data is None:
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
            normalisation_method="none",
            intensity_range=None,
            saturation_range=None,
            mask=None):
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
        if self.image_data is None:
            return

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

        if normalisation_method == "none":
            return

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
            self.normalised = True

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
            self.normalised = True

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
            self.normalised = True

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
            self.normalised = True
        else:
            raise ValueError(f"{normalisation_method} is not a valid method for normalising intensity values.")

        self.saturate(intensity_range=saturation_range)

    def crop(
            self,
            ind_ext_z=None,
            ind_ext_y=None,
            ind_ext_x=None,
            xy_only=False,
            z_only=False):
        """Crop image to the provided map extent."""

        # Skip for missing images
        if self.image_data:
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
        if self.image_data is None:
            return

        # Make local copy
        crop_size = np.array(copy.deepcopy(crop_size))

        # Determine the new grid origin in the original index space. Only the dimensions with a number are updated
        grid_origin = np.round(center - crop_size / 2.0).astype(int)

        # Update grid origin and crop_size for the remainder of the calculation
        grid_origin[np.isnan(crop_size)] = 0
        crop_size[np.isnan(crop_size)] = np.array(self.image_dimension)[np.isnan(crop_size)]

        # Determine coordinates of the box that can be copied in the original space
        max_ind_orig = grid_origin + crop_size
        min_ind_orig = grid_origin

        # Update coordinates based on boundaries in the original images
        max_ind_orig = np.minimum(max_ind_orig, np.array(self.image_dimension)).astype(int)
        min_ind_orig = np.maximum(min_ind_orig, np.array([0, 0, 0])).astype(int)

        # Determine coordinates where this box should land, i.e. perform the coordinate transformation to grid index space.
        max_ind_grid = max_ind_orig - grid_origin
        min_ind_grid = min_ind_orig - grid_origin

        # Create an empty voxel_grid to copy to
        cropped_image = np.full(crop_size.astype(int), fill_value=np.nan)

        # Get slice of voxel grid
        image_data = self.get_voxel_grid()[
            min_ind_orig[0]:max_ind_orig[0],
            min_ind_orig[1]:max_ind_orig[1],
            min_ind_orig[2]:max_ind_orig[2]
        ]

        # Put the voxel grid slice into the cropped grid
        cropped_image[
            min_ind_grid[0]:max_ind_grid[0],
            min_ind_grid[1]:max_ind_grid[1],
            min_ind_grid[2]:max_ind_grid[2]
        ] = image_data

        # Replace any remaining NaN values in the grid by the lowest intensity in voxel_grid
        cropped_image[np.isnan(cropped_image)] = np.min(image_data)

        # Restore the original dtype in case it got lost
        cropped_image = cropped_image.astype(image_data.dtype)

        # Update origin
        self.image_origin = tuple(self.to_world_coordinates(x=np.array(min_ind_orig)))

        # Set voxel grid
        self.set_voxel_grid(voxel_grid=cropped_image)