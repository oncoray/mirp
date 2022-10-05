import copy
import os
import time
import warnings
from typing import Union

import numpy as np
import pandas as pd
from pydicom import FileDataset, Sequence, Dataset

from mirp.imageMetaData import get_pydicom_meta_tag, set_pydicom_meta_tag, create_new_uid
from mirp.utilities import get_version
from mirp.importSettings import SettingsClass


class ImageClass:
    # Class for image volumes

    def __init__(self,
                 voxel_grid,
                 origin,
                 spacing,
                 orientation,
                 modality=None,
                 spat_transform="base",
                 no_image=False,
                 metadata=None,
                 slice_table=None,
                 slice_position=None):

        # Set details regarding voxel orientation and such
        self.origin = np.array(origin)
        self.orientation = np.array(orientation)

        self.spat_transform = spat_transform        # Signifies whether the current image is a base image or not
        self.slice_table = slice_table

        # The spacing, the affine matrix and its inverse are set using the set_spacing method.
        self.spacing = None
        self.m_affine = None
        self.m_affine_inv = None

        # Set voxel spacing. This also set the affine matrix and its inverse.
        self.set_spacing(new_spacing=np.array(spacing))

        # Image name
        self.name = None

        # Initialise voxel grid dependent parameters
        self.isEncoded_voxel_grid = None
        self.voxel_grid = None
        self.size = None
        self.dtype_name = None

        # Interpolation settings
        self.interpolated = False
        self.interpolation_algorithm = None

        # Discretisation settings
        self.discretised = False
        self.discretisation_algorithm = None
        self.discretisation_settings = None

        # Noise addition parameters
        self.noise = -1.0
        self.noise_iter = 0

        # Translation parameters
        self.translation: Union[None, np.ndarray] = None

        # Rotation parameters
        self.rotation_angle: Union[None, float] = None

        # Set voxel grid and image
        if not no_image:
            self.is_missing = False
            self.set_voxel_grid(voxel_grid=voxel_grid)
        else:
            self.is_missing = True

        # Set metadata and a list of update tags
        self.metadata: Union[FileDataset, None] = metadata
        self.as_parametric_map = False

        # Image modality
        if modality is None and metadata is not None:
            # Set imaging modality using metadata
            self.modality = self.get_metadata(tag=(0x0008, 0x0060), tag_type="str")  # Imaging modality
        elif modality is None:
            self.modality = "GENERIC"
        else:
            self.modality = modality

        # Normalisation flags.
        self.is_normalised = False

        # Set slice position in case slices are not evenly spaced.
        self.slice_position = slice_position

    def copy(self, drop_image=False):
        # Creates a new copy of the image object
        img_copy = copy.deepcopy(self)

        if drop_image:
            img_copy.drop_image()

        return img_copy

    def show(self, img_slice):
        import matplotlib.pyplot as plt

        if self.is_missing:
            return

        # Set the colour map
        img_colour_map = "gist_gray"
        if self.modality == "PT":
            img_colour_map = "gist_yarg"

        # Create initial figure to draw on.
        fig, ax = plt.subplots(1)

        # Set figure space in axis
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Invisible axis
        ax.axis('off')

        # Plot image.
        plt.imshow(self.get_voxel_grid()[img_slice, :, :],
                   cmap=plt.get_cmap(img_colour_map),
                   extent=[0, 1.0, 0, 1.0],
                   alpha=1.0,
                   interpolation="none")

        plt.show()

    def set_spacing(self, new_spacing):

        # Update spacing
        self.spacing: np.ndarray = new_spacing

        # Recompute the affine matrices
        self.m_affine = np.reshape(np.repeat(self.spacing, 3), [3, 3]) * self.orientation
        self.m_affine_inv = np.linalg.inv(self.m_affine)

    def set_voxel_grid(self, voxel_grid):
        """ Sets voxel grid """

        # Determine size
        self.size = np.array(voxel_grid.shape)
        self.dtype_name = voxel_grid.dtype.name

        # Encode voxel grid
        self.encode_voxel_grid(voxel_grid=voxel_grid)

    # Return None for missing images
    def get_voxel_grid(self):
        """ Gets voxel grid """
        if self.is_missing:
            return None

        if self.isEncoded_voxel_grid:
            # Decode voxel grid (typically roi)
            decoded_voxel = np.zeros(np.prod(self.size), dtype=bool)

            # Check if the voxel grid contains values
            if self.voxel_grid is not None:
                decode_zip = copy.deepcopy(self.voxel_grid)

                for ii, jj in decode_zip:
                    decoded_voxel[ii:jj + 1] = True

            # Shape into correct form
            decoded_voxel = decoded_voxel.reshape(self.size)

            return decoded_voxel
        else:
            return self.voxel_grid

    def encode_voxel_grid(self, voxel_grid):
        """Performs run length encoding of the voxel grid"""

        # Determine whether the voxel grid should be encoded (only True for boolean data types; typically roi)
        if self.dtype_name == "bool":

            # Run length encoding for "True"
            rle_end = np.array(np.append(np.where(voxel_grid.ravel()[1:] != voxel_grid.ravel()[:-1]),
                                         np.prod(self.size) - 1))
            rle_start = np.cumsum(np.append(0, np.diff(np.append(-1, rle_end))))[:-1]
            rle_val = voxel_grid.ravel()[rle_start]

            # Check whether the voxel grid is empty (consists of 0s)
            if np.all(~rle_val):
                self.voxel_grid = None
                self.isEncoded_voxel_grid = True
            else:
                # Select only True values entries for further compression
                rle_start = rle_start[rle_val]
                rle_end = rle_end[rle_val]

                # Create zip
                self.voxel_grid = zip(rle_start, rle_end)
                self.isEncoded_voxel_grid = True
        else:
            self.voxel_grid = voxel_grid
            self.isEncoded_voxel_grid = False

    def decode_voxel_grid(self):
        """Performs run length decoding of the voxel grid and converts it to a numpy array"""
        if self.dtype_name == "bool" and self.isEncoded_voxel_grid:
            decoded_voxel = np.zeros(np.prod(self.size), dtype=bool)

            # Check if the voxel grid contains values
            if self.voxel_grid is not None:
                decode_zip = copy.deepcopy(self.voxel_grid)
                for ii, jj in decode_zip:
                    decoded_voxel[ii:jj + 1] = True

            # Set shape to original grid
            decoded_voxel = decoded_voxel.reshape(self.size)

            # Update self.voxel_grid and isEncoded_voxel_grid tags
            self.voxel_grid = decoded_voxel
            self.isEncoded_voxel_grid = False

    def decimate(self, by_slice):
        """
        Decimates image voxel grid by removing every second element
        :param by_slice:
        :return:
        """

        # Skip for missing images
        if self.is_missing:
            return

        # Get the voxel grid
        img_voxel_grid = self.get_voxel_grid()

        # Update the voxel grid
        if by_slice:
            # Drop every second pixel
            img_voxel_grid = img_voxel_grid[:, slice(None, None, 2), slice(None, None, 2)]

            # Update voxel spacing
            self.spacing[[1, 2]] *= 2.0
        else:
            # Drop every second voxel
            img_voxel_grid = img_voxel_grid[slice(None, None, 2), slice(None, None, 2), slice(None, None, 2)]

            # Update voxel spacing
            self.spacing *= 2.0

        # Update voxel grid. This also updates the size attribute.
        self.set_voxel_grid(voxel_grid=img_voxel_grid)

    def interpolate(self, by_slice, settings: SettingsClass):
        """Performs interpolation of the image volume"""
        from mirp.imageProcess import interpolate_to_new_grid, gaussian_preprocess_filter

        # Skip for missing images
        if self.is_missing:
            return

        # Check whether slice positions are explicitly set.
        if self.slice_position is not None:
            self.interpolate_missing_slices()

        # Local interpolation constants
        if settings.img_interpolate.new_spacing is None:
            # Use original spacing.
            new_spacing = self.spacing

        else:
            # Use provided spacing.
            new_spacing = settings.img_interpolate.new_spacing

            # For in-slice resampling, set the first element (z-direction) to the slice spacing.
            if by_slice:
                new_spacing[0] = self.spacing[0]

            # Convert to numpy array.
            new_spacing = np.array(new_spacing)

        # Read order of multidimensional spline filter (0=nearest neighbours, 1=linear, 3=cubic)
        order = settings.img_interpolate.spline_order

        # Read interpolation flag.
        interpolate_flag = settings.img_interpolate.interpolate

        # Set spacing for interpolation across slices to the original spacing in case interpolation is only conducted
        # within the slice.
        if by_slice:
            new_spacing[0] = self.spacing[0]

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
        rotation_matrix = np.array([[np.cos(rotation_angle), np.sin(rotation_angle)],
                                    [-np.sin(rotation_angle), np.cos(rotation_angle)]])

        # Set rotation
        self.rotation_angle = settings.perturbation.rotation_angles

        # Combine rotation and translation matrix into an affine matrix. See e.g.
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html
        affine_matrix = np.identity(4, dtype=float)
        affine_matrix[1:3, 1:3] = rotation_matrix
        affine_matrix[0:3, 3] = translation

        # Skip if nor interpolation, nor affine transformation are required.
        if np.allclose(np.identity(4, dtype=float), affine_matrix) and not interpolate_flag:
            return None

        # Check if pre-processing is required
        if settings.img_interpolate.anti_aliasing:
            self.set_voxel_grid(voxel_grid=gaussian_preprocess_filter(
                orig_vox=self.get_voxel_grid(),
                orig_spacing=self.spacing,
                sample_spacing=new_spacing,
                param_beta=settings.img_interpolate.smoothing_beta,
                mode="nearest",
                by_slice=by_slice
            ))

        # Interpolate image and positioning
        self.size, sample_spacing, upd_voxel_grid, grid_origin = interpolate_to_new_grid(
            orig_dim=self.size,
            orig_spacing=self.spacing,
            orig_vox=self.get_voxel_grid(),
            sample_spacing=new_spacing,
            affine_matrix=affine_matrix,
            order=order,
            mode="nearest",
            align_to_center=True
        )

        # Update origin before spacing, because computing the origin requires the original affine matrix.
        self.origin += np.dot(self.m_affine, np.transpose(grid_origin))

        # Update spacing and affine matrix.
        self.set_spacing(sample_spacing)

        # Round intensities in case of modalities with inherently discretised intensities
        if (self.modality == "CT") and (self.spat_transform == "base"):
            upd_voxel_grid = np.round(upd_voxel_grid)
        elif (self.modality == "PT") and (self.spat_transform == "base"):
            upd_voxel_grid[upd_voxel_grid < 0.0] = 0.0

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
        self.set_voxel_grid(voxel_grid=upd_voxel_grid)

        1

    def interpolate_missing_slices(self):

        from scipy.interpolate import Akima1DInterpolator

        # Get voxel grid
        voxel_grid = self.get_voxel_grid()

        # Find the desired slice positions.
        new_slice_positions = np.arange(0.0, self.slice_position[-1] + self.spacing[0], self.spacing[0])

        # Remove out-of-bound slices.
        new_slice_positions = new_slice_positions[new_slice_positions <= self.slice_position[-1]]

        # Create a voxel grid that is subsequently updated.
        updated_voxel_grid = np.zeros((len(new_slice_positions),
                                       self.size[1],
                                       self.size[2]))

        # Since we are interpolating slices, we can use stacked 1D interpolation.
        for ii in range(self.size[2]):
            for jj in range(self.size[1]):

                values = voxel_grid[:, jj, ii]
                points = np.array(self.slice_position)

                # Instantiate interpolator. We use the Akima interpolator as we want to maintain values of known points.
                interpolator = Akima1DInterpolator(x=points,
                                                   y=values)

                # Interpolate at new positions.
                updated_voxel_grid[:, jj, ii] = interpolator(new_slice_positions)

        # Reset slice positions.
        self.slice_position = None

        # Add updated voxelgrid.
        self.set_voxel_grid(voxel_grid=updated_voxel_grid)

    def add_noise(self, noise_level, noise_iter):
        """ Adds Gaussian noise to the image volume
         noise_level: standard deviation of image noise present """

        # Add noise iteration number
        self.noise_iter = noise_iter

        # Skip for missing images
        if self.is_missing:
            return

        # Skip for invalid noise levels
        if noise_level is None:
            return
        if np.isnan(noise_level) or noise_level < 0.0:
            return

        # Add Gaussian noise to image
        voxel_grid = self.get_voxel_grid()
        voxel_grid += np.random.normal(loc=0.0, scale=noise_level, size=self.size)

        # Check for corrections due to image modality
        if self.spat_transform == "base":

            # Round CT values to the nearest integer
            if self.modality == "CT":
                voxel_grid = np.round(a=voxel_grid, decimals=0)

            # Set minimum PT to 0.0
            if self.modality == "PT":
                voxel_grid[voxel_grid < 0.0] = 0.0

        # Set noise level in image
        self.noise = noise_level

        self.set_voxel_grid(voxel_grid=voxel_grid)

    def saturate(self, intensity_range, fill_value=None):
        """
        Saturate image intensities using an intensity range
        :param intensity_range: range of intensity values
        :param fill_value: fill value for out-of-range intensities. If None, the upper and lower ranges are used
        :return:
        """
        # Skip for missing images
        if self.is_missing:
            return

        intensity_range = np.array(copy.deepcopy(intensity_range))

        if np.any(~np.isnan(intensity_range)):
            # Get voxel grid
            voxel_grid = self.get_voxel_grid()

            # Lower boundary
            if not np.isnan(intensity_range[0]):
                if fill_value is None:
                    voxel_grid[voxel_grid < intensity_range[0]] = intensity_range[0]
                else:
                    voxel_grid[voxel_grid < intensity_range[0]] = fill_value[0]

            # Upper boundary
            if not np.isnan(intensity_range[1]):
                if fill_value is None:
                    voxel_grid[voxel_grid > intensity_range[1]] = intensity_range[1]
                else:
                    voxel_grid[voxel_grid > intensity_range[1]] = fill_value[1]

            # Set the updated voxel grid
            self.set_voxel_grid(voxel_grid=voxel_grid)

    def normalise_intensities(self, norm_method="none", intensity_range=None, saturation_range=None, mask=None):
        """
        Normalises image intensities
        :param norm_method: string defining the normalisation method. Should be one of "none", "range", "standardisation"
        :param intensity_range: range of intensities for normalisation
        :return:
        """

        # Skip for missing images
        if self.is_missing:
            return

        if intensity_range is None:
            intensity_range = [np.nan, np.nan]

        if mask is None:
            mask = np.ones(self.size, dtype=bool)
        else:
            mask = mask.astype(bool)

        if np.sum(mask) == 0:
            mask = np.ones(self.size, dtype=bool)

        if saturation_range is None:
            saturation_range = [np.nan, np.nan]

        if norm_method == "none":
            return

        elif norm_method == "range":
            # Normalisation to [0, 1] range using fixed intensities.

            # Get voxel grid
            voxel_grid = self.get_voxel_grid()

            # Find maximum and minimum intensities
            if np.isnan(intensity_range[0]):
                min_int = np.min(voxel_grid[mask])
            else:
                min_int = intensity_range[0]

            if np.isnan(intensity_range[1]):
                max_int = np.max(voxel_grid[mask])
            else:
                max_int = intensity_range[1]

            # Normalise by range
            if not max_int == min_int:
                voxel_grid = (voxel_grid - min_int) / (max_int - min_int)
            else:
                voxel_grid = voxel_grid - min_int

            # Update the voxel grid
            self.set_voxel_grid(voxel_grid=voxel_grid)

            self.is_normalised = True

        elif norm_method == "relative_range":
            # Normalisation to [0, 1]-ish range using relative intensities.

            # Get voxel grid
            voxel_grid = self.get_voxel_grid()

            min_int_rel = 0.0
            if not np.isnan(intensity_range[0]):
                min_int_rel = intensity_range[0]

            max_int_rel = 1.0
            if not np.isnan(intensity_range[1]):
                max_int_rel = intensity_range[1]

            # Compute minimum and maximum intensities.
            value_range = [np.min(voxel_grid[mask]), np.max(voxel_grid[mask])]
            min_int = value_range[0] + min_int_rel * (value_range[1] - value_range[0])
            max_int = value_range[0] + max_int_rel * (value_range[1] - value_range[0])

            # Normalise by range
            if not max_int == min_int:
                voxel_grid = (voxel_grid - min_int) / (max_int - min_int)
            else:
                voxel_grid = voxel_grid - min_int

            # Update the voxel grid
            self.set_voxel_grid(voxel_grid=voxel_grid)

            self.is_normalised = True

        elif norm_method == "quantile_range":
            # Normalisation to [0, 1]-ish range based on quantiles.

            # Get voxel grid
            voxel_grid = self.get_voxel_grid()

            min_quantile = 0.0
            if not np.isnan(intensity_range[0]):
                min_quantile = intensity_range[0]

            max_quantile = 1.0
            if not np.isnan(intensity_range[1]):
                max_quantile = intensity_range[1]

            # Compute quantiles from voxel grid.
            min_int = np.quantile(voxel_grid[mask], q=min_quantile)
            max_int = np.quantile(voxel_grid[mask], q=max_quantile)

            # Normalise by range
            if not max_int == min_int:
                voxel_grid = (voxel_grid - min_int) / (max_int - min_int)
            else:
                voxel_grid -= min_int

            # Update the voxel grid
            self.set_voxel_grid(voxel_grid=voxel_grid)

            self.is_normalised = True

        elif norm_method == "standardisation":
            # Normalisation to mean 0 and standard deviation 1.

            # Get voxel grid
            voxel_grid = self.get_voxel_grid()

            # Determine mean and standard deviation of the voxel intensities
            mean_int = np.mean(voxel_grid[mask])
            sd_int = np.std(voxel_grid[mask])

            # Protect against invariance.
            if sd_int == 0.0:
                sd_int = 1.0

            # Normalise
            voxel_grid = (voxel_grid - mean_int) / sd_int

            # Update the voxel grid
            self.set_voxel_grid(voxel_grid=voxel_grid)

            self.is_normalised = True
        else:
            raise ValueError(f"{norm_method} is not a valid method for normalising intensity values.")

        self.saturate(intensity_range=saturation_range)

    def crop(self,
             ind_ext_z=None,
             ind_ext_y=None,
             ind_ext_x=None,
             xy_only=False,
             z_only=False):
        """"Crop image to the provided map extent."""

        # Skip for missing images
        if self.is_missing:
            return

        # Determine corresponding voxel indices
        max_ind = np.ceil(np.array((np.max(ind_ext_z), np.max(ind_ext_y), np.max(ind_ext_x)))).astype(int)
        min_ind = np.floor(np.array((np.min(ind_ext_z), np.min(ind_ext_y), np.min(ind_ext_x)))).astype(int)

        # Set bounding indices
        max_bound_ind = np.minimum(max_ind, self.size).astype(int)
        min_bound_ind = np.maximum(min_ind, np.array([0, 0, 0])).astype(int)

        # Get voxel grid
        voxel_grid = self.get_voxel_grid()

        # Create corresponding image volumes by slicing original volume
        if z_only:
            voxel_grid = voxel_grid[min_bound_ind[0]:max_bound_ind[0] + 1, :, :]
            min_bound_ind[1] = 0
            min_bound_ind[2] = 0
        elif xy_only:
            voxel_grid = voxel_grid[:,
                                    min_bound_ind[1]:max_bound_ind[1] + 1,
                                    min_bound_ind[2]:max_bound_ind[2] + 1]
            min_bound_ind[0] = 0
            max_bound_ind[0] = self.size[0].astype(int)
        else:
            voxel_grid = voxel_grid[min_bound_ind[0]:max_bound_ind[0] + 1,
                                    min_bound_ind[1]:max_bound_ind[1] + 1,
                                    min_bound_ind[2]:max_bound_ind[2] + 1]

        # Update origin and z-slice position
        self.origin += np.dot(self.m_affine, np.transpose(min_bound_ind))

        # Update voxel grid
        self.set_voxel_grid(voxel_grid=voxel_grid)

    def crop_to_size(self, center, crop_size, xy_only=False):
        """Crop images to the exact size"""

        # Skip for missing images
        if self.is_missing:
            return

        # Make local copy
        crop_size = np.array(copy.deepcopy(crop_size))

        # Determine the new grid origin in the original index space. Only the dimensions with a number are updated
        grid_origin = np.round(center - crop_size / 2.0).astype(int)

        # Update grid origin and crop_size for the remainder of the calculation
        grid_origin[np.isnan(crop_size)] = 0
        crop_size[np.isnan(crop_size)] = self.size[np.isnan(crop_size)]

        # Determine coordinates of the box that can be copied in the original space
        max_ind_orig = grid_origin + crop_size
        min_ind_orig = grid_origin

        # Update coordinates based on boundaries in the original images
        max_ind_orig = np.minimum(max_ind_orig, self.size).astype(int)
        min_ind_orig = np.maximum(min_ind_orig, [0, 0, 0]).astype(int)

        # Determine coordinates where this box should land, i.e. perform the coordinate transformation to grid index space.
        max_ind_grid = max_ind_orig - grid_origin
        min_ind_grid = min_ind_orig - grid_origin

        # Create an empty voxel_grid to copy to
        cropped_grid = np.full(crop_size.astype(int), fill_value=np.nan)

        # Get slice of voxel grid
        voxel_grid = self.get_voxel_grid()[min_ind_orig[0]:max_ind_orig[0],
                                           min_ind_orig[1]:max_ind_orig[1],
                                           min_ind_orig[2]:max_ind_orig[2]]

        # Put the voxel grid slice into the cropped grid
        cropped_grid[min_ind_grid[0]:max_ind_grid[0], min_ind_grid[1]:max_ind_grid[1], min_ind_grid[2]:max_ind_grid[2]] = voxel_grid

        # Replace any remaining NaN values in the grid by the lowest intensity in voxel_grid
        cropped_grid[np.isnan(cropped_grid)] = np.min(voxel_grid)

        # Restore the original dtype in case it got lost
        cropped_grid = cropped_grid.astype(voxel_grid.dtype)

        # Update origin
        self.origin += np.dot(self.m_affine, np.transpose(grid_origin))

        # Set voxel grid
        self.set_voxel_grid(voxel_grid=cropped_grid)

    def set_spatial_transform(self, transform_method: str):

        if transform_method == "base":
            self.spat_transform = "base"

        else:
            self.spat_transform = transform_method
            self.as_parametric_map = True

    def compute_diagnostic_features(self, append_str: str = ""):
        """Creates diagnostic features for the image stack"""

        # Set feature names
        feat_names = ["img_dim_x", "img_dim_y", "img_dim_z", "vox_dim_x", "vox_dim_y", "vox_dim_z", "mean_int", "min_int", "max_int"]

        # Generate an initial table
        feature_table = pd.DataFrame(np.full(shape=(1, len(feat_names)), fill_value=np.nan))
        feature_table.columns = feat_names

        if not self.is_missing:

            # Update columns with actual values
            feature_table["img_dim_x"] = self.size[2]
            feature_table["img_dim_y"] = self.size[1]
            feature_table["img_dim_z"] = self.size[0]
            feature_table["vox_dim_x"] = self.spacing[2]
            feature_table["vox_dim_y"] = self.spacing[1]
            feature_table["vox_dim_z"] = self.spacing[0]
            feature_table["mean_int"] = np.mean(self.get_voxel_grid())
            feature_table["min_int"] = np.min(self.get_voxel_grid())
            feature_table["max_int"] = np.max(self.get_voxel_grid())

            # Update column names
        feature_table.columns = ["_".join(["diag", feature, append_str]).rstrip("_") for feature in feature_table.columns]

        return feature_table

    def export(self, file_path):
        """
        Exports the image to the requested directory
        :param file_path: directory to write image to
        :return:
        """

        # Skip if the image is missing
        if self.is_missing:
            return

        # Construct file name
        file_name = self.get_export_descriptor() + ".nii.gz"

        # Export image to file
        self.write(file_path=file_path, file_name=file_name)

    def get_export_descriptor(self):
        """
        Generates an image descriptor based on parameters of the image
        :return:
        """

        descr_list = []

        # Image name and spatial transformation
        if self.name is not None:
            descr_list += [self.name]

        if self.interpolated:
            # Interpolation
            descr_list += [self.interpolation_algorithm,
                           "x", str(self.spacing[2])[:5],
                           "y", str(self.spacing[1])[:5],
                           "z", str(self.spacing[0])[:5]]

        if self.rotation_angle is not None and self.rotation_angle != 0.0:
            # Rotation angle
            descr_list += ["rot", str(self.rotation_angle)[:5]]

        if self.translation is not None and not np.all(self.translation == 0.0):
            # Translation fraction
            descr_list += ["trans",
                           "x", str(self.translation[2])[:5],
                           "y", str(self.translation[1])[:5],
                           "z", str(self.translation[0])[:5]]

        if self.noise != -1.0:
            # Noise level
            descr_list += ["noise", str(self.noise)[:5], "iter", str(self.noise_iter)]

        if not self.spat_transform == "base":
            descr_list += [self.spat_transform]

        return "_".join(descr_list)

    def convert_to_itk(self):
        """Converts the image to an itk format"""
        import itk

        if self.is_missing:
            return None

        # Get image data type and set a valid data type that can be read by simple itk
        vox_dtype = self.dtype_name
        if vox_dtype in ["float16", "float64", "float80", "float96", "float128"]:
            cast_type = np.float32
        elif vox_dtype in ["bool"]:
            cast_type = np.uint8
        else:
            cast_type = self.get_voxel_grid().dtype

        # Convert image voxels to an itk format.
        itk_img = itk.GetImageFromArray(self.get_voxel_grid().astype(cast_type), is_vector=False)
        itk_img.SetOrigin(self.origin[::-1])
        itk_img.SetSpacing(self.spacing[::-1])
        itk_img.SetDirection(itk.matrix_from_array(np.reshape(np.ravel(self.orientation)[::-1], [3, 3])))

        return itk_img

    def write(self, file_path, file_name):
        """ Writes the image to a file """
        import os
        import itk

        # Check if path exists
        file_path = os.path.normpath(file_path)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # Add file and file name
        file_path = os.path.join(file_path, file_name)

        # Convert image to simple itk format
        itk_img = self.convert_to_itk()

        # Write to file using simple itk, and the image is not missing
        if itk_img is not None:
            itk.imwrite(itk_img, file_path)

    def write_dicom(self, file_path, file_name, bit_depth=16):
        if self.metadata is None:
            raise ValueError(f"Image slice cannot be written to DICOM as metadata is missing.")

        # Set pixeldata
        self.set_pixel_data(bit_depth=bit_depth)

        # Remove 0x0002 group data.
        for filemeta_tag in list(self.metadata.group_dataset(0x0002).keys()):
            del self.metadata[filemeta_tag]

        # Save to file
        self.metadata.save_as(filename=os.path.join(file_path, file_name),
                              write_like_original=False)

    def write_dicom_series(self, file_path, file_name="", bit_depth=16):
        if self.metadata is None:
            raise ValueError(f"Image cannot be written as a DICOM series as metadata is missing.")

        # Check if the write folder exists
        if not os.path.isdir(file_path):

            if os.path.isfile(file_path):
                # Check if the write folder is a file.
                raise IOError(f"{file_path} is an existing file, not a directory. No DICOM images were exported.")
            else:
                os.makedirs(file_path, exist_ok=True)

        if self.as_parametric_map:
            self._convert_to_parametric_map_iod()

        # Provide a new series UID
        self.set_metadata(tag=(0x0020, 0x000e), value=create_new_uid(dcm=self.metadata))

        if self.modality == "PT":
            # Update the number of slices attribute.
            self.set_metadata(tag=(0x0054, 0x0081), value=self.size[0])

        # Export per slice
        for ii in np.arange(self.size[0]):
            # Obtain the slice
            slice_obj: ImageClass = self.get_slices(slice_number=ii)[0]

            # Generate the file name
            slice_file_name = file_name + f"{ii:06d}" + ".dcm"

            # Set instance number
            slice_obj.set_metadata(tag=(0x0020, 0x0013), value=ii)

            # Update the SOP instance UID to avoid collisions
            slice_obj.set_metadata(tag=(0x0008, 0x0018), value=create_new_uid(dcm=slice_obj.metadata))

            # Update instance creation date and time
            slice_obj.set_metadata(tag=(0x0008, 0x0012), value=time.strftime("%Y%m%d"))
            slice_obj.set_metadata(tag=(0x0008, 0x0013), value=time.strftime("%H%M%S"))

            # Export
            slice_obj.write_dicom(file_path=file_path, file_name=slice_file_name, bit_depth=bit_depth)

    def get_slices(self, slice_number=None):

        img_obj_list = []

        # Create a copy of the current object
        base_img_obj = self.copy()

        # Remove attributes that need to be set
        base_img_obj.isEncoded_voxel_grid = None
        base_img_obj.voxel_grid = None
        base_img_obj.size = None
        base_img_obj.dtype_name = None

        if slice_number is None:

            voxel_grid = self.get_voxel_grid()

            # Iterate over slices
            for ii in np.arange(self.size[0]):
                slice_img_obj = copy.deepcopy(base_img_obj)

                # Update origin and slice position
                slice_img_obj.origin += np.dot(self.m_affine, np.array([ii, 0, 0]))

                # Update name
                if slice_img_obj.name is not None:
                    slice_img_obj.name += "_slice_" + str(ii)

                # Copy voxel grid without dropping dimensions.
                if voxel_grid is not None:
                    slice_img_obj.set_voxel_grid(voxel_grid=voxel_grid[ii:ii+1, :, :])

                # Add to list
                img_obj_list += [slice_img_obj]
        else:
            # Extract a single slice
            slice_img_obj = copy.deepcopy(base_img_obj)

            # Update origin and slice position
            slice_img_obj.origin += np.dot(self.m_affine, np.array([slice_number, 0, 0]))

            # Update name
            if slice_img_obj.name is not None:
                slice_img_obj.name += "_slice_" + str(slice_number)

            # Copy voxel grid without dropping dimensions.
            if not self.is_missing:
                slice_img_obj.set_voxel_grid(voxel_grid=self.get_voxel_grid()[slice_number:slice_number+1, :, :])

            # Add to list
            img_obj_list += [slice_img_obj]

        return img_obj_list

    def drop_image(self):
        """Drops image, e.g. to free up memory."""
        self.isEncoded_voxel_grid = None
        self.voxel_grid = None

    def drop_metadata(self):
        self.metadata = None

    def get_metadata(self, tag, tag_type, default=None):
        # Do not attempt to read the metadata if no metadata is present.
        if self.metadata is None:
            return

        return get_pydicom_meta_tag(dcm_seq=self.metadata, tag=tag, tag_type=tag_type, default=default)

    def set_metadata(self, tag, value, force_vr=None):

        # Do not update the metadata if no metadata is present.
        if self.metadata is None:
            return None

        set_pydicom_meta_tag(dcm_seq=self.metadata, tag=tag, value=value, force_vr=force_vr)

    def has_metadata(self, tag):

        if self.metadata is None:
            return None

        else:
            return get_pydicom_meta_tag(dcm_seq=self.metadata, tag=tag, test_tag=True)

    def delete_metadata(self, tag):
        if self.metadata is None:
            return None

        elif self.has_metadata(tag):
            del self.metadata[tag]

        else:
            pass

    def cond_set_metadata(self, tag, value, force_vr=None):
        if self.set_metadata is None:
            return None

        elif not self.has_metadata(tag):
            self.set_metadata(tag=tag, value=value, force_vr=force_vr)

        else:
            pass

    def update_image_plane_metadata(self):
        # Update pixel spacing, image orientation, image position, slice thickness, slice location, rows and columns based on the image object

        # Do not update the metadata if no metadata is present.
        if self.metadata is None:
            return

        # Pixel spacing
        pixel_spacing = [self.spacing[2], self.spacing[1]]
        self.set_metadata(tag=(0x0028, 0x0030), value=pixel_spacing)

        # Image orientation. The matrix needs to be flattened first prior to extracting the orientation in the 2D plane.
        image_orientation = np.ravel(self.orientation)[::-1][:6]
        self.set_metadata(tag=(0x0020, 0x0037), value=image_orientation)

        # Image position
        self.set_metadata(tag=(0x0020, 0x0032), value=self.origin[::-1])

        # Slice thickness
        self.set_metadata(tag=(0x0018, 0x0050), value=self.spacing[0])

        # Slice location
        self.set_metadata(tag=(0x0020, 0x1041), value=self.origin[0])

        # Rows (y)
        self.set_metadata(tag=(0x0028, 0x0010), value=self.size[1])

        # Columns (x)
        self.set_metadata(tag=(0x0028, 0x0011), value=self.size[2])

    def set_pixel_data(self, bit_depth=16):
        # Important tags to update:

        if self.metadata is None:
            return

        if self.size[0] > 1:
            warnings.warn("Cannot set pixel data for image with more than one slice.", UserWarning)
            return

        # Set samples per pixel
        self.set_metadata(tag=(0x0028, 0x0002), value=1)

        # Set photometric interpretation
        self.set_metadata(tag=(0x0028, 0x0004), value="MONOCHROME2")

        # Remove the Pixel Data Provider URL attribute
        self.delete_metadata(tag=(0x0028, 0x7fe0))

        # Determine how pixel data are stored.
        if self.as_parametric_map:
            self._set_pixel_data_float(bit_depth=16)

        else:
            self._set_pixel_data_int(bit_depth=16)

    def _set_pixel_data_int(self, bit_depth):

        # Set dtype for the image
        if bit_depth == 8:
            pixel_type = np.int8
        elif bit_depth == 16:
            pixel_type = np.int16
        elif bit_depth == 32:
            pixel_type = np.int32
        elif bit_depth == 64:
            pixel_type = np.int64
        else:
            raise ValueError(f"Bit depth of DICOM images should be one of 8, 16 (default), 32, or 64., Found: {bit_depth}")

        # Update metadata related to the image data
        self.update_image_plane_metadata()

        pixel_grid = np.squeeze(self.get_voxel_grid(), axis=0)

        # Always write 16-bit data
        self.set_metadata(tag=(0x0028, 0x0100), value=bit_depth)  # Bits allocated
        self.set_metadata(tag=(0x0028, 0x0101), value=bit_depth)  # Bits stored
        self.set_metadata(tag=(0x0028, 0x0102), value=bit_depth-1)  # High-bit
        self.set_metadata(tag=(0x0028, 0x0103), value=1)  # Pixel representation (we assume signed integers)

        # Standard settings for lowest and highest pixel value
        if self.modality == "CT":
            rescale_intercept = 0.0
            rescale_slope = 1.0

        elif self.modality == "PT":
            rescale_intercept = 0.0
            rescale_slope = np.max(pixel_grid) / (2 ** (bit_depth - 1) - 1)

        elif self.modality == "MR":
            rescale_intercept = 0.0
            rescale_slope = 1.0

        else:
            raise TypeError(f"Unknown modality {self.modality}")

        # Inverse rescaling prior to dicom storage
        pixel_grid = (pixel_grid - rescale_intercept) / rescale_slope

        # Convert back to int16
        pixel_grid = pixel_grid.astype(pixel_type)

        # Store to PixelData
        self.set_metadata(tag=(0x7fe0, 0x0010), value=pixel_grid.tobytes(), force_vr="OW")

        # Delete other PixelData containers
        self.delete_metadata(tag=(0x7fe0, 0x0008))  # Float pixel data
        self.delete_metadata(tag=(0x7fe0, 0x0009))  # Double float pixel data

        # Update rescale intercept and slope (in case of CT and PET only)
        if self.modality in ["CT", "PT"]:
            self.set_metadata(tag=(0x0028, 0x1052), value=rescale_intercept)
            self.set_metadata(tag=(0x0028, 0x1053), value=rescale_slope)

        # Remove elements of the VOI LUT module
        self.delete_metadata(tag=(0x0028, 0x3010))  # VOI LUT sequence
        self.delete_metadata(tag=(0x0028, 0x1050))  # Window center
        self.delete_metadata(tag=(0x0028, 0x1051))  # Window width
        self.delete_metadata(tag=(0x0028, 0x1055))  # Window center and width explanation
        self.delete_metadata(tag=(0x0028, 0x1056))  # VOI LUT function

        # Update smallest and largest image pixel value. Cannot set more than 16 bits due to limitations of the
        # tag.
        if bit_depth <= 16:
            self.set_metadata(tag=(0x0028, 0x0106), value=np.min(pixel_grid), force_vr="SS")
            self.set_metadata(tag=(0x0028, 0x0107), value=np.max(pixel_grid), force_vr="SS")

    def _set_pixel_data_float(self, bit_depth):

        if not self.as_parametric_map:
            raise ValueError(f"Floating point representation in DICOM is only supported by parametric maps, but the image in MIRP is not marked for conversion of the metadata to"
                             f"parametric maps.")

        # Set dtype for the image
        if bit_depth == 16:
            pixel_type = np.int16
        elif bit_depth == 32:
            pixel_type = np.float32
        elif bit_depth == 64:
            pixel_type = np.float64
        else:
            raise ValueError(f"Bit depth of floating point DICOM images should be 16, 32 (default), or 64. Found: {bit_depth}")

        # Set the number of allocated bits
        self.set_metadata(tag=(0x0028, 0x0100), value=bit_depth)  # Bits allocated

        # Update metadata related to the image data
        self.update_image_plane_metadata()

        # Get the pixel grid of the slice
        pixel_grid = np.squeeze(self.get_voxel_grid(), axis=0)

        # Define rescale intercept and slope
        if bit_depth == 16:

            # DICOM ranges
            dcm_range = float(2 ** bit_depth - 1)
            dcm_min = - float(2 ** (bit_depth - 1))
            dcm_max = float(2 ** (bit_depth - 1)) - 1.0

            # Data ranges
            value_min = np.min(pixel_grid)
            value_max = np.max(pixel_grid)
            value_range = np.max(pixel_grid) - np.min(pixel_grid)

            # Rescale slope and intercept
            rescale_slope = value_range / dcm_range
            rescale_intercept = (value_min * dcm_max - value_max * dcm_min) / dcm_range

            # Inverse rescaling prior to dicom storage
            pixel_grid = pixel_grid * dcm_range - value_min * dcm_max + value_max * dcm_min
            pixel_grid *= 1.0 / value_range

            # Round pixel values for safety
            pixel_grid = np.round(pixel_grid)
        else:

            rescale_intercept = 0.0
            rescale_slope = 1.0

        # Cast to the right data type
        pixel_grid = pixel_grid.astype(pixel_type)

        # Store pixel data to the right attribute
        if bit_depth == 16:
            # Store to Pixel Data attribute
            self.set_metadata(tag=(0x7fe0, 0x0010), value=pixel_grid.tobytes(), force_vr="OW")

            # Set Image Pixel module-specific tags
            self.set_metadata(tag=(0x0028, 0x0101), value=bit_depth)  # Bits stored
            self.set_metadata(tag=(0x0028, 0x0102), value=bit_depth - 1)  # High-bit
            self.set_metadata(tag=(0x0028, 0x0103), value=1)  # Pixel representation (we assume signed integers)

            # Update smallest and largest pixel value
            self.set_metadata(tag=(0x0028, 0x0106), value=np.min(pixel_grid), force_vr="SS")
            self.set_metadata(tag=(0x0028, 0x0107), value=np.max(pixel_grid), force_vr="SS")

            # Delete other PixelData containers
            self.delete_metadata(tag=(0x7fe0, 0x0008))  # Float pixel data
            self.delete_metadata(tag=(0x7fe0, 0x0009))  # Double float pixel data

        elif bit_depth == 32:
            # Store to Float Pixel Data attribute
            self.set_metadata(tag=(0x7fe0, 0x0008), value=np.ravel(pixel_grid).tolist(), force_vr="OF")

            # Delete other PixelData containers
            self.delete_metadata(tag=(0x7fe0, 0x0009))  # Double float pixel data
            self.delete_metadata(tag=(0x7fe0, 0x0010))  # Integer type data

        elif bit_depth == 64:
            # Store to Double Float Pixel Data attribute
            self.set_metadata(tag=(0x7fe0, 0x0009), value=np.ravel(pixel_grid).tolist(), force_vr="OD")

            # Delete other PixelData containers
            self.delete_metadata(tag=(0x7fe0, 0x0008))  # Float pixel data
            self.delete_metadata(tag=(0x7fe0, 0x0010))  # Integer type data

        # Reset rescale intercept and slope attributes to default values for parametric maps.
        self.set_metadata(tag=(0x0028, 0x1052), value=rescale_intercept)  # Rescale intercept
        self.set_metadata(tag=(0x0028, 0x1053), value=rescale_slope)  # Rescale slope

        # Remove elements of the VOI LUT module
        self.delete_metadata(tag=(0x0028, 0x3010))  # VOI LUT sequence
        self.delete_metadata(tag=(0x0028, 0x1050))  # Window center
        self.delete_metadata(tag=(0x0028, 0x1051))  # Window width
        self.delete_metadata(tag=(0x0028, 0x1055))  # Window center and width explanation
        self.delete_metadata(tag=(0x0028, 0x1056))  # VOI LUT function

        # Number of frames
        self.set_metadata(tag=(0x0028, 0x0008), value=1)

    def _convert_to_parametric_map_iod(self):

        if self.metadata is None:
            return None

        # Create a copy of the metadata.
        old_dcm: FileDataset = copy.deepcopy(self.metadata)

        # Update the SOP class to that of a parametric map image
        self.set_metadata(tag=(0x0008, 0x0016), value="1.2.840.10008.5.1.4.1.1.30")

        # Update the image type attribute
        image_type = self.get_metadata(tag=(0x0008, 0x0008), tag_type="mult_str", default=[])
        image_type = [image_type[ii] if ii < len(image_type) else "" for ii in range(4)]
        image_type[0] = "DERIVED"
        image_type[1] = "PRIMARY"
        image_type[2] = image_type[2] if not image_type[2] == "" else "STATIC"
        image_type[3] = "MIXED" if self.spat_transform == "base" else "FILTERED"

        self.set_metadata(tag=(0x0008, 0x0008), value=image_type)

        # Parametric Map Image module attributes that may be missing.
        self.cond_set_metadata(tag=(0x2050, 0x0020), value="IDENTITY")  # Presentation LUT shape
        self.cond_set_metadata(tag=(0x0018, 0x9004), value="RESEARCH")  # Content qualification
        self.cond_set_metadata(tag=(0x0028, 0x0301), value="NO")  # Burned-in Annotation
        self.cond_set_metadata(tag=(0x0028, 0x0302), value="YES")  # Recognisable facial features
        self.cond_set_metadata(tag=(0x0070, 0x0080), value=self.get_export_descriptor().upper().strip()[:15])  # Content label
        self.cond_set_metadata(tag=(0x0070, 0x0081), value=self.get_export_descriptor()[:63])  # Content description
        self.cond_set_metadata(tag=(0x0070, 0x0084), value="Doe^John")

        # Set the source instance sequence
        source_instance_list = []
        for reference_instance_sop_uid in self.slice_table.sop_instance_uid:
            ref_inst = Dataset()
            set_pydicom_meta_tag(dcm_seq=ref_inst, tag=(0x0008, 0x1150), value=get_pydicom_meta_tag(dcm_seq=old_dcm, tag=(0x0008, 0x0016), tag_type="str"))
            set_pydicom_meta_tag(dcm_seq=ref_inst, tag=(0x0008, 0x1155), value=reference_instance_sop_uid)

            source_instance_list += [ref_inst]

        self.set_metadata(tag=(0x0008, 0x2112), value=Sequence(source_instance_list))

        # Attributes from the enhanced general equipment module may be missing.
        self.cond_set_metadata(tag=(0x0008, 0x0070), value="unknown")  # Manufacturer
        self.cond_set_metadata(tag=(0x0008, 0x1090), value="unknown")  # Model name
        self.cond_set_metadata(tag=(0x0018, 0x1000), value="unknown")  # Device Serial Number
        self.set_metadata(tag=(0x0018, 0x1020), value="MIRP " + get_version())

        # Items from multi-frame function groups may be missing. We currently only use a single frame.
        self.set_metadata(tag=(0x5200, 0x9229), value=Sequence())  # Shared functional groups sequence
        self.set_metadata(tag=(0x5200, 0x9230), value=Sequence())  # Per-frame functional groups sequence

        # Multi-frame Dimension module

        # Dimension organisation sequence. We copy the frame of reference as UID.
        dim_org_seq_elem = Dataset()
        set_pydicom_meta_tag(dim_org_seq_elem, tag=(0x0020, 0x9164), value=self.get_metadata(tag=(0x0020, 0x0052), tag_type="str"))  # Dimension organisation UID
        self.set_metadata(tag=(0x0020, 0x9221), value=Sequence([dim_org_seq_elem]))

        # Dimension Index sequence. We point to the instance number.
        dim_index_seq_elem = Dataset()
        set_pydicom_meta_tag(dim_index_seq_elem, tag=(0x0020, 0x9165), value=(0x0020, 0x0013))  # Dimension index pointer
        set_pydicom_meta_tag(dim_index_seq_elem,
                             tag=(0x0020, 0x9164),
                             value=self.get_metadata(tag=(0x0020, 0x0052), tag_type="str"))
        self.set_metadata(tag=(0x0020, 0x9222), value=Sequence([dim_index_seq_elem]))
