import copy

import numpy as np
import pandas as pd


class ImageClass:
    # Class for image volumes

    def __init__(self, voxel_grid, origin, slice_z_pos, spacing, orientation, modality=None, spat_transform="base", no_image=False):
        self.origin   = origin    # Coordinates of [0,0,0] voxel in mm
        self.spacing  = spacing   # Voxel spacing in mm
        self.slice_z_pos = np.array(slice_z_pos)    # Position along stack axis
        self.modality = modality                    # Imaging modality
        self.spat_transform = spat_transform        # Signifies whether the current image is a base image or not
        self.orientation = orientation

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
        self.transl_fraction_x = 0.0
        self.transl_fraction_y = 0.0
        self.transl_fraction_z = 0.0

        # Rotation parameters
        self.rotation_angle = 0.0

        # Set voxel grid and image
        if not no_image:
            self.is_missing = False
            self.set_voxel_grid(voxel_grid=voxel_grid)
        else:
            self.is_missing = True

    def copy(self, drop_image=False):
        # Creates a new copy of the image object
        img_copy = copy.deepcopy(self)

        if drop_image:
            img_copy.drop_image()

        return img_copy

    def show(self, img_slice):
        import pylab

        if self.is_missing:
            return

        pylab.imshow(self.get_voxel_grid()[img_slice, :, :], cmap=pylab.cm.bone)
        pylab.show()

    def set_voxel_grid(self, voxel_grid):
        """ Sets voxel grid """

        # Determine size
        self.size  = np.array(voxel_grid.shape)
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
            decoded_voxel = np.zeros(np.prod(self.size), dtype=np.bool)

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
            rle_end = np.array(np.append(np.where(voxel_grid.ravel()[1:] != voxel_grid.ravel()[:-1]), np.prod(self.size) - 1))
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
            decoded_voxel = np.zeros(np.prod(self.size), dtype=np.bool)

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

    def interpolate(self, by_slice, settings):
        """Performs interpolation of the image volume"""
        from mirp.imageProcess import interpolate_to_new_grid, gaussian_preprocess_filter

        # Skip for missing images
        if self.is_missing:
            return

        # Local interpolation constants
        if None not in settings.img_interpolate.new_spacing:
            iso_spacing = settings.img_interpolate.new_spacing[0]
            new_spacing = np.array([iso_spacing, iso_spacing, iso_spacing])  # Desired spacing in mm
        elif type(settings.img_interpolate.new_non_iso_spacing) in [list, tuple]:
            if None not in settings.img_interpolate.new_non_iso_spacing:
                non_iso_spacing = settings.img_interpolate.new_non_iso_spacing
                new_spacing = np.array(non_iso_spacing)
            else:
                new_spacing = self.spacing
        else:
            new_spacing = self.spacing

        # Read additional details
        order            = settings.img_interpolate.spline_order   # Order of multidimensional spline filter (0=nearest neighbours, 1=linear, 3=cubic)
        interpolate_flag = settings.img_interpolate.interpolate    # Whether to interpolate or not

        # Set spacing for interpolation across slices to the original spacing in case interpolation is only conducted within the slice
        if by_slice:    new_spacing[0] = self.spacing[0]

        # Image translation
        translate_z = settings.vol_adapt.translate_z[0]
        translate_y = settings.vol_adapt.translate_y[0]
        translate_x = settings.vol_adapt.translate_x[0]

        # Convert to [0.0, 1.0] range
        translate_x = translate_x - np.floor(translate_x)
        translate_y = translate_y - np.floor(translate_y)
        translate_z = translate_z - np.floor(translate_z)
        trans_vec = np.array([translate_z, translate_y, translate_x])

        # Add translation fractions
        self.transl_fraction_x = translate_x
        self.transl_fraction_y = translate_y
        self.transl_fraction_z = translate_z

        # Skip if translation in both directions is 0.0
        if translate_x == 0.0 and translate_y == 0.0 and translate_z == 0.0  and not interpolate_flag: return None

        # Check if pre-processing is required
        if settings.img_interpolate.anti_aliasing:
            self.set_voxel_grid(voxel_grid=gaussian_preprocess_filter(orig_vox=self.get_voxel_grid(), orig_spacing=self.spacing, sample_spacing=new_spacing,
                                                                      param_beta=settings.img_interpolate.smoothing_beta, mode="nearest", by_slice=by_slice))

        # Interpolate image and positioning
        self.size, self.origin, self.spacing, upd_voxel_grid = \
            interpolate_to_new_grid(orig_dim=self.size, orig_origin=self.origin, orig_spacing=self.spacing, orig_vox=self.get_voxel_grid(),
                                    sample_spacing=new_spacing, translation=trans_vec, order=order, mode="nearest", align_to_center=True)

        # Round intensities in case of modalities with inherently discretised intensities
        if (self.modality == "CT") and (self.spat_transform == "base"):
            upd_voxel_grid = np.round(upd_voxel_grid)
        elif (self.modality == "PT") and (self.spat_transform == "base"):
            upd_voxel_grid[upd_voxel_grid < 0.0] = 0.0

        # Update slice z positions
        self.slice_z_pos = self.origin[0] + np.arange(start=0, stop=self.size[0]) * self.spacing[0]

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

    def saturate(self, intensity_range, fill_value=np.nan):
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
                    voxel_grid[voxel_grid < intensity_range[0]] = fill_value

            # Upper boundary
            if not np.isnan(intensity_range[1]):
                if fill_value is None:
                    voxel_grid[voxel_grid > intensity_range[1]] = intensity_range[1]
                else:
                    voxel_grid[voxel_grid > intensity_range[1]] = fill_value

            # Set the updated voxel grid
            self.set_voxel_grid(voxel_grid=voxel_grid)

    def normalise_intensities(self, norm_method="none", intensity_range=[np.nan, np.nan]):
        """
        Normalises image intensities
        :param norm_method: string defining the normalisation method. Should be one of "none", "range", "standardisation"
        :intensity_range: range of intensities for normalisation
        :return:
        """

        # Skip for missing images
        if self.is_missing:
            return

        if norm_method == "none":
            return
        elif norm_method == "range":

            # Get voxel grid
            voxel_grid = self.get_voxel_grid()

            # Find maximum and minimum intensities
            if np.isnan(intensity_range[0]):
                min_int = np.min(voxel_grid)
            else:
                min_int = intensity_range[0]

            if np.isnan(intensity_range[1]):
                max_int = np.max(voxel_grid)
            else:
                max_int = intensity_range[1]

            # Normalise by range
            voxel_grid = (voxel_grid - min_int) / (max_int - min_int)

            # Update the voxel grid
            self.set_voxel_grid(voxel_grid=voxel_grid)

        elif norm_method == "standardisation":

            # Get voxel grid
            voxel_grid = self.get_voxel_grid()

            # Determine mean and standard deviation of the voxel intensities
            mean_int = np.mean(voxel_grid)
            sd_int = np.std(voxel_grid)

            # Normalise
            voxel_grid = (voxel_grid - mean_int) / sd_int

            # Update the voxel grid
            self.set_voxel_grid(voxel_grid=voxel_grid)

        else:
            raise ValueError("\"%s\" is not a valid method for normalising intensity values.", norm_method)

    def rotate(self, angle):
        """Rotate volume along z-axis."""

        # Skip for missing images
        if self.is_missing:
            return

        import scipy.ndimage as ndi
        from mirp.featureSets.volumeMorphology import get_rotation_matrix

        # Find actual output size of x-y plane
        new_z_dim = np.asmatrix([self.size[0], 0.0, 0.0]) * get_rotation_matrix(np.radians(angle), dim=3, rot_axis=0)
        new_y_dim = np.asmatrix([0.0, self.size[1], 0.0]) * get_rotation_matrix(np.radians(angle), dim=3, rot_axis=0)
        new_x_dim = np.asmatrix([0.0, 0.0, self.size[2]]) * get_rotation_matrix(np.radians(angle), dim=3, rot_axis=0)
        new_dim_flt = np.squeeze(np.array(np.abs(new_z_dim)) + np.array(np.abs(new_y_dim) + np.abs(new_x_dim)))

        # Get voxel grid
        voxel_grid = self.get_voxel_grid()

        # Rotate voxels along angle in the y-x plane and find truncated output size
        voxel_grid = ndi.rotate(voxel_grid.astype(np.float32), angle=angle, axes=(1, 2), reshape=True, order=1, mode="nearest")
        new_dim_int = np.array(np.shape(voxel_grid)) * 1.0

        if (self.modality == "CT") and (self.spat_transform == "base"):
            voxel_grid = np.round(voxel_grid)

        # Update spacing
        self.spacing *= new_dim_int / new_dim_flt

        # Set rotation angle
        self.rotation_angle = angle

        # Update voxel grid with rotated voxels
        self.set_voxel_grid(voxel_grid=voxel_grid)

    def translate(self, t_x=0.0, t_y=0.0, t_z=0.0):
        """Translate image volume"""
        from mirp.imageProcess import interpolate_to_new_grid

        # Skip for missing images
        if self.is_missing:
            return

        # Calculate the new sample origin after translation
        sample_origin = np.array(self.origin)
        sample_origin[0] += t_z * self.spacing[0]
        sample_origin[1] += t_y * self.spacing[1]
        sample_origin[2] += t_x * self.spacing[2]

        # Interpolate at shift points
        self.size, self.origin, self.spacing, upd_voxel_grid = \
            interpolate_to_new_grid(orig_dim=self.size, orig_origin=self.origin, orig_spacing=self.spacing, orig_vox=self.get_voxel_grid(),
                                    sample_dim=self.size, sample_origin=sample_origin, sample_spacing=self.spacing, order=1, mode="nearest")

        # Update voxel grid
        self.set_voxel_grid(voxel_grid=upd_voxel_grid)

    def crop(self, map_ext_z, map_ext_y, map_ext_x, xy_only=False, z_only=False):
        """"Crop image to the provided map extent."""
        from mirp.utilities import world_to_index

        # Skip for missing images
        if self.is_missing:
            return

        # Determine map extent in normalised image space
        ind_ext_z = np.around(world_to_index(map_ext_z, self.origin[0], self.spacing[0]), 5)
        ind_ext_y = np.around(world_to_index(map_ext_y, self.origin[1], self.spacing[1]), 5)
        ind_ext_x = np.around(world_to_index(map_ext_x, self.origin[2], self.spacing[2]), 5)

        # Determine corresponding voxel indices
        max_ind = np.ceil(np.array((np.max(ind_ext_z), np.max(ind_ext_y), np.max(ind_ext_x)))).astype(np.int)
        min_ind = np.floor(np.array((np.min(ind_ext_z), np.min(ind_ext_y), np.min(ind_ext_x)))).astype(np.int)

        # Set bounding indices
        max_bound_ind = np.minimum(max_ind, self.size).astype(np.int)
        min_bound_ind = np.maximum(min_ind, [0, 0, 0]).astype(np.int)

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
            max_bound_ind[0] = self.size[0].astype(np.int)
        else:
            voxel_grid = voxel_grid[min_bound_ind[0]:max_bound_ind[0] + 1,
                                    min_bound_ind[1]:max_bound_ind[1] + 1,
                                    min_bound_ind[2]:max_bound_ind[2] + 1]

        # Update origin and z-slice position
        self.origin = self.origin + np.multiply(min_bound_ind, self.spacing)
        self.slice_z_pos = self.origin[0] + np.arange(0, max_bound_ind[0]-min_bound_ind[0]+1) * self.spacing[0]

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
        grid_origin = np.round(center - crop_size / 2.0).astype(np.int)

        # Update grid origin and crop_size for the remainder of the calculation
        grid_origin[np.isnan(crop_size)] = 0
        crop_size[np.isnan(crop_size)] = self.size[np.isnan(crop_size)]

        # Determine coordinates of the box that can be copied in the original space
        max_ind_orig = grid_origin + crop_size
        min_ind_orig = grid_origin

        # Update coordinates based on boundaries in the original images
        max_ind_orig = np.minimum(max_ind_orig, self.size).astype(np.int)
        min_ind_orig = np.maximum(min_ind_orig, [0, 0, 0]).astype(np.int)

        # Determine coordinates where this box should land, i.e. perform the coordinate transformation to grid index space.
        max_ind_grid = max_ind_orig - grid_origin
        min_ind_grid = min_ind_orig - grid_origin

        # Create an empty voxel_grid to copy to
        cropped_grid = np.full(crop_size.astype(np.int), fill_value=np.nan)

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

        # Update origin and slice positions
        self.origin = self.origin + np.multiply(grid_origin, self.spacing)
        self.slice_z_pos = self.origin[0] + np.arange(crop_size[0]) * self.spacing[0]

        # Set voxel grid
        self.set_voxel_grid(voxel_grid=cropped_grid)

    def compute_diagnostic_features(self, append_str=""):
        """Creates diagnostic features for the image stack"""

        # Set feature names
        feat_names = ["img_dim_x", "img_dim_y", "img_dim_z", "vox_dim_x", "vox_dim_y", "vox_dim_z", "mean_int", "min_int", "max_int"]

        # Create pandas dataframe with one row and feature columns
        df = pd.DataFrame(np.full(shape=(1, len(feat_names)), fill_value=np.nan))
        df.columns = feat_names

        # Return prototype if the image is missing
        if self.is_missing:
            return df

        # Set features
        df.ix[0, ["img_dim_z", "img_dim_y", "img_dim_x"]] = self.size     # Image dimensions (in voxels)
        df.ix[0, ["vox_dim_z", "vox_dim_y", "vox_dim_x"]] = self.spacing  # Voxel dimensions (in mm)
        df.ix[0, "mean_int"] = np.mean(self.get_voxel_grid())               # Mean image intensity
        df.ix[0, "min_int"] = np.min(self.get_voxel_grid())                 # Minimum image intensity
        df.ix[0, "max_int"] = np.max(self.get_voxel_grid())                 # Maximum image intensity

        # Update column names
        df.columns = ["diag_" + feat + append_str for feat in df.columns]

        return df

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
        img_append_str = ""

        # Image name and spatial transformation
        if self.name is not None:
            img_append_str += self.name + "_" + self.spat_transform

            # Interpolation
            img_append_str += "_intx" + str(self.spacing[2]) + "y" + str(self.spacing[1]) + "z" + str(self.spacing[2])

        if self.rotation_angle != 0.0:
            # Rotation angle
            img_append_str += "_rot" + str(self.rotation_angle)
        if not (self.transl_fraction_x == 0.0 and self.transl_fraction_y == 0.0 and self.transl_fraction_z == 0.0):
            # Translation fraction
            img_append_str += "_trx" + str(self.transl_fraction_x) + "y" + str(self.transl_fraction_y) + str(self.transl_fraction_z)
        if self.noise != -1.0:
            # Noise level
            img_append_str += "_noise_lvl" + str(self.noise) + "_iter" + str(self.noise_iter)

        return img_append_str

    def convert2sitk(self):
        """Converts image object back to Simple ITK
        This step may precede writing to file."""

        import SimpleITK as sitk

        # Skip if the image is missing
        if self.is_missing:
            return None

        # Get image data type and set a valid data type that can be read by simple itk
        vox_dtype = self.dtype_name
        if vox_dtype in ["float16", "float64", "float80", "float96", "float128"]: cast_type = np.float32
        elif vox_dtype in ["bool"]: cast_type = np.uint8
        else: cast_type = self.get_voxel_grid().dtype

        # Convert image voxels
        sitk_img = sitk.GetImageFromArray(self.get_voxel_grid().astype(cast_type), isVector=False)
        sitk_img.SetOrigin(self.origin[::-1])
        sitk_img.SetSpacing(self.spacing[::-1])
        sitk_img.SetDirection(self.orientation[::-1])

        return sitk_img

    def write(self, file_path, file_name):
        """ Writes the image to a file """
        import SimpleITK as sitk
        import os

        # Check if path exists
        file_path = os.path.normpath(file_path)
        if not os.path.exists(file_path): os.makedirs(file_path)

        # Add file and file name
        file_path = os.path.join(file_path, file_name)

        # Convert image to simple itk format
        sitk_img = self.convert2sitk()

        # Write to file using simple itk, and the image is not missing
        if sitk_img is not None:
            sitk.WriteImage(sitk_img, file_path, True)

    def get_slices(self, slice_number=None):

        img_obj_list = []

        # Create a copy of the current object
        base_img_obj = self.copy()

        # Remove attributes that need to be set
        base_img_obj.slice_z_pos = None
        base_img_obj.origin = None
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
                slice_img_obj.origin = self.origin + ii * self.spacing[0]
                slice_img_obj.slice_z_pos = np.array(self.origin[0])

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
            slice_img_obj.origin = self.origin + slice_number * self.spacing[0]
            slice_img_obj.slice_z_pos = np.array(self.origin[0])

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
        """Drops image, e.g. to free up memory. We don't set the is_m"""
        self.isEncoded_voxel_grid = None
        self.voxel_grid = None
