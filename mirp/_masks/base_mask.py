import warnings

import numpy as np
import pandas as pd
import copy
import sys
from typing import Any
from pathlib import Path

from mirp._images.generic_image import GenericImage
from mirp._images.mask_image import MaskImage
from mirp.settings.generic import SettingsClass

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class BaseMask:
    def __init__(
            self,
            roi_name: str,
            **kwargs
    ):
        # Make cooperative.
        super().__init__()

        # Set region of interest.
        self.roi = MaskImage(**kwargs)

        # Define other types of masks.
        self.roi_intensity: None | MaskImage = None
        self.roi_morphology: None | MaskImage = None

        # Set name of the mask.
        self.roi_name: str | list[str] = roi_name

        # Set intensity range.
        self.intensity_range: tuple[Any, Any] = tuple([np.nan, np.nan])

    def get_slices(
            self,
            slice_number: None | int | list[int] = None,
            primary_mask_only: bool = False
    ) -> None | Self | list[Self]:

        mask_list = []
        return_list = True

        if slice_number is None:
            slice_number = list(range(self.roi.image_dimension[0]))
        elif isinstance(slice_number, int):
            return_list = False
            slice_number = [slice_number]

        for current_slice_id in slice_number:
            slice_mask = self.copy(drop_image=True)
            slice_mask.roi = self.roi.get_slices(slice_number=current_slice_id)
            if slice_mask.roi_intensity is not None and not primary_mask_only:
                slice_mask.roi_intensity = self.roi_intensity.get_slices(slice_number=current_slice_id)
            else:
                slice_mask.roi_intensity = None

            if slice_mask.roi_morphology is not None and not primary_mask_only:
                slice_mask.roi_morphology = self.roi_morphology.get_slices(slice_number=current_slice_id)
            else:
                slice_mask.roi_morphology = None

            if slice_mask.is_empty():
                continue

            mask_list += [slice_mask]

        if len(mask_list) == 0:
            return None
        elif return_list:
            return mask_list
        else:
            return mask_list[0]

    def copy(self, drop_image=False) -> Self:

        # Create new mask by copying the current mask.
        mask = copy.deepcopy(self)

        if drop_image:
            mask.roi.drop_image()
            if mask.roi_intensity is not None:
                mask.roi_intensity.drop_image()
            if mask.roi_morphology is not None:
                mask.roi_morphology.drop_image()

        # Creates a new copy of the roi
        return mask

    def is_empty(self) -> bool:
        if self.roi is None:
            return True

        return self.roi.is_empty()

    def is_empty_mask(self):
        return self.roi.is_empty_mask()

    def interpolate(
            self,
            image: None | GenericImage,
            settings: SettingsClass
    ):
        # Skip if image and/or mask is missing
        if self.is_empty():
            return

        if image is None or image.is_empty():
            self.roi.interpolate(settings=settings)
            if self.roi_intensity is not None:
                self.roi_intensity.interpolate(settings=settings)
            if self.roi_morphology is not None:
                self.roi_morphology.interpolate(settings=settings)
        else:
            self.register(image=image, settings=settings)

    def register(
            self,
            image: GenericImage,
            spline_order: None | int = None,
            anti_aliasing: None | bool = None,
            anti_aliasing_smoothing_beta: None | float = None,
            settings: None | SettingsClass = None
    ):
        if (spline_order is None or anti_aliasing is None or anti_aliasing is None) and settings is None:
            raise ValueError("None of the parameters for registration can be set.")

        # Check if there is any mask data to work with.
        if self.is_empty():
            return

        # Check if the mask is empty.
        empty_before_registration = self.is_empty_mask()

        if spline_order is None:
            spline_order = self.roi.get_interpolation_spline_order(settings=settings)
        if anti_aliasing is None:
            anti_aliasing = settings.img_interpolate.anti_aliasing
        if anti_aliasing_smoothing_beta is None:
            anti_aliasing_smoothing_beta = settings.img_interpolate.smoothing_beta

        self.roi.register(
            image=image,
            spline_order=spline_order,
            anti_aliasing=anti_aliasing,
            anti_aliasing_smoothing_beta=anti_aliasing_smoothing_beta,
            mode="constant"
        )

        # Warn if a previously non-empty mask is empty after registration. This may indicate issues with the frame of
        # reference, i.e. world coordinate systems that do not have the same definition.
        if self.is_empty_mask() and not empty_before_registration:
            warnings.warn(
                f"The {self.roi_name} mask is empty after registering it to its image. Please check that mask and "
                f"image use the same frame of reference.",
                UserWarning
            )

        if self.roi_intensity is not None:
            self.roi_intensity.register(
                image=image,
                spline_order=spline_order,
                anti_aliasing=anti_aliasing,
                anti_aliasing_smoothing_beta=anti_aliasing_smoothing_beta,
                mode="constant"
            )
        if self.roi_morphology is not None:
            self.roi_morphology.register(
                image=image,
                spline_order=spline_order,
                anti_aliasing=anti_aliasing,
                anti_aliasing_smoothing_beta=anti_aliasing_smoothing_beta,
                mode="constant"
            )

    def merge(self, masks: list[Self]) -> Self:
        """Merge masks"""

        roi_mask = np.zeros(self.roi.image_dimension, dtype=bool)
        roi_name = []
        for mask in masks:
            # Skip empty masks.
            if mask.roi.is_empty():
                continue

            roi_mask = np.logical_or(roi_mask, mask.roi.get_voxel_grid())
            roi_name += [mask.roi_name]

        self.roi.set_voxel_grid(voxel_grid=roi_mask)
        self.roi_name = " + ".join(roi_name)

        if self.roi_intensity is not None:
            roi_mask = np.zeros(self.roi_intensity.image_dimension, dtype=bool)
            for mask in masks:
                # Skip empty masks.
                if mask.roi_intensity.is_empty():
                    continue

                roi_mask = np.logical_or(roi_mask, mask.roi_intensity.get_voxel_grid())

            self.roi_intensity.set_voxel_grid(voxel_grid=roi_mask)

        if self.roi_morphology is not None:
            roi_mask = np.zeros(self.roi_morphology.image_dimension, dtype=bool)
            for mask in masks:
                # Skip empty masks.
                if mask.roi_morphology.is_empty():
                    continue

                roi_mask = np.logical_or(roi_mask, mask.roi_morphology.get_voxel_grid())

            self.roi_morphology.set_voxel_grid(voxel_grid=roi_mask)

        return self

    def split_mask(self) -> list[Self]:
        """Split mask into multiple masks."""
        import skimage.measure

        if self.is_empty():
            return [self]

        # Label regions
        roi_label_mask, n_regions = skimage.measure.label(self.roi.get_voxel_grid(), connectivity=2, return_num=True)

        if n_regions == 1:
            return [self]

        new_masks = []
        for ii in np.arange(start=0, stop=n_regions):
            roi_mask = roi_label_mask == ii + 1

            new_mask = self.copy(drop_image=True)
            new_mask.roi.set_voxel_grid(roi_mask)
            if self.roi_intensity is not None:
                new_mask.roi_intensity.set_voxel_grid(voxel_grid=np.logical_and(roi_mask, self.roi_intensity.get_voxel_grid()))
            if self.roi_morphology is not None:
                new_mask.roi_morphology.set_voxel_grid(voxel_grid=np.logical_and(roi_mask, self.roi_morphology.get_voxel_grid()))

            new_mask.roi_name += f"-{ii + 1}"

            new_masks += [new_mask]

        return new_masks

    def select_largest_slice(self):
        """Crops to the largest slice."""

        if self.is_empty():
            return

        # Find axial slice that contains the largest part of the mask.
        roi_size = np.sum(self.roi.get_voxel_grid(), axis=(1, 2))
        if np.all(roi_size == 0):
            return

        # Find the index of said slice
        largest_slice_index = np.argmax(roi_size)

        # Copy only largest slice.
        roi_mask = np.zeros(self.roi.image_dimension, dtype=bool)
        roi_mask[largest_slice_index, :, :] = self.roi.get_voxel_grid()[largest_slice_index, :, :]
        self.roi.set_voxel_grid(voxel_grid=roi_mask)

        if self.roi_intensity is not None:
            roi_mask = np.zeros(self.roi_intensity.image_dimension, dtype=bool)
            roi_mask[largest_slice_index, :, :] = self.roi_intensity.get_voxel_grid()[largest_slice_index, :, :]
            self.roi_intensity.set_voxel_grid(voxel_grid=roi_mask)

        if self.roi_morphology is not None:
            roi_mask = np.zeros(self.roi_morphology.image_dimension, dtype=bool)
            roi_mask[largest_slice_index, :, :] = self.roi_morphology.get_voxel_grid()[largest_slice_index, :, :]
            self.roi_morphology.set_voxel_grid(voxel_grid=roi_mask)

    def select_largest_region(self):
        """Crops to the largest region."""
        import skimage.measure

        if self.is_empty():
            return

        # Label regions
        roi_label_mask, n_regions = skimage.measure.label(self.roi.get_voxel_grid(), connectivity=2, return_num=True)

        # Determine size of regions
        roi_sizes = np.zeros(n_regions)
        for ii in np.arange(start=0, stop=n_regions):
            roi_sizes[ii] = np.sum(roi_label_mask == ii + 1)

        # Select largest region
        roi_mask = roi_label_mask == np.argmax(roi_sizes) + 1
        self.roi.set_voxel_grid(voxel_grid=roi_mask)

        if self.roi_intensity is not None:
            self.roi_intensity.set_voxel_grid(voxel_grid=np.logical_and(self.roi_intensity.get_voxel_grid(), roi_mask))
        if self.roi_morphology is not None:
            self.roi_morphology.set_voxel_grid(voxel_grid=np.logical_and(self.roi_morphology.get_voxel_grid(), roi_mask))

    def generate_masks(self):
        """"Generate roi intensity and morphology masks"""

        if self.roi is None:
            self.roi_intensity = None
            self.roi_morphology = None
        else:
            if self.roi_intensity is None:
                self.roi_intensity = self.roi.copy()
            if self.roi_morphology is None:
                self.roi_morphology = self.roi.copy()

    def decimate(self, by_slice):
        """
        Decimates the roi mask.
        :param by_slice: boolean, 2D (True) or 3D (False)
        :return:
        """
        if self.roi is not None:
            self.roi.decimate(by_slice=by_slice)
        if self.roi_intensity is not None:
            self.roi_intensity.decimate(by_slice=by_slice)
        if self.roi_morphology is not None:
            self.roi_morphology.decimate(by_slice=by_slice)

    def crop(
            self,
            ind_ext_z=None,
            ind_ext_y=None,
            ind_ext_x=None,
            xy_only=False,
            z_only=False):

        # Crop masks.
        if self.roi is not None:
            self.roi.crop(
                ind_ext_z=ind_ext_z,
                ind_ext_y=ind_ext_y,
                ind_ext_x=ind_ext_x,
                xy_only=xy_only,
                z_only=z_only)

        if self.roi_intensity is not None:
            self.roi_intensity.crop(
                ind_ext_z=ind_ext_z,
                ind_ext_y=ind_ext_y,
                ind_ext_x=ind_ext_x,
                xy_only=xy_only,
                z_only=z_only)

        if self.roi_morphology is not None:
            self.roi_morphology.crop(
                ind_ext_z=ind_ext_z,
                ind_ext_y=ind_ext_y,
                ind_ext_x=ind_ext_x,
                xy_only=xy_only,
                z_only=z_only)

    def crop_to_size(self, center, crop_size):
        """"Crops roi to a pre-defined size"""

        # Crop masks to size
        if self.roi is not None:
            self.roi.crop_to_size(center=center, crop_size=crop_size)
        if self.roi_intensity is not None:
            self.roi_intensity.crop_to_size(center=center, crop_size=crop_size)
        if self.roi_morphology is not None:
            self.roi_morphology.crop_to_size(center=center, crop_size=crop_size)

    def resegmentise_mask(
            self,
            image: GenericImage,
            resegmentation_method: None | str | list[str] = None,
            intensity_range: None | tuple[Any, Any] = None,
            sigma: None | float = None
    ):
        # Resegmentation of the mask based on image intensities.

        # Set intensity range because this is used elsewhere.
        if intensity_range is not None:
            self.intensity_range = tuple(intensity_range)

        if image.is_empty() or self.is_empty():
            return

        # Ensure that masks are generated.
        self.generate_masks()

        if resegmentation_method is None or "none" in resegmentation_method:
            return

        # Initialise range.
        updated_range = [np.nan, np.nan]

        if any(method in ["threshold", "range"] for method in resegmentation_method):
            # Filter out voxels with intensity outside prescribed range

            if intensity_range is None:
                raise ValueError("Intensity range is not provided, but required for resegmentation.")

            # Upper threshold
            if not np.isnan(intensity_range[1]):
                updated_range[1] = copy.deepcopy(intensity_range[1])

            # Lower threshold
            if not np.isnan(intensity_range[0]):
                updated_range[0] = copy.deepcopy(intensity_range[0])

        if any(method in ["sigma", "outlier"] for method in resegmentation_method):
            # Remove voxels with outlier intensities

            # Local constant
            image_data = image.get_voxel_grid()
            mask_data = self.roi.get_voxel_grid()

            # Check if the voxel grid is not empty
            if np.any(mask_data):

                # Calculate mean and standard deviation of intensities in roi
                mean_int = np.mean(image_data[mask_data])
                sd_int = np.std(image_data[mask_data])

                if not np.isnan(updated_range[0]):
                    updated_range[0] = np.max([updated_range[0], mean_int - sigma * sd_int])
                else:
                    updated_range[0] = mean_int - sigma * sd_int

                if not np.isnan(updated_range[1]):
                    updated_range[1] = np.min([updated_range[1], mean_int + sigma * sd_int])
                else:
                    updated_range[1] = mean_int + sigma * sd_int

        if not np.isnan(updated_range[0]) or not np.isnan(updated_range[1]):
            # Update intensity mask
            intensity_mask_data = self.roi.get_voxel_grid()

            if not np.isnan(updated_range[0]):
                intensity_mask_data = np.logical_and((image.get_voxel_grid() >= updated_range[0]), intensity_mask_data)

            if not np.isnan(updated_range[1]):
                intensity_mask_data = np.logical_and((image.get_voxel_grid() <= updated_range[1]), intensity_mask_data)

            # Set roi voxel volume
            self.roi_intensity.set_voxel_grid(voxel_grid=intensity_mask_data)

    def dilate(
            self,
            by_slice: bool,
            distance: None | float = None,
            voxel_distance: None | float = None
    ):
        # Skip if the mask does not exist
        if self.roi is None:
            return

        self.roi.dilate(
            by_slice=by_slice,
            distance=distance,
            voxel_distance=voxel_distance
        )

    def erode(
            self,
            by_slice: bool,
            max_eroded_volume_fraction: float = 0.8,
            distance: None | float = None,
            voxel_distance: None | float = None
    ):
        # Skip if the mask does not exist
        if self.roi is None:
            return

        self.roi.erode(
            by_slice=by_slice,
            max_eroded_volume_fraction=max_eroded_volume_fraction,
            distance=distance,
            voxel_distance=voxel_distance
        )

    def fractional_volume_change(
            self,
            by_slice: bool,
            fractional_change: None | float = None
    ):
        # Skip if the mask does not exist
        if self.roi is None:
            return

        self.roi.fractional_volume_change(
            by_slice=by_slice,
            fractional_change=fractional_change
        )

    def decode_voxel_grid(self):
        """Converts run length encoded grids to conventional volumes"""

        # Decode main ROI object
        if self.roi is not None:
            self.roi.decode_voxel_grid()

        # Decode intensity and morphological masks
        if self.roi_intensity is not None:
            self.roi_intensity.decode_voxel_grid()
        if self.roi_morphology is not None:
            self.roi_morphology.decode_voxel_grid()

    def as_pandas_dataframe(
            self,
            image: None | GenericImage,
            intensity_mask: bool = False,
            morphology_mask: bool = False,
            distance_map: bool = False,
            by_slice: bool = False
    ) -> None | pd.DataFrame:

        # Check that the image and mask are present.
        if image.is_empty() or self.is_empty():
            return None

        # Check if the masks exist and assign if not.
        self.generate_masks()

        # Create table from test object
        img_dims = image.image_dimension
        index_id = np.arange(start=0, stop=np.prod(img_dims))
        coordinates = np.unravel_index(indices=index_id, shape=img_dims)
        df_img = pd.DataFrame({
            "index_id": index_id,
            "g": np.ravel(image.get_voxel_grid()),
            "x": coordinates[2],
            "y": coordinates[1],
            "z": coordinates[0]
        })

        if intensity_mask:
            df_img["roi_int_mask"] = np.ravel(self.roi_intensity.get_voxel_grid()).astype(bool)
        if morphology_mask:
            df_img["roi_morph_mask"] = np.ravel(self.roi_morphology.get_voxel_grid()).astype(bool)

        if distance_map:
            # Calculate distance by sequential border erosion
            from scipy.ndimage import generate_binary_structure, binary_erosion

            # Set up distance map and morphological voxel grid
            dist_map = np.zeros(img_dims)
            morph_voxel_grid = self.roi_morphology.get_voxel_grid()

            if by_slice:
                # Distances are determined in 2D
                binary_struct = generate_binary_structure(rank=2, connectivity=1)

                # Iterate over slices
                for ii in np.arange(0, img_dims[0]):
                    # Calculate distance by sequential border erosion
                    roi_eroded = morph_voxel_grid[ii, :, :]

                    # Iterate distance from border
                    while np.sum(roi_eroded) > 0:
                        roi_eroded = binary_erosion(roi_eroded, structure=binary_struct)
                        dist_map[ii, :, :] += roi_eroded * 1

            else:
                # Distances are determined in 3D
                binary_struct = generate_binary_structure(rank=3, connectivity=1)

                # Copy of roi morphology mask
                roi_eroded = copy.deepcopy(morph_voxel_grid)

                # Incrementally erode the morphological mask
                while np.sum(roi_eroded) > 0:
                    roi_eroded = binary_erosion(roi_eroded, structure=binary_struct)
                    dist_map += roi_eroded * 1

            # Update distance from border, as minimum distance is 1
            dist_map[morph_voxel_grid] += 1

            # Add distance map to table
            df_img["border_distance"] = np.ravel(dist_map).astype(int)

        return df_img

    def get_bounding_box(self):
        return self.roi.get_bounding_box()

    def compute_diagnostic_features(
            self,
            image: GenericImage,
            settings: SettingsClass,
            append_str: str | None = None
    ) -> pd.DataFrame:
        """ Creates diagnostic features for the ROI """

        # Set feature names
        feature_names = [
            "int_map_dim_x", "int_map_dim_y", "int_map_dim_z", "int_bb_dim_x", "int_bb_dim_y", "int_bb_dim_z",
            "int_vox_dim_x", "int_vox_dim_y", "int_vox_dim_z", "int_vox_count", "int_mean_int", "int_min_int",
            "int_max_int",
            "mrp_map_dim_x", "mrp_map_dim_y", "mrp_map_dim_z", "mrp_bb_dim_x", "mrp_bb_dim_y", "mrp_bb_dim_z",
            "mrp_vox_dim_x", "mrp_vox_dim_y", "mrp_vox_dim_z", "mrp_vox_count", "mrp_mean_int", "mrp_min_int",
            "mrp_max_int"
        ]

        # Create pandas dataframe with one row and feature columns
        df = pd.DataFrame(np.full(shape=(1, len(feature_names)), fill_value=np.nan))
        df.columns = feature_names

        # Skip further analysis if the image and/or roi are missing
        if image is None or image.is_empty() or self.is_empty():
            return df

        # Register with image on function call to ensure that mask and image correspond to the same space.
        mask_copy = self.copy()
        mask_copy.register(
            image=image,
            settings=settings
        )

        # Make copies of intensity and morphological masks (if required)
        mask_copy.generate_masks()

        # Get image and roi voxel grids
        img_voxel_grid = image.get_voxel_grid()
        int_voxel_grid = mask_copy.roi_intensity.get_voxel_grid()
        mrp_voxel_grid = mask_copy.roi_morphology.get_voxel_grid()

        # Compute bounding boxes
        int_bounding_box_dim = mask_copy.roi_intensity.get_bounding_box()
        mrp_bounding_box_dim = mask_copy.roi_morphology.get_bounding_box()
        if any(x is None for x in int_bounding_box_dim) or any(y is None for y in mrp_bounding_box_dim):
            return df

        int_bounding_box_dim = np.squeeze(np.diff(int_bounding_box_dim, axis=1) + 1)
        mrp_bounding_box_dim = np.squeeze(np.diff(mrp_bounding_box_dim, axis=1) + 1)

        # Set intensity mask features
        df["int_map_dim_x"] = mask_copy.roi_intensity.image_dimension[2]
        df["int_map_dim_y"] = mask_copy.roi_intensity.image_dimension[1]
        df["int_map_dim_z"] = mask_copy.roi_intensity.image_dimension[0]
        df["int_bb_dim_x"] = int_bounding_box_dim[2]
        df["int_bb_dim_y"] = int_bounding_box_dim[1]
        df["int_bb_dim_z"] = int_bounding_box_dim[0]
        df["int_vox_dim_x"] = mask_copy.roi_intensity.image_spacing[2]
        df["int_vox_dim_y"] = mask_copy.roi_intensity.image_spacing[1]
        df["int_vox_dim_z"] = mask_copy.roi_intensity.image_spacing[0]
        df["int_vox_count"] = np.sum(int_voxel_grid)
        df["int_mean_int"] = np.mean(img_voxel_grid[int_voxel_grid])
        df["int_min_int"] = np.min(img_voxel_grid[int_voxel_grid])
        df["int_max_int"] = np.max(img_voxel_grid[int_voxel_grid])

        # Set morphological mask features
        df["mrp_map_dim_x"] = mask_copy.roi_morphology.image_dimension[2]
        df["mrp_map_dim_y"] = mask_copy.roi_morphology.image_dimension[1]
        df["mrp_map_dim_z"] = mask_copy.roi_morphology.image_dimension[0]
        df["mrp_bb_dim_x"] = mrp_bounding_box_dim[2]
        df["mrp_bb_dim_y"] = mrp_bounding_box_dim[1]
        df["mrp_bb_dim_z"] = mrp_bounding_box_dim[0]
        df["mrp_vox_dim_x"] = mask_copy.roi_morphology.image_spacing[2]
        df["mrp_vox_dim_y"] = mask_copy.roi_morphology.image_spacing[1]
        df["mrp_vox_dim_z"] = mask_copy.roi_morphology.image_spacing[0]
        df["mrp_vox_count"] = np.sum(mrp_voxel_grid)
        df["mrp_mean_int"] = np.mean(img_voxel_grid[mrp_voxel_grid])
        df["mrp_min_int"] = np.min(img_voxel_grid[mrp_voxel_grid])
        df["mrp_max_int"] = np.max(img_voxel_grid[mrp_voxel_grid])

        # Update column names
        if append_str is None:
            df.columns = ["_".join(["diag", feature]) for feature in df.columns]
        else:
            df.columns = ["_".join(["diag", feature, append_str]) for feature in df.columns]

        del mask_copy

        return df

    def get_center_position(self) -> list[Any]:
        """Identify location of the geometric center of the roi."""
        # Return a NaN if no roi is present
        if self.roi is None:
            return [np.nan, np.nan, np.nan]

        # Determine indices of voxels included in the roi
        z_ind, y_ind, x_ind = np.where(self.roi.get_voxel_grid())

        return [np.mean(z_ind), np.mean(y_ind), np.mean(x_ind)]

    def get_center_slice(self):
        """Identify location of the central slice in the roi."""

        # Return a NaN if no roi is present
        if self.roi is None:
            return np.nan

        # Determine indices of voxels included in the roi
        z_ind, y_ind, x_ind = np.where(self.roi.get_voxel_grid())
        z_center = (np.max(z_ind) + np.min(z_ind)) // 2

        return z_center

    def write(
            self,
            dir_path: str | Path,
            write_all: bool = False,
            file_format: str = "nifti"
    ):
        """
        Write masks to file
        :param dir_path: Path to directory where the image should be written.
        :param write_all: If true, creates NIfTI files from both intensity and morphology (original) masks.
        :param file_format: File format for image file. Can be nifti or numpy.
        :return: Nothing.
        """

        roi_str_components = self.get_file_name_descriptor()

        if write_all:
            self.roi_morphology.write(
                dir_path=dir_path,
                file_name="_".join(roi_str_components + ["morph"]),
                file_format=file_format
            )
            self.roi_intensity.write(
                dir_path=dir_path,
                file_name="_".join(roi_str_components + ["int"]),
                file_format=file_format
            )
        else:
            self.roi.write(
                dir_path=dir_path,
                file_name="_".join(roi_str_components),
                file_format=file_format
            )

    def get_file_name_descriptor(self) -> list[str]:

        return self.roi.get_file_name_descriptor() + [self.roi_name]

    def export(
            self,
            write_all=False,
            export_format: str = "dict"
    ) -> np.ndarray | list[np.ndarray] | dict[str, Any] | Self:
        if self.is_empty():
            return None

        if export_format == "dict":
            attributes = self.get_export_attributes()

            if write_all:
                intensity_mask = None if self.roi_intensity is None else self.roi_intensity.get_voxel_grid()
                morphology_mask = None if self.roi_morphology is None else self.roi_morphology.get_voxel_grid()
                attributes.update({"intensity_mask": intensity_mask, "morphology_mask": morphology_mask})
            else:
                attributes.update({"mask": self.roi.get_voxel_grid()})

            return attributes

        elif export_format == "numpy":
            if write_all:
                intensity_mask = None if self.roi_intensity is None else self.roi_intensity.get_voxel_grid()
                morphology_mask = None if self.roi_morphology is None else self.roi_morphology.get_voxel_grid()
                return [intensity_mask, morphology_mask]
            else:
                return self.roi.get_voxel_grid()

        elif export_format == "native":
            return self.copy()

        else:
            raise ValueError(f"The current value of export_format was not recognised: {export_format}")

    def get_export_attributes(self) -> dict[str, Any]:
        attributes = dict([("roi_name", self.roi_name)])
        attributes.update(self.roi.get_export_attributes())

        return attributes
