import numpy as np
import pandas as pd
import copy
from typing import Optional, List, Tuple, Any, Union

from mirp.images.genericImage import GenericImage
from mirp.images.maskImage import MaskImage
from mirp.importSettings import SettingsClass


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
        self.roi_intensity: Optional[MaskImage] = None
        self.roi_morphology: Optional[MaskImage] = None

        # Set name of the mask.
        self.roi_name: Union[str, List[str]] = roi_name

        # Set intensity range.
        self.intensity_range: Tuple[Any] = tuple([np.nan, np.nan])

    def copy(self, drop_image=False):

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

    def is_empty(self):
        if self.roi is None:
            return True

        return self.roi.is_empty()

    def append_name(self, x):
        if isinstance(self.roi_name, str):
            self.roi_name = [self.roi_name]

        if isinstance(x, str):
            self.roi_name += [x]
        elif isinstance(x, list):
            self.roi_name += x
        else:
            raise TypeError("The x attribute is expected to be a string or list of strings.")

    def interpolate(
            self,
            image: Optional[GenericImage],
            settings: SettingsClass):
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
            settings: SettingsClass
    ):
        if self.is_empty():
            return

        self.roi.register(image=image, settings=settings)
        if self.roi_intensity is not None:
            self.roi_intensity.register(image=image, settings=settings)
        if self.roi_morphology is not None:
            self.roi_morphology.register(image=image, settings=settings)

    def select_largest_slice(self):
        """Crops to the largest slice."""

        # Do not crop if there is nothing to crop
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
            resegmentation_method: List[str],
            settings: SettingsClass):
        # Resegmentation of the mask based on image intensities.

        if image.is_empty() or self.is_empty():
            return

        # Ensure that masks are generated.
        self.generate_masks()

        # Initialise range
        updated_range = [np.nan, np.nan]

        if any(method in ["threshold", "range"] for method in resegmentation_method):
            # Filter out voxels with intensity outside prescribed range

            # Local constant
            intensity_range = settings.roi_resegment.intensity_range

            # Upper threshold
            if not np.isnan(intensity_range[1]):
                updated_range[1] = copy.deepcopy(intensity_range[1])

            # Lower threshold
            if not np.isnan(intensity_range[0]):
                updated_range[0] = copy.deepcopy(intensity_range[0])

            # Set the threshold values for the mask.
            self.intensity_range = tuple(intensity_range)

        if any(method in ["sigma", "outlier"] for method in resegmentation_method):
            # Remove voxels with outlier intensities

            # Local constant
            sigma = settings.roi_resegment.sigma
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
            distance: Optional[float] = None,
            voxel_distance: Optional[float] = None
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
            distance: Optional[float] = None,
            voxel_distance: Optional[float] = None
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
            fractional_change: Optional[float] = None
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
            image: Optional[GenericImage],
            intensity_mask: bool = False,
            morphology_mask: bool = False,
            distance_map: bool = False,
            by_slice: bool = False
    ) -> Optional[pd.DataFrame]:

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

    def compute_diagnostic_features(self, img_obj, append_str=""):
        """ Creates diagnostic features for the ROI """

        # Set feature names
        feat_names = ["int_map_dim_x", "int_map_dim_y", "int_map_dim_z", "int_bb_dim_x", "int_bb_dim_y", "int_bb_dim_z",
                      "int_vox_dim_x", "int_vox_dim_y", "int_vox_dim_z", "int_vox_count", "int_mean_int", "int_min_int",
                      "int_max_int",
                      "mrp_map_dim_x", "mrp_map_dim_y", "mrp_map_dim_z", "mrp_bb_dim_x", "mrp_bb_dim_y", "mrp_bb_dim_z",
                      "mrp_vox_dim_x", "mrp_vox_dim_y", "mrp_vox_dim_z", "mrp_vox_count", "mrp_mean_int", "mrp_min_int",
                      "mrp_max_int"]

        # Create pandas dataframe with one row and feature columns
        df = pd.DataFrame(np.full(shape=(1, len(feat_names)), fill_value=np.nan))
        df.columns = feat_names

        # Skip further analysis if the image and/or roi are missing
        if img_obj.is_missing or self.roi is None:
            return df

        # Register with image on function call
        roi_copy = self.register(img_obj, apply_to_self=False)

        # Binarise (if required)
        roi_copy.binarise_mask()

        # Make copies of intensity and morphological masks (if required)
        if roi_copy.roi_intensity is None:
            roi_copy.roi_intensity = roi_copy.roi
        if roi_copy.roi_morphology is None:
            roi_copy.roi_morphology = roi_copy.roi

        # Get image and roi voxel grids
        img_voxel_grid = img_obj.get_voxel_grid()
        int_voxel_grid = roi_copy.roi_intensity.get_voxel_grid()
        mrp_voxel_grid = roi_copy.roi_morphology.get_voxel_grid()

        # Compute bounding boxes
        int_bounding_box_dim = np.squeeze(np.diff(roi_copy.get_bounding_box(roi_voxel_grid=int_voxel_grid), axis=0) + 1)
        mrp_bounding_box_dim = np.squeeze(np.diff(roi_copy.get_bounding_box(roi_voxel_grid=mrp_voxel_grid), axis=0) + 1)

        # Set intensity mask features
        df["int_map_dim_x"] = roi_copy.roi_intensity.size[2]
        df["int_map_dim_y"] = roi_copy.roi_intensity.size[1]
        df["int_map_dim_z"] = roi_copy.roi_intensity.size[0]
        df["int_bb_dim_x"] = int_bounding_box_dim[2]
        df["int_bb_dim_y"] = int_bounding_box_dim[1]
        df["int_bb_dim_z"] = int_bounding_box_dim[0]
        df["int_vox_dim_x"] = roi_copy.roi_intensity.spacing[2]
        df["int_vox_dim_y"] = roi_copy.roi_intensity.spacing[1]
        df["int_vox_dim_z"] = roi_copy.roi_intensity.spacing[0]
        df["int_vox_count"] = np.sum(int_voxel_grid)
        df["int_mean_int"] = np.mean(img_voxel_grid[int_voxel_grid])
        df["int_min_int"] = np.min(img_voxel_grid[int_voxel_grid])
        df["int_max_int"] = np.max(img_voxel_grid[int_voxel_grid])

        # Set morphological mask features
        df["mrp_map_dim_x"] = roi_copy.roi_morphology.size[2]
        df["mrp_map_dim_y"] = roi_copy.roi_morphology.size[1]
        df["mrp_map_dim_z"] = roi_copy.roi_morphology.size[0]
        df["mrp_bb_dim_x"] = mrp_bounding_box_dim[2]
        df["mrp_bb_dim_y"] = mrp_bounding_box_dim[1]
        df["mrp_bb_dim_z"] = mrp_bounding_box_dim[0]
        df["mrp_vox_dim_x"] = roi_copy.roi_morphology.spacing[2]
        df["mrp_vox_dim_y"] = roi_copy.roi_morphology.spacing[1]
        df["mrp_vox_dim_z"] = roi_copy.roi_morphology.spacing[0]
        df["mrp_vox_count"] = np.sum(mrp_voxel_grid)
        df["mrp_mean_int"] = np.mean(img_voxel_grid[mrp_voxel_grid])
        df["mrp_min_int"] = np.min(img_voxel_grid[mrp_voxel_grid])
        df["mrp_max_int"] = np.max(img_voxel_grid[mrp_voxel_grid])

        # Update column names
        df.columns = ["_".join(["diag", feature, append_str]).strip("_") for feature in df.columns]

        del roi_copy

        self.diagnostic_list += [df]

    def get_center_slice(self):
        """ Identify location of the central slice in the roi """

        # Return a NaN if no roi is present
        if self.roi is None:
            return np.nan

        # Determine indices of voxels included in the roi
        z_ind, y_ind, x_ind = np.where(self.roi.get_voxel_grid())
        z_center = (np.max(z_ind) + np.min(z_ind)) // 2

        return z_center