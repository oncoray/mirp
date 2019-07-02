import copy
import logging

import numpy as np
import pandas as pd

from mirp.contourClass import ContourClass
from mirp.imageClass import ImageClass


def merge_roi_objects(roi_list):
    """
    Combine multiple roi objects into one
    :param roi_list:
    :return:
    """

    # If there are no rois to combine, return the only roi object
    if len(roi_list) == 1:
        return roi_list[0]

    # Read basic information concerning the roi mask
    roi_origin = roi_list[0].roi.origin
    roi_spacing = roi_list[0].roi.spacing
    roi_orientation = roi_list[0].roi.orientation
    roi_size = roi_list[0].roi.size
    roi_slice_z_pos = roi_list[0].roi.slice_z_pos

    roi_mask = np.zeros(roi_size, dtype=np.bool)

    # Iterate over rois and perform checks
    for roi in roi_list:

        # Ensure that voxel masks have been created
        if roi.contour is not None:
            raise ValueError("ROI needs to exist as a mask prior to merging.")

        # Ensure that the origin is the same
        if not np.all(np.equal(roi_origin, roi.roi.origin)):
            raise ValueError("Merged ROIs have mismatching origins.")

        # Ensure that spacing is the same
        if not np.all(np.equal(roi_spacing, roi.roi.spacing)):
            raise ValueError("Merged ROIs have mismatching voxel spacing.")

        # Ensure that orientation is the same
        if not np.all(np.equal(roi_orientation, roi.roi.orientation)):
            raise ValueError("Merged ROIs have mismatching orientations.")

        # Ensure that size is the same
        if not np.all(np.equal(roi_size, roi.roi.size)):
            raise ValueError("Merged ROIs do not have the same size.")

        roi_mask = np.logical_or(roi_mask, roi.roi.get_voxel_grid())

    # Create a roi mask object
    roi_mask_obj = ImageClass(voxel_grid=roi_mask, origin=roi_origin, spacing=roi_spacing, orientation=roi_orientation,
                              slice_z_pos=roi_slice_z_pos)

    # Set name of the roi
    combined_roi_name = "_and_".join([roi.name for roi in roi_list])

    # Create a merged roi object
    combined_roi = RoiClass(name=combined_roi_name, contour=None, roi_mask=roi_mask_obj)

    return combined_roi


class RoiClass:
    # Class for regions of interest

    def __init__(self, name, contour, roi_mask=None, g_range=np.array([np.nan, np.nan]), incl_threshold=0.5):
        """"Set initial variables
        :type roi_mask: ImageClass
        """

        self.name = name
        if contour is not None:
            contour_list = []
            for curr_contour in contour:
                contour_list.append(ContourClass(contour=curr_contour))

            self.contour = contour_list
        else:
            self.contour = None

        # Fixed parameters
        self.g_range = g_range                  # Range of allowed grey level intensities
        self.incl_threshold = incl_threshold    # Threshold for partial volume effect
        self.adapt_size = 0.0                   # Shrinkage and growth of roi
        self.svx_randomisation_id = -1          # Randomisation id for supervoxel roi randomisation

        # ROI masks
        self.roi = roi_mask              # Union of intensity and morphology masks
        self.roi_intensity = None       # Intensity mask of the ROI
        self.roi_morphology = None      # Morphological mask of the ROI

        # Diagnostics features
        self.diagnostic_list = []

    def copy(self):
        # Creates a new copy of the roi
        return copy.deepcopy(self)

    def create_mask_from_contours(self, img_obj, settings, draw_method="ray_cast"):
        # Creates an image based on provided contours

        # Skip if image object is empty
        if img_obj.is_missing:
            self.roi = None
            return

        # Create an empty roi volume
        roi_mask = np.zeros(img_obj.size, dtype=np.bool)

        # Iterate over contours to fill out the mask
        for contour in self.contour:
            # Multiple methods are implemented. All methods return a slice_list (containing slice numbers (z)) and a mask list, which contain boolean masks for respective
            # slices. This are then inserted at the specified slice positions, using an OR operation. This operation is required to avoid overwriting different slices.

            # Ray casting method to draw segmentation map based on polygon contour
            if draw_method == "ray_cast":
                slice_list, mask_list = contour.contour_to_grid_ray_cast(img_obj=img_obj)
                for ii in np.arange(len(slice_list)):
                    slice_id = slice_list[ii]
                    roi_mask[slice_id, :, :] = np.logical_or(roi_mask[slice_id, :, :], mask_list[ii])

        if settings.general.divide_disconnected_roi == "keep_largest":
            # Check if the created roi mask consists of multiple, separate segments, and keep only the largest.
            import skimage.measure

            # Label regions
            roi_label_mask, n_regions = skimage.measure.label(input=roi_mask, connectivity=2, return_num=True)

            # Determine size of regions
            roi_sizes = np.zeros(n_regions)
            for ii in np.arange(start=0, stop=n_regions):
                roi_sizes[ii] = np.sum(roi_label_mask == ii + 1)

            # Select largest region
            roi_mask = roi_label_mask == np.argmax(roi_sizes) + 1

        # Store roi as image object
        self.roi = ImageClass(voxel_grid=roi_mask, origin=img_obj.origin, spacing=img_obj.spacing, orientation=img_obj.orientation,
                              slice_z_pos=img_obj.slice_z_pos)

        # Remove contour information
        self.contour = None

    def decimate(self, by_slice):
        """
        Decimates the roi
        :param by_slice: boolean, 2D (True) or 3D (False)
        :return:
        """

        # Resect masks
        if self.roi is not None:
            self.roi.decimate(by_slice=by_slice)
        if self.roi_intensity is not None:
            self.roi_intensity.decimate(by_slice=by_slice)
        if self.roi_morphology is not None:
            self.roi_morphology.decimate(by_slice=by_slice)

    def interpolate(self, img_obj, settings):

        # Skip if image and/or is missing
        if img_obj is None or self.roi is None:
            return

        if settings.img_interpolate.anti_aliasing:
            from mirp.imageProcess import gaussian_preprocess_filter
            self.roi.set_voxel_grid(voxel_grid=gaussian_preprocess_filter(orig_vox=self.roi.get_voxel_grid(), orig_spacing=self.roi.spacing,
                                                                          sample_spacing=img_obj.spacing,
                                                                          param_beta=settings.img_interpolate.smoothing_beta, mode="nearest",
                                                                          by_slice=settings.general.by_slice))

        # Register with image
        self.register(img_obj=img_obj)

        # Binarise
        self.binarise_mask()

    def register(self, img_obj, apply_to_self=True):
        """Register roi with image
        Do not apply threshold until after interpolation"""

        if apply_to_self is False:
            roi_copy = self.copy()
            roi_copy.register(img_obj=img_obj, apply_to_self=True)
            return roi_copy

        from mirp.imageProcess import interpolate_to_new_grid

        # Skip if image and/or is missing
        if img_obj is None or self.roi is None:
            return

        # Check whether registration is required
        registration_required = False

        # Mismatch in grid dimension
        if np.any([np.abs(np.array(self.roi.size) - np.array(img_obj.size)) > 0.0]):
            registration_required = True

        # Mismatch in origin
        if np.any([np.abs(self.roi.origin - img_obj.origin) > 0.0]):
            registration_required = True

        # Mismatch in spacing
        if np.any([np.abs(self.roi.spacing - img_obj.spacing) > 0.0]):
            registration_required = True

        if registration_required:
            # Register roi to image; this transforms the roi grid into
            self.roi.size, self.roi.origin, self.roi.spacing, voxel_grid = \
                interpolate_to_new_grid(orig_dim=self.roi.size, orig_origin=self.roi.origin, orig_spacing=self.roi.spacing, orig_vox=self.roi.get_voxel_grid(),
                                        sample_dim=img_obj.size, sample_origin=img_obj.origin, sample_spacing=img_obj.spacing,
                                        order=1, mode="nearest", align_to_center=False)

            # Update slice position
            self.roi.slice_z_pos = self.roi.origin[0] + np.arange(self.roi.size[0]) * self.roi.spacing[0]

            # Update voxel grid
            self.roi.set_voxel_grid(voxel_grid=voxel_grid)

    def binarise_mask(self):

        if self.roi is None:
            return

        if not self.roi.dtype_name == "bool":
            self.roi.set_voxel_grid(voxel_grid=np.around(self.roi.get_voxel_grid(), 6) >= np.around(self.incl_threshold, 6))

    def generate_masks(self):
        """"Generate roi intensity and morphology masks"""

        if self.roi is None:
            self.roi_intensity = None
            self.roi_morphology = None
        else:
            self.roi_intensity = self.roi.copy()
            self.roi_morphology = self.roi.copy()

    def update_roi(self):
        """Update region of interest based on intensity and morphological masks"""

        if self.roi is None or self.roi_intensity is None or self.roi_morphology is None:
            return

        self.roi.set_voxel_grid(voxel_grid=np.logical_or(self.roi_intensity.get_voxel_grid(), self.roi_morphology.get_voxel_grid()))

    def crop(self, map_ext_z, map_ext_y, map_ext_x, xy_only=False, z_only=False):
        """"Resects roi"""

        # Resect masks
        if self.roi is not None:
            self.roi.crop(map_ext_z=map_ext_z, map_ext_y=map_ext_y, map_ext_x=map_ext_x, xy_only=xy_only, z_only=z_only)
        if self.roi_intensity is not None:
            self.roi_intensity.crop(map_ext_z=map_ext_z, map_ext_y=map_ext_y, map_ext_x=map_ext_x, xy_only=xy_only, z_only=z_only)
        if self.roi_morphology is not None:
            self.roi_morphology.crop(map_ext_z=map_ext_z, map_ext_y=map_ext_y, map_ext_x=map_ext_x, xy_only=xy_only, z_only=z_only)

    def crop_to_size(self, center, crop_size, xy_only=False):
        """"Crops roi to a pre-defined size"""

        # Crop masks to size
        if self.roi is not None:
            self.roi.crop_to_size(center=center, crop_size=crop_size, xy_only=xy_only)
        if self.roi_intensity is not None:
            self.roi_intensity.crop(center=center, crop_size=crop_size, xy_only=xy_only)
        if self.roi_morphology is not None:
            self.roi_morphology.crop(center=center, crop_size=crop_size, xy_only=xy_only)

    def resegmentise_mask(self, img_obj, by_slice, method, settings):
        # Resegmentation of the roi map based on grey level values

        from skimage.measure import label
        from skimage.morphology import remove_small_holes

        # Skip if required voxel grids are missing
        if img_obj.is_missing or self.roi_intensity is None or self.roi_morphology is None:
            return

        ################################################################################################################
        # Resegmentation that affects both intensity and morphological maps
        ################################################################################################################

        # Initialise range
        updated_range = np.array([np.nan, np.nan])

        if bool(set(method).intersection(["threshold", "range"])):
            # Filter out voxels with intensity outside prescribed range

            # Local constant
            g_thresh = settings.roi_resegment.g_thresh  # Threshold values

            # Upper threshold
            if not np.isnan(g_thresh[1]):
                updated_range[1] = copy.deepcopy(g_thresh[1])

            # Lower threshold
            if not np.isnan(g_thresh[0]):
                updated_range[0] = copy.deepcopy(g_thresh[0])

            # Set the threshold values as g_range
            self.g_range = g_thresh

        if bool(set(method).intersection(["sigma", "outlier"])):
            # Remove voxels with outlier intensities

            # Local constant
            sigma = settings.roi_resegment.sigma
            img_voxel_grid = img_obj.get_voxel_grid()
            roi_voxel_grid = self.roi_intensity.get_voxel_grid()

            # Check if the voxel grid is not empty
            if np.any(roi_voxel_grid):

                # Calculate mean and standard deviation of intensities in roi
                mean_int = np.mean(img_voxel_grid[roi_voxel_grid])
                sd_int   = np.std(img_voxel_grid[roi_voxel_grid])

                if not np.isnan(updated_range[0]):
                    updated_range[0] = np.max([updated_range[0], mean_int - sigma * sd_int])
                else:
                    updated_range[0] = mean_int - sigma * sd_int

                if not np.isnan(updated_range[0]):
                    updated_range[1] = np.min([updated_range[1], mean_int + sigma * sd_int])
                else:
                    updated_range[1] = mean_int + sigma * sd_int

        if not np.isnan(updated_range[0]) or not np.isnan(updated_range[1]):
            # Update intensity mask
            roi_voxel_grid = self.roi_intensity.get_voxel_grid()

            if not np.isnan(updated_range[0]):
                roi_voxel_grid = np.logical_and((img_obj.get_voxel_grid() >= updated_range[0]), roi_voxel_grid)

            if not np.isnan(updated_range[1]):
                roi_voxel_grid = np.logical_and((img_obj.get_voxel_grid() <= updated_range[1]), roi_voxel_grid)

            # Set roi voxel volume
            self.roi_intensity.set_voxel_grid(voxel_grid=roi_voxel_grid)

        ################################################################################################################
        # Resegmentation that affects only morphological maps
        ################################################################################################################
        if bool(set(method).intersection("close_volume")):
            # Close internal volumes

            from scipy.ndimage import generate_binary_structure, binary_erosion

            # Read minimal volume required
            max_fill_volume = settings.roi_resegment.max_fill_volume

            # Get voxel grid of the roi morphological mask
            roi_voxel_grid = self.roi_morphology.get_voxel_grid()

            # Determine fill volume (in voxels); if max_fill_volume is less than 0.0, fill all holes
            if max_fill_volume < 0.0: fill_volume = np.prod(np.array(self.roi_morphology.size)) + 1.0
            else:                     fill_volume = np.floor(max_fill_volume / np.prod(self.roi_morphology.spacing)) + 1.0

            # If the maximum fill volume is smaller than the minimal size of a hole
            if fill_volume < 1.0: return None

            # Label all non-roi voxels and get label corresponding to voxels outside of the roi
            non_roi_label = label(np.pad(roi_voxel_grid, 1, mode="constant", constant_values=0),
                                  background=1, connectivity=3)
            outside_label = non_roi_label[0, 0, 0]

            # Crop non-roi labels and determine non-roi voxels outside of the mask
            non_roi_label = non_roi_label[1:-1, 1:-1, 1:-1]
            vox_outside = non_roi_label == outside_label

            # Determine mask of voxels which are not internal holes
            vox_not_internal = np.logical_or(roi_voxel_grid, vox_outside)

            # Check if there are any holes, otherwise continue
            if not np.any(~vox_not_internal): return None

            if by_slice:
                # 2D approach to filling holes

                for ii in np.arange(0, self.roi_morphology.size[0]):
                    # Skip operations on slides that do not contain voxels in the mask or no holes in the slice
                    if not np.any(roi_voxel_grid[ii, :, :]): continue
                    if not(np.any(~vox_not_internal[ii, :, :])): continue

                    # Fill holes up to fill_volume in voxel number
                    vox_filled = remove_small_holes(vox_not_internal[ii, :, :], min_size=np.int(fill_volume), connectivity=2)

                    # Update mask by removing outside voxels from the mask
                    roi_voxel_grid[ii, :, :] = np.squeeze(np.logical_and(vox_filled, ~vox_outside[ii, :, :]))
            else:
                # 3D approach to filling holes

                # Fill holes up to fill_volume in voxel number
                vox_filled = remove_small_holes(vox_not_internal, min_size=np.int(fill_volume), connectivity=3)

                # Update mask by removing outside voxels from the mask
                roi_voxel_grid = np.logical_and(vox_filled, ~vox_outside)

            # Update voxel grid
            self.roi_morphology.set_voxel_grid(voxel_grid=roi_voxel_grid)

        if bool(set(method).intersection("remove_disconnected")):
            # Remove disconnected voxels

            # Discover prior disconnected volumes from the roi voxel grid
            vox_disconnected = label(self.roi.get_voxel_grid(), background=0, connectivity=3)
            vox_disconnected_labels = np.unique(vox_disconnected)

            # Set up an empty morphological masks
            upd_vox_mask = np.full(shape=self.roi_morphology.size, fill_value=False, dtype=np.bool)

            # Get the minimum volume fraction for inclusion as voxels
            min_vol_fract = settings.roi_resegment.min_vol_fract

            # Iterate over disconnected labels
            for curr_volume_label in vox_disconnected_labels:

                # Skip background
                if vox_disconnected_labels == 0: continue

                # Mask only current volume, skip if empty
                curr_mask = np.logical_and(self.roi_morphology.get_voxel_grid(), vox_disconnected == curr_volume_label)
                if not np.any(curr_mask): continue

                # Find fully disconnected voxels groups and count them
                vox_mask = label(curr_mask, background=0, connectivity=3)
                vox_mask_labels, vox_label_count = np.unique(vox_mask, return_counts=True)

                # Filter out the background counts
                valid_label_id = np.nonzero(vox_mask_labels)
                vox_mask_labels = vox_mask_labels[valid_label_id]
                vox_label_count = vox_label_count[valid_label_id]

                # Normalise to maximum
                vox_label_count = vox_label_count / np.max(vox_label_count)

                # Select labels fulfilling the minimal size
                vox_mask_labels = vox_mask_labels[vox_label_count >= min_vol_fract]

                for vox_mask_label_id in vox_mask_labels:
                    upd_vox_mask += vox_mask == vox_mask_label_id

                # Update morphological voxel grid
                self.roi_morphology.set_voxel_grid(voxel_grid=upd_vox_mask > 0)

    def is_empty(self):
        """Checks whether the roi or one of its masks is empty"""

        # Original roi object
        if self.roi is not None:
            n_roi_voxels     = np.int(np.sum(self.roi.get_voxel_grid()))
            if n_roi_voxels == 0:
                return True

        # Roi intensity mask
        if self.roi_intensity is not None:
            n_roi_int_voxels = np.int(np.sum(self.roi_intensity.get_voxel_grid()))
            if n_roi_int_voxels == 0:
                return True

        # Roi morphological mask
        if self.roi_morphology is not None:
            n_roi_morph_voxels = np.int(np.sum(self.roi_morphology.get_voxel_grid()))
            if n_roi_morph_voxels == 0:
                return True

        # If none of the above rois was empty, return false
        return False

    def rotate(self, angle, img_obj):
        """ Rotates roi in the y-x plane """

        # Register with image prior to rotation
        self.register(img_obj=img_obj)

        # Rotate roi
        self.roi.rotate(angle)

    def dilate(self, by_slice, dist=None, vox_dist=None):
        from mirp.featureSets.utilities import rep
        import scipy.ndimage as ndi

        # Skip if the roi does not exist
        if self.roi is None:
            return

        # Check dtype of the roi voxel grid and binarise if necessary
        if not self.roi.dtype_name == "bool":
            logging.info("Converting roi to boolean before dilation.")
            self.binarise_mask()

        # Check if any distance is provided for dilation
        if vox_dist is None and dist is None:
            logging.error("No dilation distance provided.")

        # Check whether voxel are isometric
        if by_slice: spacing = self.roi.spacing[[1, 2]]
        else:        spacing = self.roi.spacing

        if np.any(spacing - np.max(spacing) != 0.0):
            logging.warning("Non-uniform voxel spacing was detected. Roi dilation requires uniform voxel spacing.")

        # Derive filter extension and distance
        if dist is not None:
            base_ext: int = np.max([np.floor(dist / np.max(spacing)).astype(np.int), 0])
        else:
            base_ext: int = np.int(vox_dist)
            dist     = vox_dist * np.max(spacing)

        # Check if an actual extension is required.
        if base_ext > 0:

            # Create displacement map
            df_base = pd.DataFrame({"x": rep(x=np.arange(-base_ext, base_ext + 1),
                                             each=(2 * base_ext + 1) * (2 * base_ext + 1),
                                             times=1),
                                    "y": rep(x=np.arange(-base_ext, base_ext + 1),
                                             each=2 * base_ext + 1,
                                             times=2 * base_ext + 1),
                                    "z": rep(x=np.arange(-base_ext, base_ext + 1),
                                             each=1,
                                             times=(2 * base_ext + 1) * (2 * base_ext + 1))})

            # Calculate distances for displacement map
            df_base["dist"] = np.sqrt(np.sum(np.multiply(df_base.loc[:, ("z", "y", "x")].values, self.roi.spacing) ** 2.0, axis=1))

            # Identify elements in range
            if by_slice: df_base["in_range"] = np.logical_and(df_base.dist <= dist, df_base.z == 0)
            else:        df_base["in_range"] = df_base.dist <= dist

            # Update voxel coordinates to start at [0,0,0]
            df_base.loc[:, ["x", "y", "z"]] -= df_base.loc[0, ["x", "y", "z"]]

            # Generate geometric filter structure
            geom_struct = np.zeros(shape=(np.max(df_base.z) + 1, np.max(df_base.y) + 1,
                                          np.max(df_base.x) + 1), dtype=np.bool)
            geom_struct[df_base.z.astype(np.int), df_base.y.astype(np.int), df_base.x.astype(np.int)] = df_base.in_range

            # Dilate roi mask amd store voxel grid
            self.roi.set_voxel_grid(voxel_grid=ndi.binary_dilation(self.roi.get_voxel_grid(), structure=geom_struct, iterations=1))

        else:
            logging.info("No dilation: distance %s is too small compared to voxel spacing %s.", str(dist),
                         str(np.max(spacing)))

    def adapt_volume(self, by_slice, vol_grow_fract=None):
        import scipy.ndimage

        # Skip if the roi does not exist
        if self.roi is None:
            return

        # Check dtype of the roi voxel grid and binarise if necessary
        if not self.roi.dtype_name == "bool":
            logging.info("Converting roi to boolean before dilation.")
            self.binarise_mask()

        # Check if any distance is provided for dilation
        if vol_grow_fract is None:
            logging.error("No dilation volume fraction was provided.")

        # Check whether voxels are isometric
        if by_slice:
            spacing = self.roi.spacing[[1, 2]]
        else:
            spacing = self.roi.spacing

        if np.any(spacing - np.max(spacing) != 0.0):
            logging.warning("Non-uniform voxel spacing was detected. Roi volume adaptation requires uniform voxel spacing.")

        # Set geometrical structure
        geom_struct = scipy.ndimage.generate_binary_structure(3, 1)
        if by_slice: geom_struct[(0, 2), :, :] = False  # Set structures in different slices to 0

        if not vol_grow_fract == 0.0 and not self.is_empty():
            # Determine original volume
            previous_roi = self.roi.get_voxel_grid()
            orig_volume = np.sum(previous_roi)

            # Iteratively grow or shrink the volume. The loop terminates through break statements
            while True:
                if vol_grow_fract > 0.0:
                    updated_roi = scipy.ndimage.binary_dilation(previous_roi, structure=geom_struct, iterations=1)
                else:
                    updated_roi = scipy.ndimage.binary_erosion(previous_roi, structure=geom_struct, iterations=1)

                new_volume = np.sum(updated_roi)
                if new_volume == 0:
                    break

                if vol_grow_fract > 0.0 and new_volume/orig_volume - 1.0 >= vol_grow_fract:
                    break

                if vol_grow_fract < 0.0 and new_volume/orig_volume - 1.0 <= vol_grow_fract:
                    break

                # Replace previous roi by the updated roi
                previous_roi = updated_roi

            # Randomly add/remove border voxels until desired growth/shrinkage is achieved
            if not new_volume/orig_volume - 1.0 == vol_grow_fract:
                additional_vox = np.abs(int(np.floor(orig_volume * (1.0 + vol_grow_fract) - np.sum(previous_roi))))
                if additional_vox > 0:
                    border_voxel_ind = np.array(np.where(np.logical_xor(previous_roi, updated_roi)))
                    select_ind = np.random.choice(a=border_voxel_ind.shape[1], size=additional_vox, replace=False)
                    border_voxel_ind = border_voxel_ind[:, select_ind]
                    if vol_grow_fract > 0.0:
                        previous_roi[border_voxel_ind[0, :], border_voxel_ind[1, :], border_voxel_ind[2, :]] = True
                    else:
                        previous_roi[border_voxel_ind[0, :], border_voxel_ind[1, :], border_voxel_ind[2, :]] = False

            # Set the new roi
            self.roi.set_voxel_grid(voxel_grid=previous_roi)

    def erode(self, by_slice, eroded_vol_fract=0.8, dist=None, vox_dist=None):
        """" Erosion of the roi segment """

        import scipy.ndimage as ndi

        # Skip if the roi does not exist
        if self.roi is None:
            return

        # Check whether voxels are booleans
        if not self.roi.dtype_name == "bool":
            logging.info("Converting roi to boolean before erosion.")
            self.binarise_mask()

        # Check if any distance is provided for dilation
        if vox_dist is None and dist is None:
            logging.error("No erosion distance provided.")

        # Check whether voxel are isometric
        if by_slice: spacing = self.roi.spacing[[1, 2]]
        else:        spacing = self.roi.spacing

        if np.any(spacing - np.max(spacing) != 0.0):
            logging.warning("Non-uniform voxel spacing was detected. Roi erosion requires uniform voxel spacing.")

        # Set geometrical structure
        geom_struct = ndi.generate_binary_structure(3, 1)
        if by_slice: geom_struct[(0, 2), :, :] = False    # Set structures in different slices to 0

        # Set number of erosion steps
        if vox_dist is None:
            erode_steps = np.max([np.round(np.abs(dist) / np.max(spacing)).astype(np.int), 0])
        else:
            erode_steps = np.abs(vox_dist.astype(np.int))
            dist = vox_dist * np.max(spacing)

        if erode_steps > 0:
            # Determine initial volume
            voxels_prev = self.roi.get_voxel_grid()
            vol_init = np.sum(voxels_prev)

            # Iterate over erosion steps
            for step in np.arange(0, erode_steps):

                # Perform erosion
                voxels_upd = ndi.binary_erosion(voxels_prev, structure=geom_struct, iterations=1)

                # Calculate volume of the eroded volume
                vol_curr = np.sum(voxels_upd)

                # Stop erosion if the volume shrinks below 80 percent of the original volume due to erosion and return
                # return voxels from the previous erosion step.
                if vol_curr * 1.0 / vol_init < eroded_vol_fract:
                    voxels_upd = voxels_prev
                    break
                else:
                    voxels_prev = voxels_upd

            # Set updated voxels
            self.roi.set_voxel_grid(voxel_grid=voxels_upd)
        else:
            logging.info("No erosion: distance %s is too small compared to voxel spacing %s.", str(dist), str(np.max(spacing)))

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

    def as_pandas_dataframe(self, img_obj, intensity_mask=False, morphology_mask=False, distance_map=False, by_slice=False):
        """Converts the image and roi voxel grids to a pandas dataframe for further processing"""

        # Return None if the image and/or ROI are missing
        if img_obj.is_missing or self.roi is None:
            return None

        # Check if the masks exist and assign if not
        if intensity_mask and self.roi_intensity is None:
            self.roi_intensity = self.roi.copy()
        if (morphology_mask or distance_map) and self.roi_morphology is None:
            self.roi_morphology = self.roi.copy()

        # Create table from test object
        img_dims = img_obj.size
        index_id = np.arange(start=0, stop=np.prod(img_dims))
        coords = np.unravel_index(indices=index_id, dims=img_dims)
        df_img = pd.DataFrame({"index_id": index_id,
                               "g":        np.ravel(img_obj.get_voxel_grid()),
                               "x": coords[2],
                               "y": coords[1],
                               "z": coords[0]})

        if intensity_mask:
            df_img["roi_int_mask"] = np.ravel(self.roi_intensity.get_voxel_grid()).astype(np.bool)
        if morphology_mask:
            df_img["roi_morph_mask"] = np.ravel(self.roi_morphology.get_voxel_grid()).astype(np.bool)
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
                    roi_eroded = np.squeeze(morph_voxel_grid[ii, :, :])

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
            df_img["border_distance"] = np.ravel(dist_map).astype(np.int32)

        return df_img

    def compute_diagnostic_features(self, img_obj, append_str=""):
        """ Creates diagnostic features for the ROI """

        # Set feature names
        feat_names = ["int_map_dim_x", "int_map_dim_y", "int_map_dim_z", "int_bb_dim_x", "int_bb_dim_y", "int_bb_dim_z",
                      "int_vox_dim_x", "int_vox_dim_y", "int_vox_dim_z", "int_vox_count", "int_mean_int", "int_min_int", "int_max_int",
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

        # Set intensity mask features
        df.ix[0, ["int_map_dim_z", "int_map_dim_y", "int_map_dim_x"]] = roi_copy.roi_intensity.size                         # ROI map dimensions (in voxels)
        df.ix[0, ["int_bb_dim_z", "int_bb_dim_y", "int_bb_dim_x"]] = np.squeeze(np.diff(roi_copy.get_bounding_box(roi_voxel_grid=int_voxel_grid), axis=0) + 1)            # ROI bounding box dimensions (in voxels)
        df.ix[0, ["int_vox_dim_z", "int_vox_dim_y", "int_vox_dim_x"]] = roi_copy.roi_intensity.spacing                      # Voxel dimensions (in mm)
        df.ix[0, "int_vox_count"] = np.sum(int_voxel_grid)                                                                  # Number of voxels in ROI
        df.ix[0, "int_mean_int"]  = np.mean(img_voxel_grid[int_voxel_grid])                                                 # Mean intensity in ROI
        df.ix[0, "int_min_int"]   = np.min(img_voxel_grid[int_voxel_grid])                                                  # Minimum intensity in ROI
        df.ix[0, "int_max_int"]   = np.max(img_voxel_grid[int_voxel_grid])                                                  # Maximum intensity in ROI

        # Set morphological mask features
        df.ix[0, ["mrp_map_dim_z", "mrp_map_dim_y", "mrp_map_dim_x"]] = roi_copy.roi_morphology.size                        # ROI map dimensions (in voxels)
        df.ix[0, ["mrp_bb_dim_z", "mrp_bb_dim_y", "mrp_bb_dim_x"]] = np.squeeze(np.diff(roi_copy.get_bounding_box(roi_voxel_grid=mrp_voxel_grid), axis=0) + 1)  # ROI bounding box dimensions (in voxels)
        df.ix[0, ["mrp_vox_dim_z", "mrp_vox_dim_y", "mrp_vox_dim_x"]] = roi_copy.roi_morphology.spacing                     # Voxel dimensions (in mm)
        df.ix[0, "mrp_vox_count"] = np.sum(mrp_voxel_grid)                                                                  # Number of voxels in ROI
        df.ix[0, "mrp_mean_int"] = np.mean(img_voxel_grid[mrp_voxel_grid])                                                  # Mean intensity in ROI
        df.ix[0, "mrp_min_int"] = np.min(img_voxel_grid[mrp_voxel_grid])                                                    # Minimum intensity in ROI
        df.ix[0, "mrp_max_int"] = np.max(img_voxel_grid[mrp_voxel_grid])                                                    # Maximum intensity in ROI

        del roi_copy

        # Update column names
        df.columns = ["diag_" + feat + append_str for feat in df.columns]

        self.diagnostic_list += [df]

    def get_bounding_box(self, roi_voxel_grid):
        # Calculates coordinates of ROI bounding box
        z_ind, y_ind, x_ind = np.where(roi_voxel_grid)
        max_ind = np.array((np.max(z_ind), np.max(y_ind), np.max(x_ind)))
        min_ind = np.array((np.min(z_ind), np.min(y_ind), np.min(x_ind)))
        del z_ind, y_ind, x_ind

        return min_ind, max_ind

    def get_center_slice(self):
        """ Identify location of the central slice in the roi """

        # Return a NaN if no roi is present
        if self.roi is None:
            return np.nan

        # Determine indices of voxels included in the roi
        z_ind, y_ind, x_ind = np.where(self.roi.get_voxel_grid())
        z_center = (np.max(z_ind) + np.min(z_ind)) // 2

        return z_center

    def get_all_slices(self):
        """ Identify location of all slices in the roi """

        # Return NaN in case the roi is missing
        if self.roi is None:
            return np.array([np.nan])

        z_ind, y_ind, x_ind = np.where(self.roi.get_voxel_grid())

        return np.unique(z_ind)

    def export(self, img_obj, file_path):
        """
        Export roi to file
        :param img_obj:
        :param file_path:
        :return:
        """

        roi_str = img_obj.get_export_descriptor()
        roi_str += self.get_export_descriptor()

        # Write morphological and intensity roi
        if self.roi_morphology is not None and self.roi_intensity is not None:
            self.roi_morphology.write(file_path=file_path, file_name=roi_str + "_morph.nii.gz")
            self.roi_intensity.write(file_path=file_path, file_name=roi_str + "_int.nii.gz")
        elif self.roi is not None:
            self.roi.write(file_path=file_path, file_name=roi_str + ".nii.gz")
        else:
            return

    def get_export_descriptor(self):
        """
        Generates an export string for identifying a file
        :return: export string
        """
        export_str = ""

        if self.adapt_size != 0.0:
            # Volume adaptation
            export_str += "_vol" + str(self.adapt_size)
        if self.svx_randomisation_id != -1:
            # Contour randomisation
            export_str += "_svx" + str(self.svx_randomisation_id)

        export_str += self.name

        return export_str

    def get_slices(self, slice_number=None):
        # Extract roi objects for each slice

        roi_obj_list = []

        # Create a copy of the current object
        base_roi_obj = self.copy()

        # Remove attributes that need to be set
        base_roi_obj.roi = None
        base_roi_obj.roi_intensity = None
        base_roi_obj.roi_morphology = None

        if slice_number is None:
            # Extract mask for each slice.  Copy the base roi object.

            if self.roi is not None:
                roi_slices = self.roi.get_slices()

            if self.roi_intensity is not None:
                roi_int_slices = self.roi_intensity.get_slices()
            else:
                roi_int_slices = None

            if self.roi_morphology is not None:
                roi_morph_slices = self.roi_morphology.get_slices()
            else:
                roi_morph_slices = None

            # Add masks to a roi object for each slice
            for ii in np.arange(self.roi.size[0]):
                slice_roi_obj = copy.deepcopy(base_roi_obj)

                if self.roi is not None:
                    slice_roi_obj.roi = roi_slices[ii]
                if self.roi_intensity is not None:
                    slice_roi_obj.roi_intensity = roi_int_slices[ii]
                if self.roi_morphology is not None:
                    slice_roi_obj.roi_morphology = roi_morph_slices[ii]

                # Add to list
                roi_obj_list += [slice_roi_obj]
        else:
            # Extract a single slice. Copy the base roi object.
            slice_roi_obj = copy.deepcopy(base_roi_obj)

            # Add the mask for the requested slice
            if self.roi is not None:
                slice_roi_obj.roi = self.roi.get_slices(slice_number=slice_number)
            if self.roi_intensity is not None:
                slice_roi_obj.roi_intensity = self.roi_intensity.get_slices(slice_number=slice_number)
            if self.roi_morphology is not None:
                slice_roi_obj.roi_morphology = self.roi_morphology.get_slices(slice_number=slice_number)

            # Add to list
            roi_obj_list += [slice_roi_obj]

        return roi_obj_list
