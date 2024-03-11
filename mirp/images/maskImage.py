import copy
import hashlib
import warnings

import numpy as np
import pandas as pd
from typing import Any

from mirp.images.genericImage import GenericImage
from mirp.settings.settingsGeneric import SettingsClass


class MaskImage(GenericImage):

    def __init__(self, **kwargs):
        # Declare local attributes first because these may be updated in super()__init__.
        self.image_encoded = False
        self.alteration_size: float = 0.0
        self.slic_randomisation_id: int | None = None

        super().__init__(**kwargs)

    def is_empty(self):
        if self.image_encoded:
            return False
        else:
            return self.image_data is None

    def is_empty_mask(self):
        if self.is_empty():
            return True

        if np.sum(self.get_voxel_grid()) == 0:
            return True

        return False

    def encode_voxel_grid(self):
        # Check if image data are present, or are already encoded.
        if self.is_empty() or self.image_encoded is True:
            return

        # Check that the image consists of boolean values.
        if self.image_data.dtype != bool:
            return

        rle_end = np.array(np.append(
            np.where(self.image_data.ravel()[1:] != self.image_data.ravel()[:-1]), np.prod(self.image_dimension) - 1))
        rle_start = np.cumsum(np.append(0, np.diff(np.append(-1, rle_end))))[:-1]
        rle_val = self.image_data.ravel()[rle_start]

        # Check whether the image mask is empty (consists of 0s)
        if np.all(~rle_val):
            self.image_data = None
            self.image_encoded = True
        else:
            # Select only masked parts of the image for further compression
            rle_start = rle_start[rle_val]
            rle_end = rle_end[rle_val]

            # Create zip
            self.image_data = zip(rle_start, rle_end)
            self.image_encoded = True

    def decode_voxel_grid(self, in_place=True):
        # Check if the voxel grid is already decoded.
        if not self.image_encoded:
            if in_place:
                return
            else:
                return self.image_data

        decoded_voxel_grid = np.zeros(np.prod(self.image_dimension), dtype=bool)

        # Check if the image contains masked regions, and unzip them.
        if self.image_data is not None:
            decoding_zip = copy.deepcopy(self.image_data)
            for ii, jj in decoding_zip:
                decoded_voxel_grid[ii:jj + 1] = True

        # Restore shape to original grid.
        decoded_voxel = decoded_voxel_grid.reshape(self.image_dimension)

        if in_place:
            self.image_data = decoded_voxel
            self.image_encoded = False

        else:
            return decoded_voxel

    def set_voxel_grid(self, voxel_grid: np.ndarray):
        self.image_encoded = False
        super().set_voxel_grid(voxel_grid=voxel_grid)

        # Force encoding if the data are boolean.
        if voxel_grid.dtype == bool:
            self.encode_voxel_grid()

    def get_voxel_grid(self) -> None | np.ndarray:
        return self.decode_voxel_grid(in_place=False)

    def update_image_data(self):
        # Do not update image data if the data are absent, it is encoded, or the image is already a boolean mask.
        if self.is_empty() or self.image_encoded or self.image_data.dtype == bool:
            return

        # Ensure that mask consists of boolean values.
        self.image_data = np.around(self.image_data, 6) >= np.around(0.5, 6)
        self.encode_voxel_grid()

    @staticmethod
    def get_interpolation_spline_order(settings: SettingsClass):
        return settings.roi_interpolate.spline_order

    def add_noise(self, **kwargs):
        pass

    def saturate(self, **kwargs):
        pass

    def normalise_intensities(self, **kwargs):
        pass

    def crop(
            self,
            ind_ext_z=None,
            ind_ext_y=None,
            ind_ext_x=None,
            xy_only=False,
            z_only=False):

        if self.image_encoded:
            self.decode_voxel_grid()
            super().crop(
                ind_ext_x=ind_ext_x,
                ind_ext_y=ind_ext_y,
                ind_ext_z=ind_ext_z,
                xy_only=xy_only,
                z_only=z_only
            )
            self.encode_voxel_grid()

        else:
            super().crop(
                ind_ext_x=ind_ext_x,
                ind_ext_y=ind_ext_y,
                ind_ext_z=ind_ext_z,
                xy_only=xy_only,
                z_only=z_only
            )

    def crop_to_size(self, center, crop_size):

        if self.image_encoded:
            self.decode_voxel_grid()
            super().crop_to_size(
                center=center,
                crop_size=crop_size
            )
            self.encode_voxel_grid()

        else:
            super().crop_to_size(
                center=center,
                crop_size=crop_size
            )

    def dilate(
            self,
            by_slice: bool,
            distance: float | None = None,
            voxel_distance: float | None = None
    ):
        from mirp.featureSets.utilities import rep
        import scipy.ndimage as ndi

        # Skip if the mask does not exist
        if self.is_empty():
            return

        # Check if any distance is provided for dilation
        if voxel_distance is None and distance is None:
            return

        if self.separate_slices is not None:
            by_slice = self.separate_slices

        # Check whether voxel are isotropic.
        if not self.is_isotropic(by_slice=by_slice):
            warnings.warn(
                "Non-uniform voxel spacing was detected. Mask dilation requires uniform voxel spacing.", UserWarning
            )

        # Set spacing
        if by_slice:
            spacing = np.array(self.image_spacing)[1, 2]
        else:
            spacing = np.array(self.image_spacing)

        # Derive filter extension and distance
        if distance is not None:
            base_ext: int = np.max([np.floor(distance / np.max(spacing)).astype(int), 0])
        else:
            base_ext: int = int(voxel_distance)
            distance = voxel_distance * np.max(spacing)

        # Check that the dilation size is larger than 1, i.e. base_ext > 0.
        if base_ext <= 0:
            warnings.warn(
                f"Mask was not dilated as the distance ({distance}) was too small compared to voxel spacing ("
                f"{np.max(spacing)})", UserWarning
            )
            return

        # Skip if there is no mask to dilate.
        if self.is_empty_mask():
            return

        # Create displacement map
        df_base = pd.DataFrame({
            "x": rep(
                x=np.arange(-base_ext, base_ext + 1),
                each=(2 * base_ext + 1) * (2 * base_ext + 1),
                times=1),
            "y": rep(
                x=np.arange(-base_ext, base_ext + 1),
                each=2 * base_ext + 1,
                times=2 * base_ext + 1),
            "z": rep(
                x=np.arange(-base_ext, base_ext + 1),
                each=1,
                times=(2 * base_ext + 1) * (2 * base_ext + 1))
        })

        # Calculate distances for displacement map.
        df_base["dist"] = np.sqrt(
            np.sum(np.multiply(df_base.loc[:, ("z", "y", "x")].values, self.image_spacing) ** 2.0, axis=1))

        # Identify elements in range.
        if by_slice:
            df_base["in_range"] = np.logical_and(df_base.dist <= distance, df_base.z == 0)
        else:
            df_base["in_range"] = df_base.dist <= distance

        # Update voxel coordinates to start at [0,0,0].
        df_base.loc[:, ["x", "y", "z"]] -= df_base.loc[0, ["x", "y", "z"]]

        # Generate geometric filter structure.
        geom_struct = np.zeros(
            shape=(np.max(df_base.z) + 1, np.max(df_base.y) + 1, np.max(df_base.x) + 1),
            dtype=bool
        )
        geom_struct[df_base.z.astype(int), df_base.y.astype(int), df_base.x.astype(int)] = df_base.in_range

        # Dilate roi mask and store voxel grid.
        self.set_voxel_grid(voxel_grid=ndi.binary_dilation(
            self.get_voxel_grid(),
            structure=geom_struct,
            iterations=1)
        )

        self.alteration_size = distance

    def erode(
            self,
            by_slice: bool,
            max_eroded_volume_fraction: float = 0.8,
            distance: float | None = None,
            voxel_distance: float | None = None
    ):
        import scipy.ndimage as ndi

        # Skip if the mask does not exist
        if self.is_empty():
            return

        # Check if any distance is provided for dilation
        if voxel_distance is None and distance is None:
            return

        if self.separate_slices is not None:
            by_slice = self.separate_slices

        # Check whether voxel are isotropic.
        if not self.is_isotropic(by_slice=by_slice):
            warnings.warn(
                "Non-uniform voxel spacing was detected. Mask erosion requires uniform voxel spacing.", UserWarning
            )

        # Set spacing.
        if by_slice:
            spacing = np.array(self.image_spacing)[1, 2]
        else:
            spacing = np.array(self.image_spacing)

        # Set geometrical structure. For 2D, the structures in different slices are set to 0.
        geom_struct = ndi.generate_binary_structure(3, 1)
        if by_slice:
            geom_struct[(0, 2), :, :] = False

        # Set number of erosion steps.
        if voxel_distance is None:
            erode_steps = int(np.max([np.round(np.abs(distance) / np.max(spacing)), 0]))
        else:
            erode_steps = int(np.abs(voxel_distance))
            distance = voxel_distance * np.max(spacing)

        # Check that the number of erosion steps is positive.
        if erode_steps <= 0:
            warnings.warn(
                f"Mask was not eroded as the distance ({distance}) was too small compared to voxel spacing ("
                f"{np.max(spacing)})", UserWarning
            )
            return

        # Skip if there is no mask to erode.
        if self.is_empty_mask():
            return

        # Determine initial volume.
        current_mask = previous_mask = self.get_voxel_grid()
        initial_volume = np.sum(previous_mask)

        # Iterate over erosion steps.
        for step in np.arange(0, erode_steps):

            # Perform erosion
            current_mask = ndi.binary_erosion(previous_mask, structure=geom_struct, iterations=1)

            # Calculate volume of the eroded volume
            current_volume = np.sum(current_mask)

            # Stop erosion if the volume shrinks below 80 percent of the original volume due to erosion and voxels
            # from the previous erosion step.
            if current_volume * 1.0 / initial_volume < max_eroded_volume_fraction:
                current_mask = previous_mask
                break
            else:
                previous_mask = current_mask

        # Set updated voxels.
        self.set_voxel_grid(voxel_grid=current_mask)

        self.alteration_size = distance

    def fractional_volume_change(
            self,
            by_slice: bool,
            fractional_change: float | None = None
    ):
        import scipy.ndimage as ndi

        # Skip if the roi does not exist
        if self.is_empty():
            return

        if fractional_change is None or np.around(fractional_change, 6) == 0.0:
            return

        if self.separate_slices is not None:
            by_slice = self.separate_slices

        # Check whether voxel are isotropic.
        if not self.is_isotropic(by_slice=by_slice):
            warnings.warn(
                "Non-uniform voxel spacing was detected. Mask erosion requires uniform voxel spacing.", UserWarning
            )

        # Skip if there is no mask to change.
        if self.is_empty_mask():
            return

        # Set alteration size.
        self.alteration_size = fractional_change

        # Set geometrical structure. For 2D, the structures in different slices are set to 0.
        geom_struct = ndi.generate_binary_structure(3, 1)
        if by_slice:
            geom_struct[(0, 2), :, :] = False

        # Determine original volume
        previous_mask = self.get_voxel_grid()
        initial_volume = np.sum(previous_mask)

        # Iteratively grow or shrink the volume. The loop terminates through break statements
        while True:
            if fractional_change > 0.0:
                current_mask = ndi.binary_dilation(previous_mask, structure=geom_struct, iterations=1)
            else:
                current_mask = ndi.binary_erosion(previous_mask, structure=geom_struct, iterations=1)

            current_volume = np.sum(current_mask)
            if current_volume == 0:
                break

            if 0.0 < fractional_change <= current_volume / initial_volume - 1.0:
                break

            if 0.0 > fractional_change >= current_volume / initial_volume - 1.0:
                break

            # Replace previous mask by the current mask.
            previous_mask = current_mask

        # Start randomiser.
        m = hashlib.sha1(usedforsecurity=False)
        m = self.update_hash(m=m)
        randomiser = np.random.default_rng(int(m.hexdigest(), 16))

        # Randomly add/remove border voxels until desired growth/shrinkage is achieved
        if not current_volume / initial_volume - 1.0 == fractional_change:
            additional_vox = np.abs(int(np.floor(initial_volume * (1.0 + fractional_change) - np.sum(previous_mask))))
            if additional_vox > 0:
                border_voxel_ind = np.array(np.where(np.logical_xor(previous_mask, current_mask)))
                select_ind = randomiser.choice(a=border_voxel_ind.shape[1], size=additional_vox, replace=False)
                border_voxel_ind = border_voxel_ind[:, select_ind]
                if fractional_change > 0.0:
                    previous_mask[border_voxel_ind[0, :], border_voxel_ind[1, :], border_voxel_ind[2, :]] = True
                else:
                    previous_mask[border_voxel_ind[0, :], border_voxel_ind[1, :], border_voxel_ind[2, :]] = False

        # Set the new roi
        self.set_voxel_grid(voxel_grid=previous_mask)

    def randomise_mask(
            self,
            image: GenericImage,
            boundary: float = 25.0,
            repetitions: int = 1,
            intensity_range: tuple[Any, Any] = tuple([np.nan, np.nan]),
            by_slice: bool = False
    ):
        """Use SLIC to randomise the roi based on supervoxels"""
        from scipy.ndimage import binary_closing
        from mirp.imageProcess.utilities import set_intensity_range
        from mirp.imageProcess.cropping import crop

        # Skip if no randomisation is required.
        if repetitions < 1:
            return None

        # Skip if the roi or image do not exist
        if self.is_empty() or image.is_empty():
            return None

        # Skip if there is no mask to change.
        if self.is_empty_mask():
            return None

        if self.separate_slices is not None:
            by_slice = self.separate_slices

        # Crop image and mask to accelerate segmentation process.
        cropped_image, cropped_mask = crop(
            image=image,
            masks=self,
            boundary=boundary,
            xy_only=False,
            z_only=False,
            by_slice=by_slice,
            in_place=False
        )

        # Type hinting.
        cropped_mask: MaskImage = cropped_mask

        # Get supervoxels.
        intensity_range = set_intensity_range(image=image, mask=self, intensity_range=intensity_range)
        image_segments = cropped_image.get_supervoxels(intensity_range=intensity_range)
        overlap_indices, overlap_fractions, overlap_size = cropped_mask.get_supervoxel_overlap(
            image_segments=image_segments
        )

        # Skip if there are no overlapping supervoxels.
        if overlap_indices is None:
            return None

        # Set the highest overlap to 1.0 to ensure selection of at least 1 supervoxel.
        overlap_fractions[np.argmax(overlap_fractions)] = 1.0

        # Always include supervoxels with 90% coverage and always exclude those with less than 20% coverage.
        overlap_fractions[overlap_fractions >= 0.90] = 1.0
        overlap_fractions[overlap_fractions < 0.20] = 0.0

        # Determine grid indices of the resected grid with respect to the original image grid.
        grid_origin = image.to_voxel_coordinates(x=np.array(cropped_image.image_origin))
        grid_origin = grid_origin.astype(int)

        # Initialise list of randomised masks.
        randomised_masks = []

        # Start randomiser.
        m = hashlib.sha1(usedforsecurity=False)
        m = self.update_hash(m=m)
        randomiser = np.random.default_rng(int(m.hexdigest(), 16))

        for ii in range(repetitions):
            # Draw random numbers between 0.0 and 1.0.
            random_inclusion = randomiser.random(size=len(overlap_fractions))

            # Select those segments where the random number is less than the overlap fraction - i.e. the fraction is the
            # probability of selecting the supervoxel.
            included_segments = overlap_indices[np.less(random_inclusion, overlap_fractions)]

            # Replace randomised contour in original roi voxel space.
            new_mask_data = np.zeros(shape=self.image_dimension, dtype=bool)
            new_mask_data[
                grid_origin[0]: grid_origin[0] + cropped_mask.image_dimension[0],
                grid_origin[1]: grid_origin[1] + cropped_mask.image_dimension[1],
                grid_origin[2]: grid_origin[2] + cropped_mask.image_dimension[2]
            ] = np.reshape(np.in1d(np.ravel(image_segments), included_segments), cropped_mask.image_dimension)

            # Apply binary closing to close gaps.
            new_mask_data = binary_closing(input=new_mask_data)

            # Set mask and randomisation ID.
            randomised_mask = self.copy(drop_image=True)
            randomised_mask.set_voxel_grid(new_mask_data)
            randomised_mask.slic_randomisation_id = ii
            randomised_masks += [randomised_mask]

        if len(randomised_masks) == 0:
            return None

        return randomised_masks

    def get_supervoxel_overlap(
            self,
            image_segments: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Determines overlap of supervoxels with other the region of interest"""

        # Return None in case image segments and/or mask are missing
        if image_segments is None or self.is_empty() or self.is_empty_mask():
            return None, None, None

        # Determine labels and the voxel count of the masked image.
        overlap_segment_labels, overlap_size = np.unique(
            np.multiply(image_segments, self.get_voxel_grid()),
            return_counts=True
        )

        # Find supervoxels with any overlap with the mask.
        overlap_size = overlap_size[overlap_segment_labels > 0]
        overlap_segment_labels = overlap_segment_labels[overlap_segment_labels > 0]

        if len(overlap_size) == 0:
            return None, None, None

        # Check the actual size of the segments overlapping with the current contour
        full_segment_size = list(map(lambda x: np.sum([image_segments == x]), overlap_segment_labels))

        # Calculate the fraction of overlap
        overlap_fraction = overlap_size / full_segment_size

        return overlap_segment_labels, overlap_fraction, overlap_size

    def get_bounding_box(self):
        if self.is_empty() or self.is_empty_mask():
            return None, None, None

        z_ind, y_ind, x_ind = np.where(self.get_voxel_grid())

        return (
            tuple([np.min(z_ind), np.max(z_ind)]),
            tuple([np.min(y_ind), np.max(y_ind)]),
            tuple([np.min(x_ind), np.max(x_ind)])
        )

    def get_center_position(self) -> list[Any]:
        """Identify location of the geometric center of the roi."""
        # Return a NaN if no roi is present
        if self.is_empty():
            return [np.nan, np.nan, np.nan]

        # Determine indices of voxels included in the roi
        z_ind, y_ind, x_ind = np.where(self.get_voxel_grid())

        return [np.mean(z_ind), np.mean(y_ind), np.mean(x_ind)]

    def get_file_name_descriptor(self) -> list[str]:
        descriptors = super().get_file_name_descriptor()

        # Alteration size.
        if self.alteration_size is not None and not self.alteration_size == 0.0:
            descriptors += ["vol", str(self.alteration_size)]

        # Supervoxel randomisation id
        if self.slic_randomisation_id is not None:
            descriptors += ["svx", str(self.slic_randomisation_id)]

        return descriptors

    def get_export_attributes(self) -> dict[str, Any]:
        attributes = super().get_export_attributes()

        # Alteration size.
        if self.alteration_size is not None and not self.alteration_size == 0.0:
            attributes.update({"mask_alteration_size": self.alteration_size})

        # Supervoxel randomisation id
        if self.slic_randomisation_id is not None:
            attributes.update({"mask_randomisation_id": self.slic_randomisation_id})

        return attributes
