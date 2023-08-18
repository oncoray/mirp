import copy
import numpy as np
from typing import Optional, Union

from mirp.images.genericImage import GenericImage


class MaskImage(GenericImage):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.image_encoded = False

    def encode_voxel_grid(self):
        # Check if image data are present, or are already encoded.
        if self.image_data is None or self.image_encoded is True:
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
        super().set_voxel_grid(voxel_grid=voxel_grid)

        # Force encoding if the data are boolean.
        if voxel_grid.dtype == bool:
            self.image_encoded = False
            self.encode_voxel_grid()

    def get_voxel_grid(self) -> Union[None, np.ndarray]:
        return self.decode_voxel_grid(in_place=False)

    def update_image_data(self):
        # Do not update image data if the data are absent, it is encoded, or the image is already a boolean mask.
        if self.image_data is None or self.image_encoded or self.image_data.dtype == bool:
            return

        # Ensure that mask consists of boolean values.
        self.image_data = np.around(self.image_data, 6) >= np.around(0.5, 6)
        self.encode_voxel_grid()

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
