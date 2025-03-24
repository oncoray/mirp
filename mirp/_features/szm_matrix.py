import copy

import numpy as np
import pandas as pd
import skimage.measure

from mirp._features.texture_matrix import Matrix
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask


class MatrixSZM(Matrix):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Placeholders for derivative values computed using set_values_from_matrix
        # Size zone matrix
        self.sij: pd.DataFrame | None = None

        # Marginal sum over grey levels
        self.si: pd.DataFrame | None = None

        # Marginal sum over zone sizes
        self.sj: pd.DataFrame | None = None

        # Number of zones
        self.n_s: int | None = None

    def compute(
            self,
            image: GenericImage | None = None,
            mask: BaseMask | None = None,
            **kwargs
    ):
        from mirp.utilities.utilities import real_ndim

        # Define neighbour directions
        if self.spatial_method == "3d":
            connectivity = 3
            image = copy.deepcopy(image.get_voxel_grid())
            mask = copy.deepcopy(mask.roi_intensity.get_voxel_grid())

        elif self.spatial_method in ["2d", "2.5d"]:
            connectivity = 2
            image = image.get_voxel_grid()[self.slice_id, :, :]
            mask = mask.roi_intensity.get_voxel_grid()[self.slice_id, :, :]
        else:
            connectivity = -1  # Does nothing, because _spatial_method_error throws an error.
            self._spatial_method_error()

        # Check dimensionality and update connectivity if necessary.
        connectivity = min([connectivity, real_ndim(image)])

        # Set voxels outside roi to 0.0
        image[~mask] = 0.0

        # Count the number of voxels within the roi
        self.n_voxels = np.sum(mask)

        # Label all connected voxels with the same label.
        labelled_image = skimage.measure.label(
            label_image=image,
            background=0,
            connectivity=connectivity
        )

        # Generate data frame
        data = pd.DataFrame({
            "g": np.ravel(image),
            "label_id": np.ravel(labelled_image),
            "in_roi": np.ravel(mask)
        })

        # Remove all non-roi entries and count occurrence of combinations of volume id and grey level
        data = data[data.in_roi].groupby(by=["g", "label_id"]).size().reset_index(name="zone_size")

        # Count the number of co-occurring sizes and grey values
        matrix = data.groupby(by=["g", "zone_size"]).size().reset_index(name="n")

        # Rename columns
        matrix.columns = ["i", "j", "sij"]

        # Add matrix to object
        self.matrix = matrix

    def set_values_from_matrix(self, **kwargs):
        if self.is_empty():
            return

        # Copy of matrix
        self.sij = copy.deepcopy(self.matrix)

        # Sum over grey levels
        self.si = self.sij.groupby(by="i")["sij"].sum().reset_index().rename(columns={"sij": "si"})

        # Sum over zone sizes
        self.sj = self.sij.groupby(by="j")["sij"].sum().reset_index().rename(columns={"sij": "sj"})

        # Number of zones
        self.n_s = np.sum(self.sij.sij)  # Number of size zones

    @staticmethod
    def _get_grouping_columns():
        return ["i", "j"]
