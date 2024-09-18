import copy

import numpy as np
import pandas as pd
import skimage.measure

from mirp._features.texture_matrix import Matrix
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask


class MatrixDZM(Matrix):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Placeholders for derivative values computed using set_values_from_matrix
        # Distance zone matrix
        self.dij: pd.DataFrame | None = None

        # Marginal sum over grey levels
        self.di: pd.DataFrame | None = None

        # Marginal sum over distances
        self.dj: pd.DataFrame | None = None

        # Number of distance zones
        self.n_s: int | None = None

    def compute(
            self,
            data: pd.DataFrame | None = None,
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
            data = copy.deepcopy(data)

        elif self.spatial_method in ["2d", "2.5d"]:
            connectivity = 2
            image = image.get_voxel_grid()[self.slice_id, :, :]
            mask = mask.roi_intensity.get_voxel_grid()[self.slice_id, :, :]
            data = copy.deepcopy(data[data.z == self.slice_id])

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
            background=0.0,
            connectivity=connectivity
        )

        data["label_id"] = np.ravel(labelled_image)

        # Select minimum group distance for unique groups
        data = data[data.roi_int_mask].groupby(by=["g", "label_id"])["border_distance"].min().reset_index().rename(
            columns={"border_distance": "d"}
        )

        # Count occurrence of grey level and distance
        matrix = data.groupby(by=["g", "d"]).size().reset_index(name="dij")

        # Rename columns
        matrix.columns = ["i", "j", "dij"]

        # Add matrix to object
        self.matrix = matrix

    def set_values_from_matrix(self, **kwargs):
        if self.is_empty():
            return

        # Copy of matrix
        self.dij = copy.deepcopy(self.matrix)

        # Sum over grey levels
        self.di = self.dij.groupby(by="i")["dij"].sum().reset_index().rename(columns={"dij": "di"})

        # Sum over distances
        self.dj = self.dij.groupby(by="j")["dij"].sum().reset_index().rename(columns={"dij": "dj"})

        # Number of distance zones
        self.n_s = np.sum(self.dij.dij)

    @staticmethod
    def _get_grouping_columns():
        return ["i", "j"]
