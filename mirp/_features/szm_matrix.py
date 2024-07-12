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
        # Run-length matrix
        self.rij: pd.DataFrame | None = None

        # Marginal sum over grey levels
        self.ri: pd.DataFrame | None = None

        # Marginal sum over run lengths
        self.rj: pd.DataFrame | None = None

        # Number of runs
        self.n_s: int | None = None

    def compute(
            self,
            image: GenericImage | None = None,
            mask: BaseMask | None = None,
            **kwargs
    ):
        from mirp.utilities.utilities import real_ndim

        if image is None:
            raise ValueError(
                "image cannot be None, but may not have been provided in the calling function."
            )
        if mask is None:
            raise ValueError(
                "mask cannot be None, but may not have been provided in the calling function."
            )

        # Check if data actually exists
        if image.is_empty() or mask.roi_intensity.is_empty_mask():
            return

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
            background=0.0,
            connectivity=connectivity
        )

        # Generate data frame
        df_szm = pd.DataFrame({
            "g": np.ravel(image),
            "label_id": np.ravel(labelled_image),
            "in_roi": np.ravel(mask)
        })

        # Remove all non-roi entries and count occurrence of combinations of volume id and grey level
        df_szm = df_szm[df_szm.in_roi].groupby(by=["g", "label_id"]).size().reset_index(name="zone_size")

        # Count the number of co-occurring sizes and grey values
        matrix = df_szm.groupby(by=["g", "zone_size"]).size().reset_index(name="n")

        # Rename columns
        matrix.columns = ["i", "s", "n"]

        # Add matrix to object
        self.matrix = matrix

    def set_values_from_matrix(self, **kwargs):
        if self.is_empty():
            return

        # Copy of matrix
        self.rij = copy.deepcopy(self.matrix)

        # Sum over grey levels
        self.ri = self.matrix.groupby(by="i")["rij"].sum().reset_index().rename(columns={"rij": "ri"})

        # Sum over run lengths
        self.rj = self.matrix.groupby(by="j")["rij"].sum().reset_index().rename(columns={"rij": "rj"})

        # Constant definitions
        self.n_s = np.sum(self.matrix.rij) * 1.0  # Number of runs

    @staticmethod
    def _get_grouping_columns():
        return ["i", "j"]
