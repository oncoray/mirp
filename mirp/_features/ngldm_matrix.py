import copy

import numpy as np
import pandas as pd

from mirp._features.texture_matrix import Matrix


class MatrixNGLDM(Matrix):

    def __init__(
            self,
            distance: float,
            coarseness: float,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Neighbourhood distance
        self.distance = int(distance)

        # Coarseness
        self.coarseness = coarseness

        # Placeholders for derivative values computed using set_values_from_matrix
        # Neighbourhood grey level dependence matrix
        self.sij: pd.DataFrame | None = None

        # Marginal sum over grey levels
        self.si: pd.DataFrame | None = None

        # Marginal sum over dependence counts
        self.sj: pd.DataFrame | None = None

        # Number of neighbourhoods
        self.n_s: int | None = None

    def compute(
            self,
            data: pd.DataFrame | None = None,
            image_dimension: tuple[int, int, int] | None = None,
            **kwargs
    ):
        # Check if data actually exists
        if data is None:
            return

        # Check if the roi contains any masked voxels. If this is not the case, don't construct the GLRLM.
        if not np.any(data.roi_int_mask):
            return

        if image_dimension is None:
            raise ValueError("image_dimension cannot be None, but may not have been provided in the calling function.")

        # Create local copies of the image table
        if self.spatial_method == "3d":
            data = copy.deepcopy(data)

            directions = list(self._generate_neighbour_direction(
                d=self.distance,
                metric="chebyshev",
                complete=True,
                dim3=True
            ))

        elif self.spatial_method in ["2d", "2.5d"]:
            data = copy.deepcopy(data[data.z == self.slice_id])
            data["index_id"] = np.arange(0, len(data))
            data["z"] = 0
            data = data.reset_index(drop=True)

            directions = list(self._generate_neighbour_direction(
                d=self.distance,
                metric="chebyshev",
                complete=True,
                dim3=False
            ))

        else:
            directions = None
            self._spatial_method_error()

        # Set grey level of voxels outside ROI to NaN
        data.loc[data.roi_int_mask == False, "g"] = np.nan

        # Set the number of voxels
        self.n_voxels = np.sum(data.roi_int_mask.values)

        # Initialise sum of grey levels and number of neighbours.
        data["occur"] = 0

        for direction in directions:
            # Determine potential valid neighbours.
            data["to_index"] = self.coord_to_index(
                x=data.x.values + direction[2],
                y=data.y.values + direction[1],
                z=data.z.values + direction[0],
                dims=image_dimension
            )

            # Get grey level value from transitions
            data["to_g"] = self._lookup_intensity(
                x=data.g.values,
                index=data.to_index.values
            )

            # Determine which voxels have valid neighbours
            valid_index = np.isfinite(data.to_g)

            # Determine co-occurrence within the coarseness
            data.loc[valid_index, "occur"] += ((np.abs(data.to_g - data.g)[valid_index]) <= self.coarseness) * 1

        # Work with voxels within the intensity roi
        data = data[data.roi_int_mask]

        # Drop superfluous columns
        data = data.drop(columns=["index_id", "x", "y", "z", "to_index", "to_g", "roi_int_mask"])

        # Sum s over voxels
        matrix = data.groupby(by=["g", "occur"]).size().reset_index(name="n")
        matrix.columns = ["i", "j", "sij"]

        # Add one to dependency count as features are not defined for k=0
        matrix.j += 1.0

        # Add matrix to object
        self.matrix = matrix

    def set_values_from_matrix(self, **kwargs):
        if self.is_empty():
            return

        # Neighbourhood grey level dependence matrix
        self.sij = copy.deepcopy(self.matrix)

        # Marginal sum over grey levels
        self.si = self.sij.groupby(by="i")["sij"].sum().reset_index().rename(columns={"sij": "si"})

        # Marginal sum over dependence counts
        self.sj = self.sij.groupby(by="j")["sij"].sum().reset_index().rename(columns={"sij": "sj"})

        # Number of neighbourhoods
        self.n_s = np.sum(self.sij.sij)

    @staticmethod
    def _get_grouping_columns():
        return ["i", "j"]
