import copy

import numpy as np
import pandas as pd

from mirp._features.texture_matrix import Matrix


class MatrixNGTDM(Matrix):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Occurrence matrix
        self.pi: pd.DataFrame | None = None

        # Expanded occurrence matrix
        self.pij: pd.DataFrame | None = None

        # Number of intensity bins (grey levels)
        self.n_g: int | None = None

        # Number of intensity bins (grey levels) that are represented
        self.n_p: int | None = None

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
                d=1,
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
                d=1,
                metric="chebyshev",
                complete=True,
                dim3=False
            ))

        else:
            directions = None
            self._spatial_method_error()

        # Set grey level of voxels outside ROI to NaN
        data.loc[data.roi_int_mask == False, "g"] = np.nan

        # Initialise sum of grey levels and number of neighbours.
        data["g_sum"] = 0
        data["n_nbrs"] = 0

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

            # Sum grey level and increase neighbour counter
            data.loc[valid_index, "g_sum"] += data.loc[valid_index, "to_g"]
            data.loc[valid_index, "n_nbrs"] += 1

        # Calculate average neighbourhood grey level
        data["g_nbr_avg"] = data.g_sum / data.n_nbrs

        # Work with voxels without a missing grey value and with a valid neighbourhood
        data = data[np.logical_and(np.isfinite(data.g_nbr_avg), data.roi_int_mask)]

        # Determine contribution to s per voxel
        data["s_sub"] = np.abs(data.g - data.g_nbr_avg)

        # Drop superfluous columns
        data = data.drop(
            labels=["index_id", "x", "y", "z", "g_sum", "n_nbrs", "to_index", "to_g", "g_nbr_avg", "roi_int_mask"],
            axis=1
        )

        # Sum occurrences of grey level transitions after making the matrix symmetric.
        data = data.groupby(by="g")
        matrix = data.sum().join(pd.DataFrame(data.size(), columns=["n"])).reset_index()

        # Rename columns
        matrix.columns = ["i", "s", "n"]

        # Set the number of voxels
        self.n_voxels = np.sum(matrix.n)

        # Add matrix to object
        self.matrix = matrix

    def set_values_from_matrix(self, intensity_range: tuple[int, int], **kwargs):
        from mirp._features.utilities import rep

        if self.is_empty():
            return

        # Neighbourhood occurrence.
        self.pi = copy.deepcopy(self.matrix)
        self.pi["pi"] = self.pi.n / np.sum(self.pi.n)

        # Number of grey levels.
        self.n_g = intensity_range[1] - intensity_range[0] + 1

        # Number of valid grey levels.
        self.n_p = len(self.pi)

        # Neighbourhood occurrence with missing intensity levels.
        levels = np.arange(start=0, stop=self.n_g) + 1.0
        missing_levels = levels[np.logical_not(np.isin(levels, self.pi.i))]
        n_missing = len(missing_levels)
        if n_missing > 0:
            self.pi = pd.concat([
                self.pi,
                pd.DataFrame({
                    "i": missing_levels,
                    "s": np.zeros(n_missing),
                    "n": np.zeros(n_missing),
                    "pi": np.zeros(n_missing)
                })
            ],
                ignore_index=True
            )

        # Compose occurrence correspondence table
        self.pij = copy.deepcopy(self.pi)
        self.pij = self.pij.rename(columns={"s": "si"})
        self.pij = self.pij.iloc[rep(np.arange(start=0, stop=self.n_g), each=self.n_g).astype(int), :]
        self.pij["j"] = rep(self.pi.i, each=1, times=self.n_g)
        self.pij["pj"] = rep(self.pi.pi, each=1, times=self.n_g)
        self.pij["sj"] = rep(self.pi.s, each=1, times=self.n_g)
        self.pij = self.pij.loc[(self.pij.pi > 0) & (self.pij.pj > 0), :].reset_index()

    @staticmethod
    def _get_grouping_columns():
        return ["i", "s"]
