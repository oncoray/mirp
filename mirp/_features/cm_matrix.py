import copy

import numpy as np
import pandas as pd

from mirp._features.texture_matrix import DirectionalMatrix


class MatrixCM(DirectionalMatrix):

    def __init__(self, distance: int, **kwargs):
        super().__init__(**kwargs)

        # Set distance
        self.distance = int(distance)

        # Placeholders for derivative values computed using set_values_from_matrix
        # Co-occurrence matrix, expressed as probabilities.
        self.pij: pd.DataFrame | None = None

        # Marginal sum over grey levels of originating pixels i
        self.pi: pd.DataFrame | None = None

        # Marginal sum over grey levels of target pixels j
        self.pj: pd.DataFrame | None = None

        # Diagonal probabilities
        self.pimj: pd.DataFrame | None = None

        # Cross-diagonal probabilities
        self.pipj: pd.DataFrame | None = None

        # Number of intensities bins (grey levels)
        self.n_g: int | None = None

        # Mean co-occurrence weighted intensity
        self.mu: float | None = None

        # Marginal co-occurrence weighted intensity mean
        self.mu_marg: float | None = None

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
        if self.spatial_method in ["3d_average", "3d_volume_merge"]:
            data = copy.deepcopy(data)

        elif self.spatial_method in ["2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge"]:
            data = copy.deepcopy(data[data.z == self.slice_id])
            data["index_id"] = np.arange(0, len(data))
            data["z"] = 0
            data = data.reset_index(drop=True)

        else:
            self._spatial_method_error()

        # Set grey level of voxels outside ROI to NaN
        data.loc[data.roi_int_mask == False, "g"] = np.nan

        # Determine potential transitions. The distance parameter determines the lookup distance.
        data["to_index"] = self.coord_to_index(
            x=data.x.values + self.direction[2] * self.distance,
            y=data.y.values + self.direction[1] * self.distance,
            z=data.z.values + self.direction[0] * self.distance,
            dims=image_dimension
        )

        # Look up intensity of neighbours.
        data["to_g"] = self._lookup_intensity(
            x=data.g.values,
            index=data.to_index.values
        )

        # Check if any valid neighbours were found.
        if np.all(np.isnan(data[["to_g"]])):
            return

        # Count occurrences intensity combinations.
        data = data.groupby(by=["g", "to_g"]).size().reset_index(name="n")

        # We assume symmetric GLCM. Hence, append grey level transitions in opposite direction.
        additional_data = pd.DataFrame({
            "g": data.to_g,
            "to_g": data.g,
            "n": data.n
        })
        data = pd.concat([data, additional_data], ignore_index=True)

        # Sum occurrences of grey level transitions after making the matrix symmetric.
        matrix = data.groupby(by=["g", "to_g"]).sum().reset_index()

        # Rename columns
        matrix.columns = ["i", "j", "n"]

        # Set the number of voxels
        self.n_voxels = np.sum(matrix.n)

        # Add matrix to object
        self.matrix = matrix

    def set_values_from_matrix(self, intensity_range: tuple[int, int], **kwargs):
        if self.is_empty():
            return

        self.pij = copy.deepcopy(self.matrix)
        self.pij["pij"] = self.pij.n / sum(self.pij.n)
        self.pi = self.pij.groupby(by="i")["pij"].sum().reset_index().rename(columns={"pij": "pi"})
        self.pj = self.pij.groupby(by="j")["pij"].sum().reset_index().rename(columns={"pij": "pj"})

        # Diagonal probabilities p(i-j)
        self.pimj = copy.deepcopy(self.pij)
        self.pimj["k"] = np.abs(self.pimj.i - self.pimj.j)
        self.pimj = self.pimj.groupby(by="k")["pij"].sum().reset_index().rename(columns={"pij": "pimj"})

        # Cross-diagonal probabilities p(i+j)
        self.pipj = copy.deepcopy(self.pij)
        self.pipj["k"] = self.pipj.i + self.pipj.j
        self.pipj = self.pipj.groupby(by="k")["pij"].sum().reset_index().rename(columns={"pij": "pipj"})

        # Merger of df.p_ij, df.p_i and df.p_j
        self.pij = pd.merge(self.pij, self.pi, on="i")
        self.pij = pd.merge(self.pij, self.pj, on="j")

        # Number of grey levels
        self.n_g = intensity_range[1] - intensity_range[0] + 1

        # Mean co-occurrence weighted intensity
        self.mu = np.sum(self.pij.i * self.pij.pij)

        # Marginal co-occurrence weighted intensity mean
        self.mu_marg = np.sum(self.pi.i * self.pi.pi)

    @staticmethod
    def _get_grouping_columns():
        return ["i", "j"]
