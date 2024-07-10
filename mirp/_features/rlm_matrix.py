import copy

import numpy as np
import pandas as pd

from mirp._features.texture_matrix import DirectionalMatrix


class MatrixRLM(DirectionalMatrix):

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
            data: pd.DataFrame | None,
            image_dimension: tuple[int, int, int] | None = None,
            **kwargs
    ):
        # Check if data actually exists
        if data is None:
            return

        # Check if the roi contains any masked voxels. If this is not the case, don't construct the GLRLM.
        if not np.any(data.roi_int_mask):
            return

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

        # Set the number of voxels
        self.n_voxels = np.sum(data.roi_int_mask.values)

        # Determine update index number for direction
        if ((
                self.direction[2]
                + self.direction[1] * image_dimension[2]
                + self.direction[0] * image_dimension[2] * image_dimension[1]
        ) >= 0):
            direction = self.direction
        else:
            direction = tuple(-x for x in self.direction)

        # Step size
        index_update = (
                direction[2]
                + direction[1] * image_dimension[2]
                + direction[0] * image_dimension[2] * image_dimension[1]
        )

        # Generate information concerning segments
        n_segments = index_update  # Number of segments

        # Check if the number of segments is greater than one
        if n_segments == 0:
            return

        # Nominal segment length
        segment_length = (len(data) - 1) // index_update + 1

        # Initial segment length for transitions (nominal length - 1)
        trans_segment_length = np.tile([segment_length - 1], reps=n_segments)

        # Number of full segments
        full_len_trans = n_segments - n_segments * segment_length + len(data)

        # Update full segments
        trans_segment_length[0:full_len_trans] += 1

        # Create transition vector
        trans_vec = (
                np.tile(
                    np.arange(start=0, stop=len(data), step=index_update),
                    reps=index_update
                )
                + np.repeat(
                    np.arange(start=0, stop=n_segments),
                    repeats=segment_length
                )
            )
        trans_vec = trans_vec[trans_vec < len(data)]

        # Determine valid transitions
        to_index = self.coord_to_index(
            x=data.x.values + direction[2],
            y=data.y.values + direction[1],
            z=data.z.values + direction[0],
            dims=image_dimension
        )

        # Determine which transitions are valid
        end_ind = np.nonzero(to_index[trans_vec] < 0)[0]  # Find transitions that form an endpoint.

        # Get an interspersed array of intensities. Runs are broken up by np.nan
        intensities = np.insert(data.g.values[trans_vec], end_ind + 1, np.nan)

        # Determine run length start and end indices
        rle_end = np.array(np.append(np.where(intensities[1:] != intensities[:-1]), len(intensities) - 1))
        rle_start = np.cumsum(np.append(0, np.diff(np.append(-1, rle_end))))[:-1]

        # Generate matrix
        matrix = pd.DataFrame({
            "i": intensities[rle_start],
            "r": rle_end - rle_start + 1
        })
        matrix = matrix.loc[~np.isnan(matrix.i), :]
        matrix = matrix.groupby(by=["i", "j"]).size().reset_index(name="rij")

        # Add matrix to object
        self.matrix = matrix

    def set_values_from_matrix(self):
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
