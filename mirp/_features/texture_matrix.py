import copy

import pandas as pd
import numpy as np
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from typing import Generator

from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask


class Matrix(object):
    def __init__(
            self,
            spatial_method: str | None,
            matrix: pd.DataFrame | None = None,
            slice_id: int | None = None,
            n_voxels: int | None = None
    ):
        # Placeholder for the actual matrix.
        self.matrix = matrix

        # Spatial method for aggregating and merging matrices.
        self.spatial_method = spatial_method

        # Slice ID that is required for merging matrices within a slice.
        self.slice_id = slice_id

        # Number of voxels represented in the matrix.
        self.n_voxels = n_voxels

    def copy(self):
        return copy.deepcopy(self)

    def is_empty(self):
        return self.matrix is None or len(self.matrix) == 0

    def compute(
            self,
            data: pd.DataFrame,
            image_dimension: tuple[int, int, int] | None = None,
            **kwargs
    ):
        raise NotImplementedError("Implement in subclasses.")

    def set_values_from_matrix(self):
        raise NotImplementedError("Implement in subclasses.")

    def generate(self, prototype, n_slices: int, **kwargs) -> Generator[Self, None, None]:
        if self.spatial_method in ["2d", "2.5d"]:
            # Extract texture matrices from 2D slices.
            for slice_id in np.arange(0, n_slices):
                yield prototype(
                    spatial_method=self.spatial_method,
                    slice_id=slice_id,
                    **kwargs
                )

        elif self.spatial_method == "3d":
            # Extract texture matrix for the whole 3D volume.
            yield prototype(
                spatial_method=self.spatial_method,
                **kwargs
            )

        else:
            raise ValueError(
                f"One of  \"2d\", \"2.5d\" or \"3d\" is expected as spatial method. Found: {self.spatial_method}."
            )

    def merge(
            self,
            matrix_list: list[Self],
            prototype,
            **kwargs
    ) -> list[Self]
    
        if self.spatial_method == "2d":
            # Average features over slices: maintain original 2D texture matrices.
            return [matrix for matrix in matrix_list if not matrix.is_empty()]

        elif self.spatial_method in ["2.5d", "3d"]:
            # Merge 2D or 3D texture matrices into a single matrix.

            # Remove all empty matrices.
            updated_matrix_list = [matrix for matrix in matrix_list if not matrix.is_empty()]
            if len(updated_matrix_list) == 0:
                return []

            collected_matrices = [matrix.matrix for matrix in updated_matrix_list]
            n_voxels = np.sum([matrix.n_voxels for matrix in updated_matrix_list])

            # Merge collected matrices.
            merged_matrix = pd.concat(collected_matrices, axis=0)
            merged_matrix = merged_matrix.groupby(by=self._get_grouping_columns()).sum().reset_index()

            return [prototype(
                spatial_method=self.spatial_method,
                matrix=merged_matrix,
                n_voxels=n_voxels,
                **kwargs
            )]

    @staticmethod
    def _get_grouping_columns() -> list[str, ...]:
        raise NotImplementedError("Implement in subclasses.")

    @staticmethod
    def coord_to_index(x, y, z, dims):
        # Translate coordinates to indices
        index = x + y * dims[2] + z * dims[2] * dims[1]

        # Mark invalid transitions
        index[np.logical_or(x < 0, x >= dims[2])] = -99999
        index[np.logical_or(y < 0, y >= dims[1])] = -99999
        index[np.logical_or(z < 0, z >= dims[0])] = -99999

        return index

    @staticmethod
    def _generate_neighbour_direction(
            d: float = 1.8,
            metric: str = "euclidian",
            keep_centre: bool = False,
            complete: bool = False,
            dim3: bool = True
    ) -> Generator[tuple[int, int, int], None, None]:
        from mirp._features.utilities import rep

        # Base transition vector
        trans = np.arange(start=-np.ceil(d), stop=np.ceil(d) + 1)
        n = np.size(trans)

        # Build transition array [z,y,x]
        nbrs = np.array([
            rep(x=trans, each=1, times=n * n),
            rep(x=trans, each=n, times=n),
            rep(x=trans, each=n * n, times=1)
        ], dtype=np.int32)

        # Initiate maintenance index
        index = np.zeros(np.shape(nbrs)[1], dtype=bool)

        # Remove neighbours more than distance d from the center.
        if metric.lower() in ["manhattan", "l1", "l_1"]:
            # Manhattan distance
            index = np.logical_or(index, np.sum(np.abs(nbrs), axis=0) <= d)
        elif metric.lower() in ["euclidian", "l2", "l_2"]:
            # Eucldian distance
            index = np.logical_or(index, np.sqrt(np.sum(np.multiply(nbrs, nbrs), axis=0)) <= d)
        elif metric in ["chebyshev", "linf", "l_inf"]:
            # Chebyshev distance
            index = np.logical_or(index, np.max(np.abs(nbrs), axis=0) <= d)
        else:
            raise ValueError(f"Did not recognize distance metric: {metric}")

        # Check if centre voxel [0,0,0] should be maintained; False indicates removal
        if not keep_centre:
            index = np.logical_and(index, (np.sum(np.abs(nbrs), axis=0)) > 0)

        # Check if a complete neighbourhood should be returned
        # False indicates that only half of the vectors are returned
        if not complete:
            index[np.arange(start=0, stop=len(index)//2 + 1)] = False

        # Check if neighbourhood should be 3D or 2D
        if not dim3:
            index[nbrs[0, :] != 0] = False

        for ii, flag in enumerate(index):
            if flag:
                yield tuple(nbrs[:, ii].flatten())

    def _spatial_method_error(self):
        raise ValueError(
            f"One of  \"2d\", \"2.5d\" or \"3d\" is expected as spatial method. Found: {self.spatial_method}."
        )


class DirectionalMatrix(Matrix):

    def __init__(
            self,
            direction_id: int | None = None,
            direction: tuple[int, int, int] | None = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        # Set the neighbourhood directions and its index.
        self.direction_id = direction_id
        self.direction = direction

    def generate(self, prototype, n_slices: int, **kwargs) -> Generator[Self, None, None]:
        if self.spatial_method in ["2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge"]:
            # Extract directional texture matrices from 2D slices.
            for slice_id in np.arange(0, n_slices):
                for direction_id, direction in enumerate(self._generate_neighbour_direction(
                    d=1,
                    metric="chebyshev",
                    dim3=False
                )):
                    yield prototype(
                        spatial_method=self.spatial_method,
                        slice_id=slice_id,
                        direction=direction,
                        direction_id=direction_id,
                        **kwargs
                    )

        elif self.spatial_method in ["3d_average", "3d_volume_merge"]:
            # Extract texture matrices for the whole 3D volume.
            for direction_id, direction in enumerate(self._generate_neighbour_direction(
                    d=1,
                    metric="chebyshev",
                    dim3=True
            )):
                yield prototype(
                    spatial_method=self.spatial_method,
                    slice_id=None,
                    direction=direction,
                    direction_id=direction_id,
                    **kwargs
                )

        else:
            self._spatial_method_error()

    def merge(
            self,
            matrix_list: list[Self],
            prototype,
            **kwargs
    ) -> list[Self]:

        # Remove empty matrices.
        matrix_list = [matrix for matrix in matrix_list if not matrix.is_empty()]
        if len(matrix_list) == 0:
            return []

        if self.spatial_method in ["2d_average", "3d_average"]:
            # Average features over directions: maintain original directional texture matrices.
            return matrix_list

        elif self.spatial_method == "2d_slice_merge":
            # Merge directional matrices within each slice.
            out_matrix_list = []

            # Identify slice ids and iterate over unique slice identifiers.
            slice_ids = [matrix.slice_id for matrix in matrix_list]
            for slice_id in np.unique(slice_ids):
                slice_matrix_list = [matrix for matrix in matrix_list if matrix.slice_id == slice_id]
                collected_matrices = [matrix.matrix for matrix in slice_matrix_list]
                n_voxels = np.sum([matrix.n_voxels for matrix in slice_matrix_list])

                merged_matrix = pd.concat(collected_matrices, axis=0)
                merged_matrix = merged_matrix.groupby(by=self._get_grouping_columns()).sum().reset_index()

                out_matrix_list += [prototype(
                    spatial_method=self.spatial_method,
                    matrix=merged_matrix,
                    n_voxels=n_voxels,
                    **kwargs
                )]

            return out_matrix_list

        elif self.spatial_method == "2.5d_direction_merge":
            # Merge directional matrices for each direction.
            out_matrix_list = []

            # Identify direction ids and iterate over unique direction identifiers.
            direction_ids = [matrix.direction_id for matrix in matrix_list]
            for direction_id in np.unique(direction_ids):
                direction_matrix_list = [matrix for matrix in matrix_list if matrix.direction_id == direction_id]
                collected_matrices = [matrix.matrix for matrix in direction_matrix_list]
                n_voxels = np.sum([matrix.n_voxels for matrix in direction_matrix_list])

                merged_matrix = pd.concat(collected_matrices, axis=0)
                merged_matrix = merged_matrix.groupby(by=self._get_grouping_columns()).sum().reset_index()

                out_matrix_list += [prototype(
                    spatial_method=self.spatial_method,
                    matrix=merged_matrix,
                    n_voxels=n_voxels,
                    **kwargs
                )]

            return out_matrix_list

        elif self.spatial_method in ["2.5d_volume_merge", "3d_volume_merge"]:
            # Merge 2D or 3D texture matrices into a single matrix.

            # Remove all empty matrices.
            updated_matrix_list = [matrix for matrix in matrix_list if not matrix.is_empty()]
            if len(updated_matrix_list) == 0:
                return []

            collected_matrices = [matrix.matrix for matrix in updated_matrix_list]
            n_voxels = np.sum([matrix.n_voxels for matrix in updated_matrix_list])

            # Merge collected matrices.
            merged_matrix = pd.concat(collected_matrices, axis=0)
            merged_matrix = merged_matrix.groupby(by=self._get_grouping_columns()).sum().reset_index()

            return [prototype(
                spatial_method=self.spatial_method,
                matrix=merged_matrix,
                n_voxels=n_voxels,
                **kwargs
            )]

        else:
            self._spatial_method_error()

    def _spatial_method_error(self):
        raise ValueError(
            f"One of  \"2d_average\", \"2d_slice_merge\", \"2.5d_direction_merge\", \"2.5d_volume_merge\", "
            f"\"3d_average\" or \"3d_volume_merge\" is expected as spatial method. Found: {self.spatial_method}."
        )
