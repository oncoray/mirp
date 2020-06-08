import numpy as np
import scipy.ndimage as ndi
import pandas as pd

from copy import deepcopy


def pool_voxel_grids(x1, x2, pooling_method):

    if pooling_method == "max":
        # Perform max pooling by selecting the maximum intensity of each voxel.
        return np.maximum(x1, x2)

    elif pooling_method == "min":
        # Perform min pooling by selecting the minimum intensity of each voxel.
        return np.minimum(x1, x2)

    elif pooling_method in ["mean", "sum"]:
        # Perform mean / sum pooling by summing the intensities of each voxel.
        return np.add(x1, x2)

    else:
        raise ValueError(f"Unknown pooling method encountered: {pooling_method}")


class FilterSet:
    def __init__(self, filter_x, filter_y, filter_z=None):
        self.x = filter_x
        self.y = filter_y
        self.z = filter_z

    def permute_filters(self, rotational_invariance=True):

        # Return an encapsulated version of the object.
        if not rotational_invariance:
            return [self]

        permuted_filters = []

        # Initiate filter strings
        g_x = "gx"
        g_y = "gy"
        g_z = "gz"
        jg_x = "jgx"
        jg_y = "jgy"
        jg_z = "jgz"

        # Test if x and y filters are the same.
        if np.array_equiv(self.x, self.y):
            g_y = g_x
            jg_y = jg_x

        # Test if x and z filters are the same
        if self.z is not None:
            if np.array_equiv(self.x, self.z):
                g_z = g_x
                jg_z = jg_x

        # Test if y and z filters are the same
        if self.z is not None:
            if np.array_equiv(self.y, self.z):
                g_z = g_y
                jg_z = jg_y

        # Test if the x-filter is symmetric.
        if np.array_equiv(self.x, np.flip(self.x)):
            jg_x = g_x

        # Test if the y-filter is symmetric.
        if np.array_equiv(self.y, np.flip(self.y)):
            jg_y = g_y

        # Test if the y-filter is symmetric.
        if self.z is not None:
            if np.array_equiv(self.z, np.flip(self.z)):
                jg_z = g_z

        if self.z is None:
            # 2D right-hand permutations
            permuted_filters += [{"x": g_x, "y": g_y}]
            permuted_filters += [{"x": jg_y, "y": g_x}]
            permuted_filters += [{"x": jg_x, "y": jg_y}]
            permuted_filters += [{"x": g_y, "y": jg_x}]

        else:
            # 3D right-hand permutations
            permuted_filters += [{"x": g_x, "y": g_y, "z": g_z}]
            permuted_filters += [{"x": jg_z, "y": g_y, "z": g_x}]
            permuted_filters += [{"x": jg_x, "y": g_y, "z": jg_z}]
            permuted_filters += [{"x": g_z, "y": g_y, "z": jg_x}]

            permuted_filters += [{"x": g_y, "y": g_z, "z": g_x}]
            permuted_filters += [{"x": g_y, "y": jg_z, "z": jg_x}]
            permuted_filters += [{"x": g_y, "y": jg_x, "z": g_z}]
            permuted_filters += [{"x": jg_x, "y": jg_y, "z": g_z}]

            permuted_filters += [{"x": jg_y, "y": g_x, "z": g_z}]
            permuted_filters += [{"x": jg_z, "y": jg_x, "z": g_y}]
            permuted_filters += [{"x": jg_z, "y": jg_y, "z": jg_x}]
            permuted_filters += [{"x": jg_z, "y": g_x, "z": jg_y}]

            permuted_filters += [{"x": jg_y, "y": jg_x, "z": jg_z}]
            permuted_filters += [{"x": g_x, "y": jg_y, "z": jg_z}]
            permuted_filters += [{"x": g_y, "y": g_x, "z": jg_z}]
            permuted_filters += [{"x": g_z, "y": jg_x, "z": jg_y}]

            permuted_filters += [{"x": g_z, "y": jg_y, "z": g_x}]
            permuted_filters += [{"x": g_z, "y": g_x, "z": g_y}]
            permuted_filters += [{"x": jg_x, "y": g_z, "z": g_y}]
            permuted_filters += [{"x": jg_y, "y": g_z, "z": jg_x}]

            permuted_filters += [{"x": g_x, "y": g_z, "z": jg_y}]
            permuted_filters += [{"x": jg_x, "y": jg_z, "z": jg_y}]
            permuted_filters += [{"x": jg_y, "y": jg_z, "z": g_x}]
            permuted_filters += [{"x": g_x, "y": jg_z, "z": g_y}]

        # Combine filters into a table.
        permuted_filters = pd.DataFrame(permuted_filters)

        # Remove duplicates.
        permuted_filters = permuted_filters.drop_duplicates(ignore_index=True)

        filter_set_list = []
        for ii in range(len(permuted_filters)):
            permuted_filter_set = permuted_filters.loc[ii, :]

            if self.z is None:
                filter_set_list += [FilterSet(filter_x=self._translate_filter(permuted_filter_set.x),
                                              filter_y=self._translate_filter(permuted_filter_set.y))]

            else:
                filter_set_list += [FilterSet(filter_x=self._translate_filter(permuted_filter_set.x),
                                              filter_y=self._translate_filter(permuted_filter_set.y),
                                              filter_z=self._translate_filter(permuted_filter_set.z))]

        return filter_set_list

    def _translate_filter(self, filter_symbol):

        if filter_symbol == "gx":
            return self.x
        elif filter_symbol == "gy":
            return self.y
        elif filter_symbol == "gz":
            return self.z
        elif filter_symbol == "jgx":
            return np.flip(self.x)
        elif filter_symbol == "jgy":
            return np.flip(self.y)
        elif filter_symbol == "jgz":
            return np.flip(self.z)
        else:
            raise ValueError(f"Encountered unrecognised filter symbol: {filter_symbol}")

    def convolve(self, voxel_grid, mode):

        # Ensure that we work from a local copy of voxel_grid to prevent updating it by reference.
        voxel_grid = deepcopy(voxel_grid)

        # Apply filter along the z-axis. Note that the voxel grid is stored with z, y, x indexing. Hence the z-axis is
        # the first axis, the y-axis the second, and the x-axis the third.
        if self.z is not None:
            voxel_grid = ndi.convolve1d(voxel_grid, weights=self.z, axis=0, mode=mode)

        # Apply filter along the y-axis.
        voxel_grid = ndi.convolve1d(voxel_grid, weights=self.y, axis=1, mode=mode)

        # Apply filter along the x-axis.
        voxel_grid = ndi.convolve1d(voxel_grid, weights=self.x, axis=2, mode=mode)

        return voxel_grid
