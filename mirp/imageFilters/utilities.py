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
    def __init__(self, filter_x, filter_y, filter_z=None,
                 pre_filter_x=None, pre_filter_y=None, pre_filter_z=None):
        self.x = filter_x
        self.y = filter_y
        self.z = filter_z

        self.pr_x = pre_filter_x
        self.pr_y = pre_filter_y
        self.pr_z = pre_filter_z

        # Extend even-sized filters.
        for attr in ["x", "y", "z", "pr_x", "pr_y", "pr_z"]:
            if self.__dict__[attr] is not None:

                # Check if the kernel is even or odd.
                if len(self.__dict__[attr]) % 2 == 0:
                    self.__dict__[attr] = np.append(self.__dict__[attr], 0.0)

    def permute_filters(self, rotational_invariance=True, require_pre_filter=False, as_filter_table=False):

        if require_pre_filter:
            if self.pr_x is None or self.pr_y is None:
                raise ValueError("The pre-filter should be set for all dimensions.")

            if self.z is not None and self.pr_z is None:
                raise ValueError("The pre-filter should have a component in the z-direction.")

            elif self.z is None and self.pr_z is not None:
                raise ValueError("The pre-filter should not have a component in the z-direction.")

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

        if require_pre_filter:
            # Create a pre-filter to derive a table with filter orientations.
            pre_filter_set = FilterSet(filter_x=self.pr_x,
                                       filter_y=self.pr_y,
                                       filter_z=self.pr_z)

            permuted_pre_filters = pre_filter_set.permute_filters(rotational_invariance=rotational_invariance,
                                                                  as_filter_table=True)

            # Update the columns names
            permuted_pre_filters.rename(columns={"x": "pr_x",
                                                 "y": "pr_y",
                                                 "z": "pr_z"},
                                        inplace=True)

            # Join with the permuted_filters table.
            permuted_filters = pd.concat([permuted_pre_filters, permuted_filters], axis=1)

        if as_filter_table:
            return permuted_filters

        # Remove duplicates.
        permuted_filters = permuted_filters.drop_duplicates(ignore_index=True)

        filter_set_list = []
        for ii in range(len(permuted_filters)):
            permuted_filter_set = permuted_filters.loc[ii, :]

            filter_obj = deepcopy(self)

            if require_pre_filter:
                if self.z is None:
                    filter_set_list += [FilterSet(filter_x=filter_obj._translate_filter(permuted_filter_set.x),
                                                  filter_y=filter_obj._translate_filter(permuted_filter_set.y),
                                                  pre_filter_x=filter_obj._translate_filter(permuted_filter_set.pr_x, True),
                                                  pre_filter_y=filter_obj._translate_filter(permuted_filter_set.pr_y, True))]

                else:
                    filter_set_list += [FilterSet(filter_x=filter_obj._translate_filter(permuted_filter_set.x),
                                                  filter_y=filter_obj._translate_filter(permuted_filter_set.y),
                                                  filter_z=filter_obj._translate_filter(permuted_filter_set.z),
                                                  pre_filter_x=filter_obj._translate_filter(permuted_filter_set.pr_x, True),
                                                  pre_filter_y=filter_obj._translate_filter(permuted_filter_set.pr_y, True),
                                                  pre_filter_z=filter_obj._translate_filter(permuted_filter_set.pr_z, True))]

            else:
                if self.z is None:
                    filter_set_list += [FilterSet(filter_x=filter_obj._translate_filter(permuted_filter_set.x),
                                                  filter_y=filter_obj._translate_filter(permuted_filter_set.y))]

                else:
                    filter_set_list += [FilterSet(filter_x=filter_obj._translate_filter(permuted_filter_set.x),
                                                  filter_y=filter_obj._translate_filter(permuted_filter_set.y),
                                                  filter_z=filter_obj._translate_filter(permuted_filter_set.z))]

        return filter_set_list

    def _translate_filter(self, filter_symbol, use_pre_filter=False):

        if filter_symbol == "gx":
            if use_pre_filter:
                return self.pr_x
            else:
                return self.x

        elif filter_symbol == "gy":
            if use_pre_filter:
                return self.pr_y
            else:
                return self.y

        elif filter_symbol == "gz":
            if use_pre_filter:
                return self.pr_z
            else:
                return self.z

        elif filter_symbol == "jgx":
            if use_pre_filter:
                return np.flip(self.pr_x)
            else:
                return np.flip(self.x)

        elif filter_symbol == "jgy":
            if use_pre_filter:
                return np.flip(self.pr_y)
            else:
                return np.flip(self.y)

        elif filter_symbol == "jgz":
            if use_pre_filter:
                return np.flip(self.pr_z)
            else:
                return np.flip(self.z)

        else:
            raise ValueError(f"Encountered unrecognised filter symbol: {filter_symbol}")

    def decompose_filter(self, method="a_trous"):

        if method == "a_trous":
            # Add in 0s for the à trous algorithm

            # Iterate over filters.
            for attr in ["x", "y", "z", "pr_x", "pr_y", "pr_z"]:
                if self.__dict__[attr] is not None:
                    # Strip zeros from tail and head.
                    old_filter_kernel = np.trim_zeros(deepcopy(self.__dict__[attr]))

                    # Create an array of zeros
                    new_filter_kernel = np.zeros(len(old_filter_kernel) * 2 - 1, dtype=np.float)

                    # Place the original filter constants at every second position. This creates a hole (0.0) between
                    # each of the filter constants.
                    new_filter_kernel[::2] = old_filter_kernel

                    # Update the attribute.
                    self.__dict__[attr] = new_filter_kernel

        else:
            raise ValueError(f"Unknown filter decomposition method: {method}")

    def convolve(self, voxel_grid, mode, use_pre_filter=False):

        # Ensure that we work from a local copy of voxel_grid to prevent updating it by reference.
        voxel_grid = deepcopy(voxel_grid)

        if use_pre_filter:
            if self.pr_x is None or self.pr_y is None or (self.z is not None and self.pr_z is None):
                raise ValueError("Pre-filter kernels are expected, but not found.")

            # Apply filter along the z-axis. Note that the voxel grid is stored with z, y, x indexing. Hence the
            # z-axis is the first axis, the y-axis the second, and the x-axis the third.
            if self.pr_z is not None:
                voxel_grid = ndi.convolve1d(voxel_grid, weights=self.pr_z, axis=0, mode=mode)

            # Apply filter along the y-axis.
            voxel_grid = ndi.convolve1d(voxel_grid, weights=self.pr_y, axis=1, mode=mode)

            # Apply filter along the x-axis.
            voxel_grid = ndi.convolve1d(voxel_grid, weights=self.pr_x, axis=2, mode=mode)

        else:
            # Apply filter along the z-axis. Note that the voxel grid is stored with z, y, x indexing. Hence the
            # z-axis is the first axis, the y-axis the second, and the x-axis the third.
            if self.z is not None:
                voxel_grid = ndi.convolve1d(voxel_grid, weights=self.z, axis=0, mode=mode)

            # Apply filter along the y-axis.
            voxel_grid = ndi.convolve1d(voxel_grid, weights=self.y, axis=1, mode=mode)

            # Apply filter along the x-axis.
            voxel_grid = ndi.convolve1d(voxel_grid, weights=self.x, axis=2, mode=mode)

        return voxel_grid
