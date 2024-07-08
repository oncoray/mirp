import numpy as np

from mirp._features.utilities import rep


def coord2Index(x, y, z, dims):
    # Translate coordinates to indices
    index = x + y * dims[2] + z * dims[2] * dims[1]

    # Mark invalid transitions
    index[np.logical_or(x < 0, x >= dims[2])] = -99999
    index[np.logical_or(y < 0, y >= dims[1])] = -99999
    index[np.logical_or(z < 0, z >= dims[0])] = -99999

    return index


def get_intensity_value(x, index, replace_invalid="nan"):
    # Get grey levels or other variables from table on index

    # Initialise placeholder
    read_x = np.zeros(np.shape(x))

    # Read variables for valid indices
    read_x[index >= 0] = x[index[index >= 0]]

    if replace_invalid == "nan":
        # Set variables for invalid indices to nan
        read_x[index < 0] = np.nan

        # Set variables for invalid initial indices to nan
        read_x[np.isnan(x)] = np.nan

    return read_x


def get_neighbour_directions(d=1.8, distance="euclidian", centre=False, complete=False, dim3=True):
    # Defines transitions to neighbour voxels

    # Base transition vector
    trans = np.arange(start=-np.ceil(d), stop=np.ceil(d) + 1)
    n = np.size(trans)

    # Build transition array [z,y,x]
    nbrs = np.array([rep(x=trans, each=1, times=n * n),
                     rep(x=trans, each=n, times=n),
                     rep(x=trans, each=n * n, times=1)], dtype=np.int32)

    # Initiate maintenance index
    index = np.zeros(np.shape(nbrs)[1], dtype=bool)

    ####################################################################################################################
    # Remove neighbours more than distance d from the center
    ####################################################################################################################

    # Manhattan distance
    if distance.lower() in ["manhattan", "l1", "l_1"]:
        index = np.logical_or(index, np.sum(np.abs(nbrs), axis=0) <= d)
    # Eucldian distance
    if distance.lower() in ["euclidian", "l2", "l_2"]:
        index = np.logical_or(index, np.sqrt(np.sum(np.multiply(nbrs, nbrs), axis=0)) <= d)
    # Chebyshev distance
    if distance.lower() in ["chebyshev", "linf", "l_inf"]:
        index = np.logical_or(index, np.max(np.abs(nbrs), axis=0) <= d)

    # Check if centre voxel [0,0,0] should be maintained; False indicates removal
    if not centre:
        index = np.logical_and(index, (np.sum(np.abs(nbrs), axis=0)) > 0)

    # Check if a complete neighbourhood should be returned
    # False indicates that only half of the vectors are returned
    if not complete:
        index[np.arange(start=0, stop=len(index)//2 + 1)] = False

    # Check if neighbourhood should be 3D or 2D
    if not dim3:
        index[nbrs[0, :] != 0] = False

    return nbrs[:, index]


def is_list_all_none(x):
    return all(y is None for y in x)

