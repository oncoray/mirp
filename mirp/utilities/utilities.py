import os
import numpy as np


def real_ndim(x: np.ndarray):
    """
    Determine dimensionality of an array based on its shape. This is unlike the ndim method, which shows the number
    of dimensions based on the length of shape.

    :param x: A numpy array.
    :return: The number of dimensions.
    """
    # Compute the number of dimensions by getting the number of dimensions that have length 0 or 1.
    number_dims = x.ndim - sum([current_dim <= 1 for current_dim in x.shape])

    # If all dimensions are 1, dimensionality is technically 0, but 1 in practice.
    if number_dims < 1:
        number_dims = 1

    return number_dims


def random_string(k: int) -> str:
    import string
    from random import choices

    character_list = list(string.ascii_uppercase + string.digits)
    return "".join(choices(character_list, k=k))
