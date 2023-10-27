import os
import numpy as np


def check_string(input_string):
    """Updates string characters that may lead to errors when written."""
    input_string = input_string.replace(" ", "_")
    input_string = input_string.replace(",", "_")
    input_string = input_string.replace(";", "_")
    input_string = input_string.replace(":", "_")
    input_string = input_string.replace("\"", "_")
    input_string = input_string.replace("=", "_equal_")
    input_string = input_string.replace(">", "_greater_")
    input_string = input_string.replace("<", "_smaller_")
    input_string = input_string.replace("&", "_and_")
    input_string = input_string.replace("|", "_or_")

    return input_string


def get_version():
    with open(os.path.join("../..", 'VERSION.txt')) as version_file:
        version = version_file.read().strip()

    return version


def makedirs_check(path):
    """
    Checks if the given path is an existing directory
    structure, otherwise creates it.
    """

    if not os.path.isdir(path):
        os.makedirs(path)


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
