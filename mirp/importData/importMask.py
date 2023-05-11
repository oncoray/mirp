from functools import singledispatch

import os.path
import numpy as np
import pandas as pd


@singledispatch
def import_mask(mask, **kwargs):
    raise NotImplementedError(f"Unsupported mask type: {type(mask)}")


@import_mask.register(list)
def _(mask: list, **kwargs):
    mask_list = import_mask(
        mask=mask,
        **kwargs)

    return mask_list


@import_mask.register(str)
def _(mask: str, **kwargs):
    # Mask is a string, which could be a path to a xml file, to a csv file, or just a regular
    # path a path to a file, or a path to a directory. Test which it is and then dispatch.

    if mask.lower().endswith("xml"):
        ...

    elif mask.lower().endswith("csv"):
        ...

    elif os.path.isdir(mask):
        return import_mask(
            MaskDirectory(directory=mask, **kwargs))

    elif os.path.exists(mask):
        return import_mask(
            MaskFile(file_path=mask, **kwargs).create())

    else:
        raise ValueError("The mask path does not point to a xml file, a csv file, a valid image file or a directory "
                         "containing imaging.")


@import_mask.register(pd.DataFrame)
def _(mask: pd.DataFrame, **kwargs):
    ...


@import_mask.register(np.ndarray)
def _(mask: np.ndarray, **kwargs):
    ...


@import_mask.register(MaskFile)
def _(mask: MaskFile, **kwargs):
    ...


@import_mask.register(MaskDirectory)
def _(mask: MaskDirectory, **kwargs):

    # Check first if the data are consistent.
    mask.check(raise_error=True)

    # Yield image files
