import os
import os.path
from functools import singledispatch
from typing import Union, List

import numpy as np
import pandas as pd

from mirp.importData.importImageDirectory import ImageDirectory
from mirp.importData.importImageFile import ImageFile, MaskFile
from mirp.importData.importImageDicomFile import ImageDicomFile
from mirp.importData.importImageFileStack import ImageFileStack
from mirp.importData.utilities import flatten_list


def import_data(
        image,
        sample_name: Union[None, str, List[str]] = None,
        image_name=None,
        image_file_type=None,
        image_modality=None,
        image_sub_folder=None,
        mask=None,
        mask_name=None,
        mask_sub_folder=None):

    # Generate list of images.
    image_list, mask_list = import_image_and_mask(
        image,
        mask,
        sample_name=sample_name,
        image_name=image_name,
        image_file_type=image_file_type,
        image_modality=image_modality,
        image_sub_folder=image_sub_folder,
        mask_name=mask_name,
        mask_sub_folder=mask_sub_folder
    )

    if not isinstance(image_list, list):
        image_list = [image_list]

    # Flatten the list.
    image_list = flatten_list(image_list)

    # Form the initial set of stacks, i.e. the data as they should be formatted.
    image_stack_list: List[ImageFileStack] = list(_create_image_stack(
        image_list=image_list,
        drop_contents=False))



def _create_image_stack(
        image_list: List[ImageFile],
        drop_contents: bool = True):
    """
    Generates image stacks from the image files in image_list. With identifiers=="extended" the stack will be created
    based on all available information. This will always yield
    Otherwise ("basic") only basic information, i.e. the sample name and the modality, will be used.

    :param image_list: List of image files that should be organised.
    :param drop_contents: False or True. Use True to maintain skeletons only and prevent undue copying of (large)
    voxel arrays.
    :return: an image stack.
    """
    # Extract identifiers
    image_id_table = pd.concat(
        [image_file.get_identifiers() for image_file in image_list],
        ignore_index=True)

    # Assign grouping identifier for all rows with the same information.
    image_id_table["group_id"] = image_id_table.groupby(image_id_table.columns.values.tolist()).ngroup()

    # Add positional index to the table. This helps relate the information in image_list to the group they belong to.
    image_id_table["list_id"] = np.arange(len(image_list))

    for ii in np.unique(image_id_table["group_id"].values):
        # Find all images that share identifiers.
        proposed_stack = [
            image_list[jj] for jj in image_id_table.loc[image_id_table["group_id"].values == ii, :]["list_id"].values
        ]

        proposed_dicom_stack = [image_file for image_file in proposed_stack if isinstance(image_file, ImageDicomFile)]

        if len(proposed_dicom_stack) > 0:
            file_stack = ImageFileStack(image_list=proposed_dicom_stack)

            yield file_stack

        # Non-DICOM files will form their own individual stacks.
        proposed_non_dicom_stack = [
            image_file for image_file in proposed_stack if not isinstance(image_file, ImageDicomFile)
        ]

        if len(proposed_non_dicom_stack) > 0:
            for non_dicom_image_file in proposed_non_dicom_stack:
                file_stack = ImageFileStack(image_list=[non_dicom_image_file])

                yield file_stack


def import_image_and_mask(image, mask, **kwargs):
    image_list = import_image(
        image=image,
        **kwargs)

    mask_list = import_mask(
        mask=mask,
        **kwargs)

    return image_list, mask_list


@singledispatch
def import_image(image, **kwargs):
    raise NotImplementedError(f"Unsupported image type: {type(image)}")


@singledispatch
def import_mask(mask, **kwargs):
    raise NotImplementedError(f"Unsupported mask type: {type(mask)}")


@import_image.register(list)
def _(image: list, **kwargs):
    # List can be anything. Hence, we dispatch import_image for the individual list elements.
    image_list = [import_image(
        image=current_image,
        **kwargs
    ) for current_image in image]

    return image_list


@import_mask.register(list)
def _(mask: list, **kwargs):
    mask_list = import_mask(
        mask=mask,
        **kwargs)

    return mask_list


@import_image.register(str)
def _(image: str, **kwargs):
    # Image is a string, which could be a path to a xml file, to a csv file, or just a regular
    # path a path to a file, or a path to a directory. Test which it is and then dispatch.

    if image.lower().endswith("xml"):
        ...

    elif image.lower().endswith("csv"):
        ...

    elif os.path.isdir(image):
        return import_image(
            ImageDirectory(directory=image, **kwargs))

    elif os.path.exists(image):
        return import_image(
            ImageFile(file_path=image, **kwargs).create())

    else:
        raise ValueError("The image path does not point to a xml file, a csv file, a valid image file or a directory "
                         "containing imaging.")


@import_mask.register(list)
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


@import_image.register(pd.DataFrame)
def _(image: pd.DataFrame,
      image_modality: Union[None, str] = None,
      **kwargs):
    ...


@import_mask.register(pd.DataFrame)
def _(mask: pd.DataFrame, **kwargs):
    ...


@import_image.register(np.ndarray)
def _(image: np.ndarray,
      sample_name: Union[None, str] = None,
      image_modality: Union[None, str] = None,
      **kwargs):
    ...


@import_mask.register(np.ndarray)
def _(image: np.ndarray, **kwargs):
    ...


@import_image.register(ImageFile)
def _(image: ImageFile, **kwargs):

    if not issubclass(type(image), ImageFile):
        image = image.create()

    # Check if the data are consistent.
    image.check(raise_error=True)

    # Complete image data and add identifiers (if any)
    image.complete()

    return image


@import_mask.register(MaskFile)
def _(image: MaskFile, **kwargs):
    ...


@import_image.register(ImageDirectory)
def _(image: ImageDirectory, **kwargs):

    # Check first if the data are consistent for a directory.
    image.check(raise_error=True)

    # Yield image files.
    image_list = image.create_images()

    # Dispatch to import_image method for
    return [import_image(current_image) for current_image in image_list]


@import_mask.register(MaskDirectory)
def _(image: MaskDirectory, **kwargs):
    ...