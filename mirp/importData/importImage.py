import os
import os.path
from functools import singledispatch
from itertools import chain
from typing import Union, List

import numpy as np
import pandas as pd

from mirp.importData.importImageDirectory import ImageDirectory
from mirp.importData.importImageFile import ImageFile
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
        mask_sub_folder=None):

    # Generate list of images.
    image_list = import_image(
        image,
        sample_name=sample_name,
        image_name=image_name,
        image_file_type=image_file_type,
        image_modality=image_modality,
        image_sub_folder=image_sub_folder)

    if not isinstance(image_list, list):
        image_list = [image_list]

    # Flatten the list.
    image_list = flatten_list(image_list)

    # TODO: Remove any datasets that are masks.

    # Form the initial set of stacks, i.e. the data as they should be formatted.
    image_stack_list: List[ImageFileStack] = list(_create_image_stack(image_list=image_list,
                                                                      identifiers="extended",
                                                                      drop_contents=False))

    # Tally the number of image stacks that should be generated.
    n_image_stacks = len(image_stack_list)

    # If the provided sample name matches the length of the image stack, update image_list.
    if sample_name is not None and any(image_stack.sample_name is None for image_stack in image_stack_list):
        if not isinstance(sample_name, list):
            sample_name = [sample_name]

        # Override or fill missing names in case the number of provided sample names matches the number of image stacks.
        if len(sample_name) == len(image_stack_list):
            # Update names. This also updates the name in the image list.
            image_list = [image_stack.set_sample_name(sample_name[ii]) for ii, image_stack in
                          enumerate(image_stack_list)]
            # TODO: check that this indeed updates by reference. We need to create an unpack method otherwise.

    # To resolve the issue of having to create unique identifiers for each set, iteratively update identifier data,
    # notably the sample name identifier, to form the expected grouping.
    image_stack_list: List[ImageFileStack] = list(_create_image_stack(image_list=image_list,
                                                                      identifiers="basic",
                                                                      drop_contents=False))

    # Check that everything is correct.
    flag_all_image_stacks_unique = False
    if all(image_stack.check() for image_stack in image_stack_list)\
            and len(image_stack_list) == n_image_stacks:
        flag_all_image_stacks_unique = True

    # Try to update by appending a number to separate stacks with the same name. This is only possible if no sample
    # names are equal to None.
    # if not flag_all_image_stacks_unique\
    #         and not any(image_stack.sample_name is None for image_stack in image_stack_list):

    # Try to update with the name of the directory.

    # Create separate image stacks.
    # image_stack_list: List[ImageFileStack] = list(_create_image_stack(image_list=image_list,
    #                                                                   image_id_table=image_id_table))




    if any(image_stack.sample_name is None for image_stack in image_stack_list):
        ...


def _create_image_stack(image_list: List[ImageFile],
                        identifiers: str = "basic",
                        drop_contents: bool = True):
    """
    Generates image stacks from the image files in image_list. With identifiers=="extended" the stack will be created
    based on all available information. This will always yield
    Otherwise ("basic") only basic information, i.e. the sample name and the modality, will be used.

    :param image_list: List of image files that should be organised.
    :param identifiers: "basic" or "extended"
    :param drop_contents: False or True. Use True to maintain skeletons only and prevent undue copying of (large)
    voxel arrays.
    :return: an image stack.
    """
    # Extract identifiers
    image_id_table = pd.concat([image_file.get_identifiers(style=identifiers)
                                for image_file in image_list], ignore_index=True)

    # Assign grouping identifier for all rows with the same information.
    image_id_table["group_id"] = image_id_table.groupby(image_id_table.columns.values.tolist()).ngroup()

    # Add positional index to the table. This helps relate the information in image_list to the group they belong to.
    image_id_table["list_id"] = np.arange(len(image_list))

    for ii in np.unique(image_id_table["group_id"].values):
        # Find all images that share identifiers.
        proposed_stack = [image_list[jj]
                          for jj in image_id_table.loc[image_id_table["group_id" == ii], :]["list_id"].values]

        if identifiers == "extended":
            # Only DICOM slices may form stacks.
            proposed_dicom_stack = [image_file for image_file in proposed_stack if isinstance(image_file, ImageDicomFile)]

            if len(proposed_dicom_stack) > 0:
                file_stack = ImageFileStack(
                    image_list=proposed_dicom_stack
                )

                yield file_stack

            # Non-DICOM files will form their own individual stacks.
            proposed_non_dicom_stack = [image_file for image_file in proposed_stack if not isinstance(image_file,
                                                                                                      ImageDicomFile)]

            if len(proposed_non_dicom_stack) > 0:
                for non_dicom_image_file in proposed_non_dicom_stack:
                    file_stack = ImageFileStack(
                        image_list=proposed_dicom_stack
                    )

                    yield file_stack

        elif identifiers == "basic":
            yield proposed_stack

        else:
            raise ValueError(f"The identifiers argument is expected to be one of basic and extended.")


@singledispatch
def import_image(image, **kwargs):
    raise NotImplementedError(f"Unsupported type: {type(image)}")


@import_image.register(list)
def _(image: list, sample_name, image_file_type, image_modality, image_sub_folder):
    # List can be anything. Hence, we dispatch import_image for the individual list elements.
    image_list = [import_image(image=current_image,
                               sample_name=sample_name,
                               image_file_type=image_file_type,
                               image_modality=image_modality,
                               image_sub_folder=image_sub_folder)
                  for current_image in image]


@import_image.register(str)
def _(image: str,
      sample_name: Union[None, str, List[str]] = None,
      image_name: Union[None, str] = None,
      image_file_type: Union[None, str] = None,
      image_modality: Union[None, str] = None,
      image_sub_folder: Union[None, str] = None):
    # Image is a string, which could be a path to a xml file, to a csv file, or just a regular
    # path a path to a file, or a path to a directory. Test which it is and then dispatch.

    if image.lower().endswith("xml"):
        ...
    elif image.lower().endswith("csv"):
        ...
    elif os.path.isdir(image):
        return import_image(
            ImageDirectory(
                directory=image,
                sample_name=sample_name,
                image_name=image_name,
                sub_folder=image_sub_folder,
                modality=image_modality,
                file_type=image_file_type))

    elif os.path.exists(image):
        return import_image(
            ImageFile(
                file_path=image,
                sample_name=sample_name,
                image_name=image_name,
                modality=image_modality,
                file_type=image_file_type).create())

    else:
        raise ValueError("The image path does not point to a xml file, a csv file, a valid image file or a directory "
                         "containing imaging.")


@import_image.register(pd.DataFrame)
def _(image: pd.DataFrame,
      image_modality: Union[None, str] = None,
      **kwargs):
    ...


@import_image.register(np.ndarray)
def _(image: np.ndarray,
      sample_name: Union[None, str] = None,
      image_modality: Union[None, str] = None,
      **kwargs):
    ...


@import_image.register(ImageFile)
def _(image: ImageFile,
      **kwargs):

    if not issubclass(type(image), ImageFile):
        image = image.create()

    # Check if the data are consistent.
    image.check(raise_error=True)

    # Complete image data and add identifiers (if any)
    image.complete()

    return image


@import_image.register(ImageDirectory)
def _(image: ImageDirectory,
      **kwargs):

    # Check first if the data are consistent for a directory.
    image.check(raise_error=True)

    # Yield image files.
    image_list = image.create_images()

    # Dispatch to import_image method for
    return [import_image(current_image) for current_image in image_list]
