import os
import os.path
from functools import singledispatch
from typing import Union, List

import numpy as np
import pandas as pd

from mirp.importData.imageDirectory import ImageDirectory
from mirp.importData.imageGenericFile import ImageFile
from mirp.importData.utilities import supported_file_types, supported_image_modalities


def import_image(
        image,
        sample_name: Union[None, str, List[str]] = None,
        image_name: Union[None, str, List[str]] = None,
        image_file_type: Union[None, str] = None,
        image_modality: Union[None, str, List[str]] = None,
        image_sub_folder: Union[None, str] = None,
        stack_images: str = "auto"
) -> List[ImageFile]:
    """
    Import image files that refer to image files. Actual image data are generally not loaded.
    :param image: A path to an image file, a path to a directory, a path to a config_data.xml file, a path to
    :param sample_name:
    :param image_name:
    :param image_file_type:
    :param image_modality:
    :param image_sub_folder:
    :param stack_images: One of auto, yes or no. If image files in the same directory cannot be assigned to
    different samples, and are 2D (slices) of the same size, they might belong to the same 3D image stack. "auto"
    will stack 2D numpy arrays, but not other file types. "yes" will stack all files that contain 2D images,
    that have the same dimensions, orientation and spacing, except for DICOM files. "no" will not stack any files.
    DICOM files ignore this argument, because their stacking can be determined from metadata.
    :return: list of image files.
    """
    # Check modality.
    if image_modality is not None:
        if not isinstance(image_modality, str):
            raise TypeError(
                f"The image_modality argument is expected to be a single character string or None. The following "
                f"modalities are supported: {', '.join(supported_image_modalities(None))}.")
        _ = supported_image_modalities(image_modality.lower())

    # Check image_file_type.
    if image_file_type is not None:
        if not isinstance(image_file_type, str):
            raise TypeError(
                f"The image_file_type argument is expected to be a single character string, or None. The following file "
                f"types are supported: {', '.join(supported_file_types(None))}.")
        _ = supported_file_types(image_file_type)

    # Check stack_images
    if stack_images not in ["yes", "auto", "no"]:
        raise ValueError(
            f"The stack_images argument is expected to be one of yes, auto, or no. Found: {stack_images}."
        )

    image_list = _import_image(
        image,
        sample_name=sample_name,
        image_name=image_name,
        image_file_type=image_file_type,
        image_modality=image_modality,
        image_sub_folder=image_sub_folder,
        stack_images=stack_images
    )

    if not isinstance(image_list, list):
        image_list = [image_list]

    return image_list


@singledispatch
def _import_image(image, **kwargs):
    raise NotImplementedError(f"Unsupported image type: {type(image)}")


@_import_image.register(list)
def _(image: list, **kwargs):
    # List can be anything. Hence, we dispatch import_image for the individual list elements.
    image_list = [_import_image(
        image=current_image,
        **kwargs
    ) for current_image in image]

    return image_list


@_import_image.register(str)
def _(image: str, **kwargs):
    # Image is a string, which could be a path to a xml file, to a csv file, or just a regular
    # path a path to a file, or a path to a directory. Test which it is and then dispatch.

    if image.lower().endswith("xml"):
        ...

    elif image.lower().endswith("csv"):
        ...

    elif os.path.isdir(image):
        return _import_image(
            ImageDirectory(directory=image, **kwargs))

    elif os.path.exists(image):
        return _import_image(
            ImageFile(file_path=image, **kwargs).create())

    else:
        raise ValueError("The image path does not point to a xml file, a csv file, a valid image file or a directory "
                         "containing imaging.")


@_import_image.register(pd.DataFrame)
def _(image: pd.DataFrame,
      image_modality: Union[None, str] = None,
      **kwargs):
    ...


@_import_image.register(np.ndarray)
def _(image: np.ndarray,
      sample_name: Union[None, str] = None,
      image_modality: Union[None, str] = None,
      **kwargs):
    ...


@_import_image.register(ImageFile)
def _(image: ImageFile, **kwargs):

    if not issubclass(type(image), ImageFile):
        image = image.create()

    # Check if the data are consistent.
    image.check(raise_error=True)

    # Complete image data and add identifiers (if any)
    image.complete()

    return image


@_import_image.register(ImageDirectory)
def _(image: ImageDirectory, **kwargs):

    # Check first if the data are consistent for a directory.
    image.check(raise_error=True)

    # Yield image files.
    image_list = image.create_images()

    # Dispatch to import_image method for
    return [_import_image(current_image) for current_image in image_list]
