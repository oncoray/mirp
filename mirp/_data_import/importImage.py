import os
import os.path
from functools import singledispatch
import numpy as np
import pandas as pd

from mirp._data_import.imageDirectory import ImageDirectory, MaskDirectory
from mirp._data_import.imageGenericFile import ImageFile, MaskFile
from mirp._data_import.utilities import supported_file_types, supported_image_modalities, flatten_list
from mirp.settings.importDataSettings import import_data_settings


def import_image(
        image,
        sample_name: None | str | list[str] = None,
        image_name: None | str | list[str] = None,
        image_file_type: None | str = None,
        image_modality: None | str | list[str] = None,
        image_sub_folder: None | str = None,
        stack_images: str = "auto"
) -> list[ImageFile]:
    """
    Creates and curates references to image files.

    Parameters
    ----------
    image: Any
        A path to an image file, a path to a directory containing image files, a path to a config_data.xml
        file, a path to a csv file containing references to image files, a pandas.DataFrame containing references to
        image files, or a numpy.ndarray.

    sample_name: str or list of str, default: None
        Name of expected sample names. This is used to select specific image files. If None, no image files are
        filtered based on the corresponding sample name (if known).

    image_name: str, optional, default: None
        Pattern to match image files against. The matches are exact. Use wildcard symbols ("*") to
        match varying structures. The sample name (if part of the file name) can also be specified using "#". For
        example, image_name = '#_*_image' would find John_Doe in John_Doe_CT_image.nii or John_Doe_001_image.nii.
        File extensions do not need to be specified. If None, file names are not used for filtering files and
        setting sample names.

    image_file_type: {"dicom", "nifti", "nrrd", "numpy", "itk"}, optional, default: None
        The type of file that is expected. If None, the file type is not used for filtering files.
        "itk" comprises "nifti" and "nrrd" file types.

    image_modality: {"ct", "pet", "pt", "mri", "mr", "rtdose", "generic"}, optional, default: None
        The type of modality that is expected. If None, modality is not used for filtering files. Note that only
        DICOM files contain metadata concerning modality.

    image_sub_folder: str, optional, default: None
        Fixed directory substructure where image files are located. If None, the directory substructure is not used
        for filtering files.

    stack_images: {"auto", "yes", "no"}, optional, default: "str"
        If image files in the same directory cannot be assigned to different samples, and are 2D (slices) of the same
        size, they might belong to the same 3D image stack. "auto" will stack 2D numpy arrays, but not other file types.
        "yes" will stack all files that contain 2D _images, that have the same dimensions, orientation and spacing,
        except for DICOM files. "no" will not stack any files. DICOM files ignore this argument, because their stacking
        can be determined from metadata.

    Returns
    -------
    list of ImageFile
        The functions returns a list of ImageFile objects, if any were found with the specified filters.
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
                f"The image_file_type argument is expected to be a single character string, or None. The following "
                f" file types are supported: {', '.join(supported_file_types(None))}.")
        _ = supported_file_types(image_file_type)

    # Check stack_images
    if stack_images not in ["yes", "auto", "no"]:
        raise ValueError(
            f"The stack_images argument is expected to be one of yes, auto, or no. Found: {stack_images}."
        )

    image_list: ImageFile | list[ImageFile] = _import_image(
        image,
        sample_name=sample_name,
        image_name=image_name,
        image_file_type=image_file_type,
        image_modality=image_modality,
        image_sub_folder=image_sub_folder,
        stack_images=stack_images,
        is_mask=False
    )

    if not isinstance(image_list, list):
        image_list = [image_list]

    # Flatten list.
    image_list = flatten_list(image_list)

    replacement_image_modality = None
    if isinstance(image_modality, str):
        replacement_image_modality = image_modality
    elif isinstance(image_modality, list) and len(image_modality) == 1:
        replacement_image_modality = image_modality[1]

    for image in image_list:
        image.set_modality(modality=replacement_image_modality)

    return image_list


@singledispatch
def _import_image(image, **kwargs):
    raise NotImplementedError(f"Unsupported image type: {type(image)}")


@_import_image.register(list)
def _(image: list, **kwargs):
    # List can be anything. Hence, we dispatch import_image for the individual list elements.
    image_list = [_import_image(current_image, **kwargs) for current_image in image]

    return image_list


@_import_image.register(str)
def _(
        image: str,
        is_mask: bool = False,
        **kwargs
):
    # Image is a string, which could be a path to a xml file, to a csv file, or just a regular
    # path a path to a file, or a path to a directory. Test which it is and then dispatch.

    if image.lower().endswith("xml"):
        data_arguments = import_data_settings(path=image, is_mask=is_mask)
        image_list = []

        for current_data_arguments in data_arguments:
            image = current_data_arguments["image"]
            current_data_arguments.pop("image")
            image_list += [_import_image(image, is_mask=is_mask, **current_data_arguments)]

        return image_list

    elif image.lower().endswith("csv"):
        ...

    elif os.path.isdir(image):
        if is_mask:
            return _import_image(MaskDirectory(directory=image, **kwargs), remove_metadata=True)
        else:
            return _import_image(ImageDirectory(directory=image, **kwargs), remove_metadata=True)

    elif os.path.exists(image):
        if is_mask:
            return _import_image(MaskFile(file_path=image, **kwargs), remove_metadata=True)
        else:
            return _import_image(ImageFile(file_path=image, **kwargs), remove_metadata=True)

    else:
        raise ValueError("The image path does not point to a xml file, a csv file, a valid image file or a directory "
                         "containing imaging.")


@_import_image.register(pd.DataFrame)
def _(
        image: pd.DataFrame,
        image_modality: None | str = None,
        **kwargs
):
    ...


@_import_image.register(np.ndarray)
def _(
        image: np.ndarray,
        is_mask: bool = False,
        **kwargs
):

    from mirp._data_import.imageNumpyFile import ImageNumpyFile, MaskNumpyFile

    if is_mask:
        image_object = MaskNumpyFile(**kwargs)
    else:
        image_object = ImageNumpyFile(**kwargs)

    image_object.image_data = image_object.image_metadata = image
    image_object.complete()
    image_object.update_image_data()
    image_object.check(raise_error=True, remove_metadata=False)

    return image_object


@_import_image.register(ImageFile)
def _(
        image: ImageFile,
        remove_metadata: bool = False,
        **kwargs
):

    # Create image.
    image = image.create()

    # Check if the data are consistent.
    image.check(raise_error=True)

    # Complete image data and add identifiers (if any)
    image.complete()

    if remove_metadata:
        image.remove_metadata()

    return image


@_import_image.register(ImageDirectory)
def _(
        image: ImageDirectory,
        remove_metadata: bool = False,
        **kwargs
):

    # Check first if the data are consistent for a directory.
    image.check(raise_error=True)

    # Yield image files.
    image.create_images()

    # Dispatch to import_image method for ImageFile objects. This performs a last check and completes the object.
    return [
        _import_image(current_image, remove_metadata=remove_metadata, **kwargs)
        for current_image in image.image_files
    ]
