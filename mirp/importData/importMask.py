from typing import Union, List, Dict

from mirp.importData.imageGenericFile import MaskFile
from mirp.importData.importImage import _import_image
from mirp.importData.utilities import supported_mask_modalities, supported_file_types, flatten_list


def import_mask(
        mask,
        sample_name: Union[None, str, List[str]] = None,
        mask_name: Union[None, str, List[str]] = None,
        mask_file_type: Union[None, str] = None,
        mask_modality: Union[None, str, List[str]] = None,
        mask_sub_folder: Union[None, str] = None,
        roi_name: Union[None, str, List[str], Dict[str, str]] = None,
        stack_masks: str = "auto"
) -> List[MaskFile]:
    """
    Creates and curates references to mask files. Masks determine the location of regions of interest. They are
    usually created by manual, semi-automatic or fully automatic segmentation.

    :param mask: A path to a mask file, a path to a directory containing mask files, a path to a config_data.xml
    file, a path to a csv file containing references to mask files, a pandas.DataFrame containing references to
    mask files, or a numpy.ndarray.
    :param sample_name: Name of expected sample names. This is used to select specific mask files. If None,
    no mask files are filtered based on the corresponding sample name (if known).
    :param mask_name: Pattern to match mask files against. The matches are exact. Use wildcard symbols ("*") to
    match varying structures. The sample name (if part of the file name) can also be specified using "#". For example,
    mask_name = '#_*_mask' would find John_Doe in John_Doe_CT_mask.nii or John_Doe_001_mask.nii. File extensions
    do not need to specified. If None, file names are not used for filtering files and setting sample names.
    :param mask_file_type: The type of file that is expected. If None, the file type is not used for filtering
    files. Options: "dicom", "nifti", "nrrd", "numpy" and "itk". "itk" comprises "nifti" and "nrrd" file types.
    :param mask_modality: The type of modality that is expected. If None, modality is not used for filtering files.
    Note that only DICOM files contain metadata concerning modality. Options: "rtstruct", "seg" or "generic_mask".
    :param mask_sub_folder: Fixed directory substructure where mask files are located. If None, this directory
    substructure is not used for filtering files.
    :param roi_name: Name of the regions of interest that should be assessed.
    :param stack_masks: One of auto, yes or no. If mask files in the same directory cannot be assigned to
    different samples, and are 2D (slices) of the same size, they might belong to the same 3D mask stack. "auto"
    will stack 2D numpy arrays, but not other file types. "yes" will stack all files that contain 2D images,
    that have the same dimensions, orientation and spacing, except for DICOM files. "no" will not stack any files.
    DICOM files ignore this argument, because their stacking can be determined from metadata.
    :return: list of mask files.
    """
    # Check modality.
    if mask_modality is not None:
        if not isinstance(mask_modality, str):
            raise TypeError(
                f"The mask_modality argument is expected to be a single character string or None. The following "
                f"modalities are supported: {', '.join(supported_mask_modalities(None))}.")
        _ = supported_mask_modalities(mask_modality.lower())

    # Check image_file_type.
    if mask_file_type is not None:
        if not isinstance(mask_file_type, str):
            raise TypeError(
                f"The mask_file_type argument is expected to be a single character string, or None. The following file "
                f"types are supported: {', '.join(supported_file_types(None))}.")
        _ = supported_file_types(mask_file_type)

    # Check stack_images
    if stack_masks not in ["yes", "auto", "no"]:
        raise ValueError(
            f"The stack_images argument is expected to be one of yes, auto, or no. Found: {stack_masks}."
        )

    mask_list = _import_image(
        mask,
        sample_name=sample_name,
        image_name=mask_name,
        image_file_type=mask_file_type,
        image_modality=mask_modality,
        image_sub_folder=mask_sub_folder,
        stack_images=stack_masks,
        roi_name=roi_name,
        is_mask=True
    )

    if not isinstance(mask_list, list):
        mask_list = [mask_list]

    # Flatten list.
    mask_list = flatten_list(mask_list)

    return mask_list
