from mirp._data_import.generic_file import MaskFile
from mirp.data_import.import_image import _import_image
from mirp._data_import.utilities import supported_mask_modalities, supported_file_types, flatten_list


def import_mask(
        mask,
        sample_name: None | str | list[str] = None,
        mask_name: None | str | list[str] = None,
        mask_file_type: None | str = None,
        mask_modality: None | str | list[str] = None,
        mask_sub_folder: None | str = None,
        roi_name: None | str | list[str] | dict[str, str] = None,
        stack_masks: str = "auto"
) -> list[MaskFile]:
    """
    Creates and curates references to mask files. Masks determine the location of regions of interest. They are
    usually created by manual, semi-automatic or fully automatic segmentation.

    Parameters
    ----------
    mask: Any
        A path to a mask file, a path to a directory containing mask files, a path to a config_data.xml
        file, a path to a csv file containing references to mask files, a pandas.DataFrame containing references to
        mask files, or a numpy.ndarray.

    sample_name: str or list of str, optional, default: None
        Name of expected sample names. This is used to select specific mask files. If None, no mask files are filtered
         based on the corresponding sample name (if known).

    mask_name: str, optional, default: None
        Pattern to match mask files against. The matches are exact. Use wildcard symbols ("*") to match varying
        structures. The sample name (if part of the file name) can also be specified using "#". For example,
        mask_name = '#_*_mask' would find John_Doe in John_Doe_CT_mask.nii or John_Doe_001_mask.nii. File extensions
        do not need to be specified. If None, file names are not used for filtering files and setting sample names.

    mask_file_type: {"dicom", "nifti", "nrrd", "numpy", "itk"}, optional, default: None
        The type of file that is expected. If None, the file type is not used for filtering files.
        "itk" comprises "nifti" and "nrrd" file types.

    mask_modality: {"rtstruct", "seg", "generic_mask"}, optional, default: None
        The type of modality that is expected. If None, modality is not used for filtering files.
        Note that only DICOM files contain metadata concerning modality. Masks from non-DICOM files are considered to
        be "generic_mask".

    mask_sub_folder: str, optional, default: None
        Fixed directory substructure where mask files are located. If None, the directory substructure is not used for
        filtering files.

    roi_name: str or list of str or dict, optional, default: None
        Name of the regions of interest that should be assessed.

    stack_masks: {"auto", "yes", "no"}, optional, default: "str"
        If mask files in the same directory cannot be assigned to different samples, and are 2D (slices) of the same
        size, they might belong to the same 3D mask stack. "auto" will stack 2D numpy arrays, but not other file
        types. "yes" will stack all files that contain 2D images, that have the same dimensions, orientation and
        spacing, except for DICOM files. "no" will not stack any files. DICOM files ignore this argument,
        because their stacking can be determined from metadata.

    Returns
    -------
    list of MaskFile
        The functions returns a list of MaskFile objects, if any were found with the specified filters.

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
