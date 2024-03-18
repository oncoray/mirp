from mirp._data_import.import_image import import_image
from mirp._data_import.import_mask import import_mask
from mirp._data_import.generic_file import ImageFile, MaskFile
from mirp._data_import.dicom_file import ImageDicomFile, MaskDicomFile
from mirp._data_import.dicom_file_stack import ImageDicomFileStack
from mirp._utilities.utilities import random_string


def import_image_and_mask(
        image,
        mask=None,
        sample_name: None | str | list[str] = None,
        image_name: None | str | list[str] = None,
        image_file_type: None | str = None,
        image_modality: None | str | list[str] = None,
        image_sub_folder: None | str = None,
        mask_name: None | str | list[str] = None,
        mask_file_type: None | str = None,
        mask_modality: None | str | list[str] = None,
        mask_sub_folder: None | str = None,
        roi_name: None | str | list[str] | dict[str | str] = None,
        association_strategy: None | str | list[str] = None,
        stack_images: str = "auto",
        stack_masks: str = "auto"
) -> list[ImageFile]:
    """
    Creates and curates references to image and mask files. This function is usually called internally by other
    functions such as :func:`~mirp.extractFeaturesAndImages.extract_features`.

    Parameters
    ----------
    image: Any
        A path to an image file, a path to a directory containing image files, a path to a config_data.xml
        file, a path to a csv file containing references to image files, a pandas.DataFrame containing references to
        image files, or a numpy.ndarray.

    mask: Any
        A path to a mask file, a path to a directory containing mask files, a path to a config_data.xml
        file, a path to a csv file containing references to mask files, a pandas.DataFrame containing references to
        mask files, or a numpy.ndarray.

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

    mask_name: str or list of str, optional, default: None
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

    roi_name: str, optional, default: None
        Name of the regions of interest that should be assessed.

    association_strategy: {"frame_of_reference", "sample_name", "file_distance", "file_name_similarity",  "list_order", "position", "single_image"}
        The preferred strategy for associating _images and _masks. File association is preferably done using frame of
        reference UIDs (DICOM), or sample name (NIfTI, numpy). Other options are relatively frail, except for
        `list_order` which may be applicable when a list with _images and a list with _masks is provided and both lists
        are of equal length.

    stack_images: {"auto", "yes", "no"}, optional, default: "str"
        If image files in the same directory cannot be assigned to different samples, and are 2D (slices) of the same
        size, they might belong to the same 3D image stack. "auto" will stack 2D numpy arrays, but not other file types.
        "yes" will stack all files that contain 2D _images, that have the same dimensions, orientation and spacing,
        except for DICOM files. "no" will not stack any files. DICOM files ignore this argument, because their stacking
        can be determined from metadata.

    stack_masks: {"auto", "yes", "no"}, optional, default: "str"
        If mask files in the same directory cannot be assigned to different samples, and are 2D (slices) of the same
        size, they might belong to the same 3D mask stack. "auto" will stack 2D numpy arrays, but not other file
        types. "yes" will stack all files that contain 2D _images, that have the same dimensions, orientation and
        spacing, except for DICOM files. "no" will not stack any files. DICOM files ignore this argument,
        because their stacking can be determined from metadata.

    Returns
    -------
    list[ImageFile]
        The functions returns a list of ImageFile objects, if any were found with the specified filters.
    """
    if mask is None:
        mask = image

    # Generate list of _images.
    image_list = import_image(
        image,
        sample_name=sample_name,
        image_name=image_name,
        image_file_type=image_file_type,
        image_modality=image_modality,
        image_sub_folder=image_sub_folder,
        stack_images=stack_images
    )

    # Generate list of _images.
    mask_list = import_mask(
        mask,
        sample_name=sample_name,
        mask_name=mask_name,
        mask_file_type=mask_file_type,
        mask_modality=mask_modality,
        mask_sub_folder=mask_sub_folder,
        stack_masks=stack_masks,
        roi_name=roi_name
    )

    if len(image_list) == 0:
        raise ValueError(f"No _images were found. Possible reasons are lack of _images with the preferred modality.")
    if len(mask_list) == 0:
        raise ValueError(f"No _masks were found. Possible reasons are lack of _masks with the preferred modality.")

    # Determine association strategy, if this is unset.
    possible_association_strategy = set_association_strategy(
        image_list=image_list,
        mask_list=mask_list
    )

    if association_strategy is None:
        association_strategy = possible_association_strategy
    elif isinstance(association_strategy, str):
        association_strategy = [association_strategy]

    if not isinstance(association_strategy, set):
        association_strategy = set(association_strategy)

    # Test association strategy.
    unavailable_strategy = association_strategy - possible_association_strategy
    if len(unavailable_strategy) > 0:
        raise ValueError(
            f"One or more strategies for associating _images and _masks are not available for the provided image and "
            f"mask set: {', '.join(list(unavailable_strategy))}. Only the following strategies are available: "
            f"{'. '.join(list(possible_association_strategy))}"
        )

    if len(possible_association_strategy) == 0:
        raise ValueError(
            f"No strategies for associating _images and _masks are available, indicating that there is no clear way to "
            f"establish an association."
        )

    # Start association.
    if association_strategy == {"list_order"}:
        # If only the list_order strategy is available, use this.
        for ii, image in enumerate(image_list):
            image.associated_masks = [mask_list[ii]]

    elif association_strategy == {"single_image"}:
        # If single_image is the only strategy, use this.
        image_list[0].associated_masks = mask_list

    else:
        for ii, image in enumerate(image_list):
            image.associate_with_mask(
                mask_list=mask_list,
                association_strategy=association_strategy
            )

        if all(image.associated_masks is None for image in image_list):
            if "single_image" in association_strategy:
                image_list[0].associated_masks = mask_list
            elif "list_order" in association_strategy:
                for ii, image in enumerate(image_list):
                    image.associated_masks = [mask_list[ii]]

    # Ensure that we are working with deep copies from this point - we don't want to propagate changes to _masks,
    # _images by reference.
    image_list = [image.copy() for image in image_list]

    # Set sample names. First we check if all sample names are missing.
    if all(image.sample_name is None for image in image_list):
        if isinstance(sample_name, str):
            sample_name = [sample_name]

        if isinstance(sample_name, list) and len(sample_name) == len(image_list):
            for ii, image in enumerate(image_list):
                image.set_sample_name(sample_name=sample_name[ii])
                if image.associated_masks is not None:
                    for mask in image.associated_masks:
                        mask.set_sample_name(sample_name=sample_name[ii])

        elif all(image.file_name is not None for image in image_list):
            for image in image_list:
                image.set_sample_name(sample_name=image.file_name)

                if image.associated_masks is not None:
                    for mask in image.associated_masks:
                        mask.set_sample_name(sample_name=image.file_name)

    # Then set any sample names for _images that still miss them.
    if any(image.sample_name is None for image in image_list):
        for ii, image in enumerate(image_list):
            if image.sample_name is None:
                generated_sample_name = str(ii + 1) + "_" + random_string(16)
                image.set_sample_name(sample_name=generated_sample_name)
                if image.associated_masks is not None:
                    for mask in image.associated_masks:
                        mask.set_sample_name(sample_name=generated_sample_name)

    return image_list


def set_association_strategy(
        image_list: list[ImageFile] | list[ImageDicomFile],
        mask_list: list[MaskFile] | list[MaskDicomFile]
) -> set[str]:
    # Association strategy is set by a process of elimination.
    possible_strategies = {
        "frame_of_reference", "sample_name", "file_distance", "file_name_similarity",  "list_order", "position",
        "single_image"
    }

    # Check that _images and _masks are available
    if len(mask_list) == 0 or len(image_list) == 0:
        return set([])

    # Check if association by list order is possible.
    if len(image_list) != len(mask_list):
        possible_strategies.remove("list_order")

    # Check that association with a single image is possible.
    if len(image_list) > 1:
        possible_strategies.remove("single_image")

    # Check if association by frame of reference UID is possible.
    if (any(isinstance(image, ImageDicomFile) or isinstance(image, ImageDicomFileStack) for image in image_list) and
            any(isinstance(mask, MaskDicomFile) for mask in mask_list)):
        dcm_image_list: list[ImageDicomFile | ImageDicomFileStack] = [
            image for image in image_list
            if isinstance(image, ImageDicomFile) or isinstance(image, ImageDicomFileStack)]
        dcm_mask_list: list[MaskDicomFile] = [mask for mask in mask_list if isinstance(mask, MaskDicomFile)]

        # If frame of reference UIDs are completely absent.
        if all(image.frame_of_reference_uid is None for image in dcm_image_list) or \
                all(mask.frame_of_reference_uid is None for mask in dcm_mask_list):
            possible_strategies.remove("frame_of_reference")

    else:
        possible_strategies.remove("frame_of_reference")

    # Check if association by sample name is possible.
    if all(image.sample_name is None for image in image_list) or all(mask.sample_name is None for mask in mask_list):
        possible_strategies.remove("sample_name")

    # Check if file_distance is possible. If directory are absent or singular, file distance cannot be used for
    # association.
    image_dir_path = set(image.dir_path for image in image_list) - {None}
    mask_dir_path = set(mask.dir_path for mask in mask_list) - {None}
    if len(image_dir_path) == 0 or len(mask_dir_path) <= 1:
        possible_strategies.remove("file_distance")

    # Check if file_name_similarity is possible. If file names are absent, this is not possible.
    if all(image.file_name is None for image in image_list) or all(mask.file_name is None for mask in mask_list):
        possible_strategies.remove("file_name_similarity")

    # Check if position can be used.
    if all(image.image_origin is None for image in image_list) or all(mask.image_origin is None for mask in mask_list):
        possible_strategies.remove("position")
    else:
        image_position_data = set([
            image.get_image_origin(as_str=True) + image.get_image_spacing(as_str=True) +
            image.get_image_dimension(as_str=True) + image.get_image_orientation(as_str=True)
            for image in image_list if image.image_origin is not None
        ])
        mask_position_data = set([
            mask.get_image_origin(as_str=True) + mask.get_image_spacing(as_str=True) +
            mask.get_image_dimension(as_str=True) + mask.get_image_orientation(as_str=True)
            for mask in mask_list if mask.image_origin is not None
        ])

        # Check that there are more
        if len(image_position_data) <= 1 or len(mask_position_data) <= 1:
            possible_strategies.remove("position")

    return possible_strategies
