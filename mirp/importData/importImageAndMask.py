from typing import Union, List, Set

from mirp.importData.importImage import import_image
from mirp.importData.importMask import import_mask
from mirp.importData.imageGenericFile import ImageFile, MaskFile
from mirp.importData.imageDicomFile import ImageDicomFile, MaskDicomFile

def import_image_and_mask(
        image,
        mask,
        sample_name: Union[None, str, List[str]] = None,
        image_name: Union[None, str, List[str]] = None,
        image_file_type: Union[None, str] = None,
        image_modality: Union[None, str, List[str]] = None,
        image_sub_folder: Union[None, str] = None,
        mask_name: Union[None, str, List[str]] = None,
        mask_file_type: Union[None, str] = None,
        mask_modality: Union[None, str, List[str]] = None,
        mask_sub_folder: Union[None, str] = None,
        association_strategy: Union[None, str, List[str]] = None,
        stack_masks: str = "auto",
        stack_images: str = "auto"):

    # Generate list of images.
    image_list = import_image(
        image,
        sample_name=sample_name,
        image_name=image_name,
        image_file_type=image_file_type,
        image_modality=image_modality,
        image_sub_folder=image_sub_folder,
        stack_images=stack_images
    )

    # Generate list of images.
    mask_list = import_mask(
        mask,
        sample_name=sample_name,
        mask_name=mask_name,
        mask_file_type=mask_file_type,
        mask_modality=mask_modality,
        mask_sub_folder=mask_sub_folder,
        stack_masks=stack_masks
    )

    # Determine association strategy, if this is unset.
    possible_association_strategy = set_association_strategy(image_list=image_list, mask_list=mask_list)
    if association_strategy is None:
        association_strategy = possible_association_strategy

    # Test association strategy.
    ...

    # Associate images with mask objects.
    # This is done using the following steps:
    # 1. Associate based on frame of reference identifiers.
    # 2. Associate based on sample name.
    # 3. Associate based on file distance.
    # 4. Associate based on order (only if lists have the same length).
    associated_masks = associate_masks_with_images(
        image_list=image_list,
        mask_list=mask_list
    )

    # Assign sample names to images and associated masks.
    ...


def set_association_strategy(
        image_list: List[ImageFile, ImageDicomFile],
        mask_list: List[MaskFile, MaskDicomFile]
) -> Set[str]:
    # Association strategy is set by a process of elimination.
    possible_strategies = {
        "frame_of_reference", "sample_name", "file_distance", "file_name_similarity",  "list_order", "position"
    }

    # Check if set is ava
    if len(mask_list) == 0 or len(image_list) == 0:
        return set([])

    # Check if association by list order is possible.
    if len(image_list) != len(mask_list):
        possible_strategies.remove(element="list_order")

    # Check if association by frame of reference UID is possible.
    if any(isinstance(image, ImageDicomFile) for image in image_list) and \
            any(isinstance(mask, MaskDicomFile) for mask in mask_list):
        dcm_image_list: List[ImageDicomFile] = [image for image in image_list if isinstance(image, ImageDicomFile)]
        dcm_mask_list: List[MaskDicomFile] = [mask for mask in mask_list if isinstance(mask, MaskDicomFile)]

        # If frame of reference UIDs are completely absent.
        if all(image.frame_of_reference_uid is None for image in dcm_image_list) or \
                all(mask.frame_of_reference_uid is None for mask in dcm_mask_list):
            possible_strategies.remove(element="frame_of_reference")

    else:
        possible_strategies.remove(element="frame_of_reference")

    # Check if association by sample name is possible.
    if all(image.sample_name is None for image in image_list) or all(mask.sample_name is None for mask in mask_list):
        possible_strategies.remove(element="sample_name")

    # Check if file_distance is possible. If directory are absent or singular, file distance cannot be used for
    # association.
    image_dir_path = set(image.dir_path for image in image_list) - {None}
    mask_dir_path = set(mask.dir_path for mask in mask_list) - {None}
    if len(image_dir_path) <= 1 or len(mask_dir_path) <= 1:
        possible_strategies.remove(element="file_distance")

    # Check if file_name_similarity is possible. If file names are absent, this is not possible.
    if all(image.file_name is None for image in image_list) and all(mask.file_name is None for mask in mask_list):
        possible_strategies.remove(element="file_name_similarity")

    # Check if position can be used.
    ...


def associate_masks_with_images(
        image_list: List[ImageFile],
        mask_list: List[MaskFile]
):
    associated_mask_list = [
        image_file.associate_with_mask(mask_list=mask_list)
        for image_file in image_list
    ]

    if all(mask_files is None for mask_files in associated_mask_list) and len(image_list) == len(mask_list):
        associated_mask_list = [
            [mask_file] for mask_file in mask_list
        ]

    return associated_mask_list
