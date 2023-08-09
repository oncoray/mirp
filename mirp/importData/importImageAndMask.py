from typing import Union, List

from mirp.importData.importImage import import_image
from mirp.importData.importMask import import_mask
from mirp.importData.imageGenericFile import ImageFile, MaskFile

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
