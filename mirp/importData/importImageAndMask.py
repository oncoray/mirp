from typing import Union, List, Dict, Set

from mirp.importData.importImage import import_image
from mirp.importData.importMask import import_mask
from mirp.importData.imageGenericFile import ImageFile, MaskFile
from mirp.importData.imageDicomFile import ImageDicomFile, MaskDicomFile


def import_image_and_mask(
        image,
        mask=None,
        sample_name: Union[None, str, List[str]] = None,
        image_name: Union[None, str, List[str]] = None,
        image_file_type: Union[None, str] = None,
        image_modality: Union[None, str, List[str]] = None,
        image_sub_folder: Union[None, str] = None,
        mask_name: Union[None, str, List[str]] = None,
        mask_file_type: Union[None, str] = None,
        mask_modality: Union[None, str, List[str]] = None,
        mask_sub_folder: Union[None, str] = None,
        roi_name: Union[None, str, List[str], Dict[str, str]] = None,
        association_strategy: Union[None, str, List[str]] = None,
        stack_masks: str = "auto",
        stack_images: str = "auto"
) -> List[ImageFile]:

    if mask is None:
        mask = image

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
        stack_masks=stack_masks,
        roi_name=roi_name
    )

    if len(image_list) == 0:
        raise ValueError(f"No images were present.")

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
            f"One or more strategies for associating images and masks are not available for the provided image and "
            f"mask set: {', '.join(list(unavailable_strategy))}. Only the following strategies are available: "
            f"{'. '.join(list(possible_association_strategy))}"
        )

    if len(possible_association_strategy) == 0:
        raise ValueError(
            f"No strategies for associating images and masks are available, indicating that there is no clear way to "
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

    # Ensure that we are working with deep copies from this point - we don't want to propagate changes to masks,
    # images by reference.
    image_list = [image.copy() for image in image_list]

    return image_list


def set_association_strategy(
        image_list: Union[List[ImageFile], List[ImageDicomFile]],
        mask_list: Union[List[MaskFile], List[MaskDicomFile]]
) -> Set[str]:
    # Association strategy is set by a process of elimination.
    possible_strategies = {
        "frame_of_reference", "sample_name", "file_distance", "file_name_similarity",  "list_order", "position",
        "single_image"
    }

    # Check that images and masks are available
    if len(mask_list) == 0 or len(image_list) == 0:
        return set([])

    # Check if association by list order is possible.
    if len(image_list) != len(mask_list):
        possible_strategies.remove("list_order")

    # Check that association with a single image is possible.
    if len(image_list) > 1:
        possible_strategies.remove("single_image")

    # Check if association by frame of reference UID is possible.
    if any(isinstance(image, ImageDicomFile) for image in image_list) and \
            any(isinstance(mask, MaskDicomFile) for mask in mask_list):
        dcm_image_list: List[ImageDicomFile] = [image for image in image_list if isinstance(image, ImageDicomFile)]
        dcm_mask_list: List[MaskDicomFile] = [mask for mask in mask_list if isinstance(mask, MaskDicomFile)]

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
