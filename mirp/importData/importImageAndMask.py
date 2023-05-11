from typing import Union, List

from mirp.importData.importImage import import_image
from mirp.importData.importMask import import_mask


def import_image_and_mask(
        image,
        sample_name: Union[None, str, List[str]] = None,
        image_name=None,
        image_file_type=None,
        image_modality=None,
        image_sub_folder=None,
        mask=None,
        mask_name=None,
        mask_file_type=None,
        mask_sub_folder=None):

    # Generate list of images.
    image_list = import_image(
        image,
        sample_name=sample_name,
        image_name=image_name,
        image_file_type=image_file_type,
        image_modality=image_modality,
        image_sub_folder=image_sub_folder
    )

    mask_list = import_mask(
        mask=mask,
        mask_name=mask_name,
        mask_file_type=mask_file_type,
        mask_sub_folder=mask_sub_folder
    )

    # Stack image and mask together.

    ...
