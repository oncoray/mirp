import os
import os.path

import numpy as np

from typing import Union, List, Tuple

from mirp.importData.imageGenericFile import ImageFile


class ImageFileStack(ImageFile):
    def is_stackable(self, stack_images: str):
        return False

    def _complete_image_origin(self):
        ...

    def _complete_image_orientation(self):
        ...

    def _complete_image_spacing(self):
        ...

    def _complete_image_dimensions(self):
        ...

    def __init__(
            self,
            image_file_objects: List[ImageFile],
            dir_path: Union[None, str] = None,
            sample_name: Union[None, str] = None,
            image_name: Union[None, str, List[str]] = None,
            image_modality: Union[None, str] = None,
            image_file_type: Union[None, str] = None,
            **kwargs):

        if dir_path is None:
            dir_path = image_file_objects[0].dir_path

        if sample_name is None:
            sample_name = image_file_objects[0].sample_name

        if image_name is None:
            image_name = image_file_objects[0].image_name

        if image_modality is None:
            image_modality = image_file_objects[0].modality

        if image_file_type is None:
            image_file_type = image_file_objects[0].file_type

        # Aspects regarding the image itself are set based on the stack itself.
        super().__init__(
            file_path=None,
            dir_path=dir_path,
            sample_name=sample_name,
            file_name=None,
            image_name=image_name,
            image_modality=image_modality,
            image_file_type=image_file_type,
            image_data=None,
            image_origin=None,
            image_orientation=None,
            image_spacing=None,
            image_dimensions=None
        )

    def create(self):
        # TODO: dispatch to sub-classes based on file-type.
        ...

    def load_metadata(self):
        ...

    def load_data(self):
        ...
