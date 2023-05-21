from typing import List

from mirp.importData.imageITKFile import ImageITKFile
from mirp.importData.imageGenericFileStack import ImageFileStack


class ImageITKFileStack(ImageFileStack):

    def __init__(
            self,
            image_file_objects: List[ImageITKFile],
            **kwargs
    ):
        super().__init__(image_file_objects, **kwargs)

    def complete(self, remove_metadata=True):
        # TODO: Order files by image origin.
        ...
