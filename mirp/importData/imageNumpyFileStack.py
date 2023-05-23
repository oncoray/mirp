from typing import List

from mirp.importData.imageNumpyFile import ImageNumpyFile
from mirp.importData.imageGenericFileStack import ImageFileStack


class ImageNumpyFileStack(ImageFileStack):

    def __init__(
            self,
            image_file_objects: List[ImageNumpyFile],
            **kwargs
    ):
        super().__init__(image_file_objects, **kwargs)

    def complete(self, remove_metadata=True, force=False):
        # TODO: Order files by file name.
        ...
