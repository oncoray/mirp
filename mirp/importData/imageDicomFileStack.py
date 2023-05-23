from typing import List

from mirp.importData.imageDicomFile import ImageDicomFile
from mirp.importData.imageGenericFileStack import ImageFileStack


class ImageDicomFileStack(ImageFileStack):

    def __init__(
            self,
            image_file_objects: List[ImageDicomFile],
            **kwargs
    ):
        super().__init__(image_file_objects, **kwargs)

    def complete(self, remove_metadata=True, force=False):
        # TODO: Order files by image origin.
        ...

