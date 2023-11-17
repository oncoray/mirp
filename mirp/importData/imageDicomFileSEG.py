from typing import Optional, List

from mirp.importData.imageDicomFile import MaskDicomFile
from mirp.masks.baseMask import BaseMask


class MaskDicomFileSEG(MaskDicomFile):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def is_stackable(self, stack_images: str):
        return False

    def create(self):
        return self

    def _complete_image_origin(self, force=False):
        return

    def _complete_image_orientation(self, force=False):
        return

    def _complete_image_spacing(self, force=False):
        return

    def _complete_image_dimensions(self, force=False):
        return

    def load_data(self, **kwargs):
        ...

    def to_object(self, **kwargs) -> Optional[List[BaseMask]]:
        ...
