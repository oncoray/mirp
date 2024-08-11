from mirp._data_import.generic_file import ImageFile
from mirp._data_import.dicom_file import ImageDicomFile


class ImageDicomMultiFrame(ImageDicomFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create(self):
        # This method is called from ImageDicomFile.create amd dispatches to modality-specific multi-frame objects.
        ...

    def is_stackable(self, stack_images: str):
        # Multi-frame images might be actually be stackable (concatenated), but ignore that for now.
        return False

    def _complete_image_origin(self, force=False):
        ...

    def _complete_image_orientation(self, force=False):
        ...

    def _complete_image_spacing(self, force=False):
        ...

    def _complete_image_dimensions(self, force=False):
        ...


class ImageDicomMultiFrameSingle(ImageFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)