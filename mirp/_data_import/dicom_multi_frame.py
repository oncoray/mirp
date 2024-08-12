from mirp._data_import.generic_file import ImageFile
from mirp._data_import.dicom_file import ImageDicomFile
from mirp._data_import.utilities import get_pydicom_meta_tag


class ImageDicomMultiFrame(ImageDicomFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create(self):
        # This method is called from ImageDicomFile.create amd dispatches to modality-specific multi-frame objects.
        ...

    def is_stackable(self, stack_images: str):
        # Multi-frame images might be actually be stackable (concatenated), but ignore that for now.
        return False

    def _complete_image_dimensions(self, force=False):
        if self.image_dimension is None:
            # Load relevant metadata.
            self.load_metadata(limited=True)

            dimensions = tuple([
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0008), tag_type="int"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0010), tag_type="int"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0011), tag_type="int")
            ])

            self.image_dimension = dimensions


class ImageDicomMultiFrameSingle(ImageFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)