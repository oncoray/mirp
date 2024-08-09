from mirp._data_import.generic_file import ImageFile
from mirp._data_import.dicom_file import ImageDicomFile


class ImageDicomMultiFrame(ImageDicomFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_stackable(self, stack_images: str):
        return False

    def _complete_image_origin(self, force=False):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_sample_origin. Please specify "
            f"implementation for subclasses."
        )

    def _complete_image_orientation(self, force=False):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_sample_orientation. Please specify "
            f"implementation for subclasses."
        )

    def _complete_image_spacing(self, force=False):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_sample_spacing. Please specify "
            f"implementation for subclasses."
        )

    def _complete_image_dimensions(self, force=False):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_sample_dimensions. Please specify "
            f"implementation for subclasses."
        )


class ImageDicomMultiFrameSingle(ImageFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)