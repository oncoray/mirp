from mirp._data_import.dicom_file_ct import ImageDicomFileCT
from mirp._data_import.dicom_multi_frame import ImageDicomMultiFrameStack, ImageDicomMultiFrameIndividual


class ImageDicomFileCTMultiFrameStack(ImageDicomMultiFrameStack, ImageDicomFileCT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_individual_frame_class():
        return ImageDicomFileCTMultiFrameIndividual


class ImageDicomFileCTMultiFrameIndividual(ImageDicomMultiFrameIndividual, ImageDicomFileCTMultiFrameStack):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)