from mirp._data_import.dicom_file_mr import ImageDicomFileMR
from mirp._data_import.dicom_multi_frame import ImageDicomMultiFrameStack, ImageDicomMultiFrameIndividual


class ImageDicomFileMRMultiFrameStack(ImageDicomMultiFrameStack, ImageDicomFileMR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_individual_frame_class():
        return ImageDicomFileMRMultiFrameIndividual


class ImageDicomFileMRMultiFrameIndividual(ImageDicomMultiFrameIndividual, ImageDicomFileMRMultiFrameStack):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
