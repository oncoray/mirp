from mirp._data_import.dicom_multi_frame import ImageDicomMultiFrameStack, ImageDicomMultiFrameIndividual
from mirp._data_import.dicom_file_mr import ImageDicomFileMR


class ImageDicomFileMRDCE(ImageDicomFileMR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ImageDicomFileMRDCEMultiFrameStack(ImageDicomMultiFrameStack, ImageDicomFileMRDCE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_individual_frame_class():
        return ImageDicomFileMRDCEMultiFrameIndividual


class ImageDicomFileMRDCEMultiFrameIndividual(ImageDicomMultiFrameIndividual, ImageDicomFileMRDCEMultiFrameStack):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
