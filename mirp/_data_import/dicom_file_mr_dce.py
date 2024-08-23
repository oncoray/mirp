from typing import Any

from mirp._data_import.dicom_multi_frame import ImageDicomMultiFrame
from mirp._data_import.dicom_file_mr import ImageDicomFileMR
from mirp._data_import.utilities import get_pydicom_meta_tag


class ImageDicomFileMRDCE(ImageDicomFileMR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ImageDicomFileMRDCEMultiFrame(ImageDicomMultiFrame, ImageDicomFileMRDCE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)