from typing import Any

from mirp._data_import.dicom_file_mr import ImageDicomFileMR
from mirp._data_import.utilities import get_pydicom_meta_tag


class ImageDicomFileMRADC(ImageDicomFileMR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def export_metadata(self, self_only=False, **kwargs) -> None | dict[str, Any]:
        if not self_only:
            metadata = super().export_metadata()
        else:
            metadata = {}

        self.load_metadata()

        dcm_meta_data = []

        metadata.update(dict(dcm_meta_data))
        return metadata
    