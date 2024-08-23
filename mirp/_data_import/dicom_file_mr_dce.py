from typing import Any

from mirp._data_import.dicom_multi_frame import ImageDicomMultiFrame
from mirp._data_import.dicom_file_mr import ImageDicomFileMR
from mirp._data_import.utilities import get_pydicom_meta_tag


class ImageDicomFileMRDCE(ImageDicomFileMR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def export_metadata(self, self_only=False, **kwargs) -> None | dict[str, Any]:
        if not self_only:
            metadata = super().export_metadata()
        else:
            metadata = {}

        self.load_metadata()

        dcm_meta_data = []

        # Diffusion b-value
        # b_value = get_pydicom_meta_tag(
        #     dcm_seq=self.image_metadata,
        #     tag=(0x0018, 0x9087),
        #     tag_type="float",
        #     macro_dcm_seq=(0x0018, 0x9117)
        # )
        # if b_value is not None:
        #     dcm_meta_data += [("diffusion_b_value", b_value)]

        metadata.update(dict(dcm_meta_data))
        return metadata


class ImageDicomFileMRDCEMultiFrame(ImageDicomMultiFrame, ImageDicomFileMRDCE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)