from typing import Any
from mirp._data_import.dicom_file import ImageDicomFile
from mirp._data_import.utilities import get_pydicom_meta_tag


class ImageDicomFileCT(ImageDicomFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_stackable(self, stack_images: str):
        return True

    def create(self):
        return self

    def load_data(self, **kwargs):
        self.image_data = self.load_data_generic()

    def export_metadata(self, self_only=False, **kwargs) -> None | dict[str, Any]:
        if not self_only:
            metadata = super().export_metadata()
        else:
            metadata = {}

        self.load_metadata()

        dcm_meta_data = []

        # Scanner type
        scanner_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x1090),
            tag_type="str"
        )
        if scanner_type is not None:
            dcm_meta_data += [("scanner_type", scanner_type)]

        # Scanner manufacturer
        manufacturer = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x0070),
            tag_type="str"
        )
        if manufacturer is not None:
            dcm_meta_data += [("manufacturer", manufacturer)]

        # Image type
        image_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x0008),
            tag_type="str"
        )
        if image_type is not None:
            dcm_meta_data += [("image_type", image_type)]

        # Convolution kernel
        kernel = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x1210),
            tag_type="str"
        )
        if kernel is not None:
            dcm_meta_data += [("kernel", kernel)]

        # Peak kilo voltage output
        kvp = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0060),
            tag_type="float"
        )
        if kvp is not None:
            dcm_meta_data += [("kvp", kvp)]

        # Tube current in mA
        tube_current = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x1151),
            tag_type="float"
        )
        if tube_current is not None:
            dcm_meta_data += [("tube_current", tube_current)]

        # Exposure time in milliseconds
        exposure_time = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x1150),
            tag_type="float"
        )
        if exposure_time is not None:
            dcm_meta_data += [("exposure_time", exposure_time)]

        # Radiation exposure in mAs
        exposure = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x1152),
            tag_type="float"
        )
        if exposure is not None:
            dcm_meta_data += [("exposure", exposure_time)]

        # Contrast/bolus agent
        contrast_agent = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0010),
            tag_type="str"
        )
        if contrast_agent is not None:
            dcm_meta_data += [("contrast_agent", contrast_agent)]

        metadata.update(dict(dcm_meta_data))
        return metadata
