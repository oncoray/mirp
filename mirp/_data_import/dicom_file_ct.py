from typing import Any
from mirp._data_import.dicom_file import ImageDicomFile
from mirp._data_import.dicom_multi_frame import ImageDicomMultiFrame
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

        # Convolution kernel
        kernel = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x1210),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9314)
        )
        if kernel is not None:
            dcm_meta_data += [("kernel", kernel)]

        # Reconstruction algorithm
        reconstruction_algorithm = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9315),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9314)
        )
        if reconstruction_algorithm is not None:
            dcm_meta_data += [("reconstruction_algorithm", reconstruction_algorithm)]

        # Peak kilo voltage output
        kvp = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0060),
            tag_type="float",
            macro_dcm_seq=(0x0018, 0x9325)
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

        # Contrast/bolus agent phase
        contrast_agent_phase = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9344),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9341)
        )
        if contrast_agent_phase is not None:
            dcm_meta_data += [("contrast_agent_phase", contrast_agent_phase)]

        metadata.update(dict(dcm_meta_data))
        return metadata


class ImageDicomFileCTMultiFrame(ImageDicomMultiFrame, ImageDicomFileCT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
