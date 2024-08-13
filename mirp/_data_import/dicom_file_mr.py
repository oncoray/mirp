from typing import Any
from mirp._data_import.dicom_file import ImageDicomFile
from mirp._data_import.dicom_multi_frame import ImageDicomMultiFrame
from mirp._data_import.utilities import get_pydicom_meta_tag


class ImageDicomFileMR(ImageDicomFile):
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

        # Receive coil name
        receive_coil = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x1250),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9042)
        )
        if receive_coil is not None:
            dcm_meta_data += [("receive_coil_name", receive_coil)]

        # Receive coil manufacturer
        receive_coil_manufacturer = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9041),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9042)
        )
        if receive_coil_manufacturer is not None:
            dcm_meta_data += [("receive_coil_manufacturer", receive_coil_manufacturer)]

        # Receive coil type
        receive_coil_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9043),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9042)
        )
        if receive_coil_type is not None:
            dcm_meta_data += [("receive_coil_type", receive_coil_type)]

        # Transmit coil name
        transmit_coil = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x1251),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9049)
        )
        if transmit_coil is not None:
            dcm_meta_data += [("transmit_coil_name", transmit_coil)]

        # Transmit coil manufacturer
        transmit_coil_manufacturer = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9050),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9049)
        )
        if transmit_coil_manufacturer is not None:
            dcm_meta_data += [("transmit_coil_manufacturer", transmit_coil_manufacturer)]

        # Transmit coil type
        transmit_coil_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9051),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9049)
        )
        if transmit_coil_type is not None:
            dcm_meta_data += [("transmit_coil_type", transmit_coil_type)]

        # Magnetic field strength
        magnetic_field_strength = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0087),
            tag_type="float"
        )
        if magnetic_field_strength is not None:
            dcm_meta_data += [("magnetic_field_strength", magnetic_field_strength)]

        # Image type
        image_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x0008),
            tag_type="str"
        )
        if image_type is not None:
            dcm_meta_data += [("image_type", image_type)]

        # Scanning sequence
        scanning_sequence = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0020),
            tag_type="str"
        )
        if scanning_sequence is not None:
            dcm_meta_data += [("scanning_sequence", scanning_sequence)]

        # Scanning sequence variant
        scanning_sequence_variant = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0021),
            tag_type="str"
        )
        if scanning_sequence_variant is not None:
            dcm_meta_data += [("scanning_sequence_variant", scanning_sequence_variant)]

        # Sequence name
        scanning_sequence_name = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0024),
            tag_type="str"
        )
        if scanning_sequence_name is not None:
            dcm_meta_data += [("scanning_sequence_name", scanning_sequence_name)]

        # Acquisition contrast
        acquisition_contrast = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x9209),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9226)
        )
        if acquisition_contrast is not None:
            dcm_meta_data += [("acquisition_contrast", acquisition_contrast)]

        # Scan options
        scan_options = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0022),
            tag_type="str"
        )
        if scan_options is not None:
            dcm_meta_data += [("scan_options", scan_options)]

        # Acquisition type
        acquisition_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0023),
            tag_type="str"
        )
        if acquisition_type is not None:
            dcm_meta_data += [("acquisition_type", acquisition_type)]

        # Parallel acquisition technique
        parallel_acquisition_technique = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x9078),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9115)
        )
        if parallel_acquisition_technique is not None:
            dcm_meta_data += [("parallel_acquisition_technique", parallel_acquisition_technique)]

        # Repetition time
        repetition_time = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0080),
            tag_type="float",
            macro_dcm_seq=(0x0018, 0x9112)
        )
        if repetition_time is not None:
            dcm_meta_data += [("repetition_time", repetition_time)]

        # Echo time
        echo_time = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0081),
            tag_type="float",
            macro_dcm_seq=(0x0018, 0x9112)
        )
        if echo_time is not None:
            dcm_meta_data += [("echo_time", echo_time)]

        # Echo train length
        echo_train_length = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0091),
            tag_type="float",
            macro_dcm_seq=(0x0018, 0x9112)
        )
        if echo_train_length is not None:
            dcm_meta_data += [("echo_train_length", echo_train_length)]

        # Inversion time
        inversion_time = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0082),
            tag_type="float"
        )
        if inversion_time is not None:
            dcm_meta_data += [("inversion_time", inversion_time)]

        # Trigger time
        trigger_time = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x1060),
            tag_type="float"
        )
        if trigger_time is not None:
            dcm_meta_data += [("trigger_time", trigger_time)]

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


class ImageDicomFileMRMultiFrame(ImageDicomMultiFrame, ImageDicomFileMR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
