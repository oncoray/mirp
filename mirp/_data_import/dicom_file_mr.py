import numpy as np

from typing import Any
from mirp._data_import.dicom_file import ImageDicomFile
from mirp._data_import.utilities import get_pydicom_meta_tag


class ImageDicomFileMR(ImageDicomFile):
    def __init__(
            self,
            file_path: None | str = None,
            dir_path: None | str = None,
            sample_name: None | str | list[str] = None,
            file_name: None | str = None,
            image_name: None | str = None,
            image_modality: None | str = None,
            image_file_type: None | str = None,
            image_data: None | np.ndarray = None,
            image_origin: None | tuple[float, float, float] = None,
            image_orientation: None | np.ndarray = None,
            image_spacing: None | tuple[float, float, float] = None,
            image_dimensions: None | tuple[int] = None,
            **kwargs
    ):

        super().__init__(
            file_path=file_path,
            dir_path=dir_path,
            sample_name=sample_name,
            file_name=file_name,
            image_name=image_name,
            image_modality=image_modality,
            image_file_type=image_file_type,
            image_data=image_data,
            image_origin=image_origin,
            image_orientation=image_orientation,
            image_spacing=image_spacing,
            image_dimensions=image_dimensions
        )

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

        # Repetition time
        repetition_time = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0080),
            tag_type="float"
        )
        if repetition_time is not None:
            dcm_meta_data += [("repetition_time", repetition_time)]

        # Echo time
        echo_time = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0081),
            tag_type="float"
        )
        if echo_time is not None:
            dcm_meta_data += [("echo_time", echo_time)]

        # Echo train length
        echo_train_length = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0091),
            tag_type="float"
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

        metadata.update(dict(dcm_meta_data))
        return metadata
