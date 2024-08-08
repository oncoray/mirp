import warnings
import os

import numpy as np

from typing import Any
from mirp._data_import.dicom_file import ImageDicomFile
from mirp._data_import.utilities import get_pydicom_meta_tag


class ImageDicomFileRTDose(ImageDicomFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_stackable(self, stack_images: str):
        return False

    def create(self):
        return self

    def _complete_image_orientation(self, force=False):
        if self.image_orientation is None:
            # Load relevant metadata.
            self.load_metadata(limited=True)

            # This is orientation for x and y directions.
            orientation: list[float] = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0020, 0x0037),
                tag_type="mult_float"
            )

            # First compute z-orientation.
            orientation += list(np.cross(orientation[0:3], orientation[3:6]))
            self.image_orientation = np.reshape(orientation[::-1], [3, 3], order="F")

    def _complete_image_spacing(self, force=False):
        if self.image_spacing is None:

            # Load relevant metadata.
            self.load_metadata(limited=True)

            # Get pixel-spacing.
            spacing = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0030), tag_type="mult_float")

            # Get Grid Frame Offset Vector
            z_spacing = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x3004, 0x000C),
                tag_type="mult_float"
            )

            if len(z_spacing) > 1:
                # DICOM file contains multiple frames.
                z_spacing = np.unique(np.diff(z_spacing))
                if len(z_spacing) > 1:
                    raise ValueError(
                        f"Spacing of radiation dose grid is inconsistent: {', '.join(z_spacing)}. "
                        f"[{self.describe_self()}]"
                    )
                z_spacing = z_spacing[0]
            else:
                # DICOM file contains only a single frame. Use a default 1.0 mm value.
                z_spacing = 1.0
                warnings.warn(
                    f"Radiation dose grid only contains a single frame (slice). A default frame spacing of 1.0 mm is "
                    f"assumed. Within-plane spacing is not affected. [{self.describe_self()}]"
                )

            if z_spacing < 0.0:
                self.image_orientation = np.matmul(
                    self.image_orientation, [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                )
                z_spacing = np.abs(z_spacing)

            spacing += [z_spacing]

            self.image_spacing = tuple(spacing[::-1])

    def _complete_image_dimensions(self, force=False):
        if self.image_dimension is None:

            # Load relevant metadata.
            self.load_metadata(limited=True)

            dimensions = tuple([
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0008), tag_type="int"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0010), tag_type="int"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0011), tag_type="int")
            ])

            self.image_dimension = dimensions

    def load_data(self, **kwargs):
        if self.image_data is not None:
            return self.image_data

        if self.file_path is not None and not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The image file could not be found at the expected location: {self.file_path} [{self.describe_self()}]"
            )

        if self.file_path is None:
            raise ValueError(f"A path to a file was expected, but not present. [{self.describe_self()}]")

        # Load metadata.
        self.load_metadata(include_image=True)
        image_data = self.image_metadata.pixel_array.astype(np.float32)

        # Update data with dose grid scaling
        dose_grid_scaling = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x3004, 0x000E),
            tag_type="float",
            default=1.0
        )
        image_data = image_data * dose_grid_scaling

        self.image_data = image_data

    def export_metadata(self, self_only: bool = False, **kwargs) -> None | dict[str, Any]:
        if not self_only:
            metadata = super().export_metadata()
        else:
            metadata = {}

        self.load_metadata()

        dcm_meta_data = []

        # Manufacturer
        manufacturer = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x0070),
            tag_type="str"
        )
        if manufacturer is not None:
            dcm_meta_data += [("manufacturer", manufacturer)]

        # Dose Units
        dose_units = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x3004, 0x0002),
            tag_type="str"
        )
        if dose_units is not None:
            dcm_meta_data += [("dose_units", dose_units)]

        # Dose Type
        dose_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x3004, 0x0004),
            tag_type="str"
        )
        if dose_type is not None:
            dcm_meta_data += [("dose_type", dose_type)]

        # Dose comment
        dose_comment = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x3004, 0x0006),
            tag_type="str"
        )
        if dose_comment is not None:
            dcm_meta_data += [("dose_comment", dose_comment)]

        # Dose summation type
        dose_summation_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x3004, 0x000a),
            tag_type="str"
        )
        if dose_summation_type is not None:
            dcm_meta_data += [("dose_summation_type", dose_summation_type)]

        metadata.update(dict(dcm_meta_data))
        return metadata
