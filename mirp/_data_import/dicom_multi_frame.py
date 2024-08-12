import os.path
import numpy as np

from mirp._data_import.generic_file import ImageFile
from mirp._data_import.dicom_file import ImageDicomFile
from mirp._data_import.utilities import get_pydicom_meta_tag


class ImageDicomMultiFrame(ImageDicomFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create(self):
        # This method is called from ImageDicomFile.create amd dispatches to modality-specific multi-frame objects.
        return self

    def is_stackable(self, stack_images: str):
        # Multi-frame images might be actually be stackable (concatenated), but ignore that for now.
        return False

    def _complete_image_origin(self, force=False, frame_id=None):
        if self.image_origin is None:

            # Load relevant metadata.
            self.load_metadata(limited=True)

            if frame_id is None:
                frame_id = 0

            origin = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0020, 0x0032),
                tag_type="mult_float",
                macro_dcm_seq=(0x0020, 0x9113),
                frame_id=frame_id
            )[::-1]
            self.image_origin = tuple(origin)

    def _complete_image_orientation(self, force=False, frame_id=None):
        if self.image_orientation is None:

            # Load relevant metadata.
            self.load_metadata(limited=True)

            if frame_id is None:
                frame_id = 0

            orientation: list[float] = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0020, 0x0037),
                tag_type="mult_float",
                macro_dcm_seq=(0x0020, 0x9116),
                frame_id=frame_id
            )

            # First compute z-orientation.
            # noinspection PyUnreachableCode
            orientation += list(np.cross(orientation[0:3], orientation[3:6]))
            self.image_orientation = np.reshape(orientation[::-1], [3, 3], order="F")

    def _complete_image_spacing(self, force=False, frame_id=None):
        if self.image_spacing is None:
            # Load relevant metadata.
            self.load_metadata(limited=True)

            if frame_id is None:
                frame_id = 0

            # Get pixel-spacing.
            spacing = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0028, 0x0030),
                tag_type="mult_float",
                macro_dcm_seq=(0x0028, 0x9110),
                frame_id=frame_id
            )

            # First try to get spacing between slices.
            z_spacing = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0018, 0x0088),
                tag_type="float",
                macro_dcm_seq=(0x0028, 0x9110),
                frame_id=frame_id
            )

            # If spacing between slices is not set, get slice thickness.
            if z_spacing is None:
                z_spacing = get_pydicom_meta_tag(
                    dcm_seq=self.image_metadata,
                    tag=(0x0018, 0x0050),
                    tag_type="float",
                    macro_dcm_seq=(0x0028, 0x9110),
                    frame_id=frame_id
                )

            # If slice thickness is not set, use a default value.
            if z_spacing is None:
                z_spacing = 1.0
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


class ImageDicomMultiFrameSingle(ImageFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)