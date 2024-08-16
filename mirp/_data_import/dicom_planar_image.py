import os.path
import numpy as np

from mirp._data_import.generic_file import ImageFile
from mirp._data_import.dicom_file import ImageDicomFile
from mirp._data_import.utilities import get_pydicom_meta_tag


class ImageDicomPlanarImage(ImageDicomFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_stackable(self, stack_images: str):
        # Multi-frame images might be actually be stackable (concatenated), but ignore that for now.
        return False

    def _complete_image_origin(self, force=False, frame_id=None):
        if self.image_origin is None:
            self.image_origin = tuple([0.0, 0.0, 0.0])

    def _complete_image_orientation(self, force=False, frame_id=None):
        if self.image_orientation is None:
            self.image_orientation = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def _complete_image_spacing(self, force=False, frame_id=None):
        if self.image_spacing is None:
            # Load relevant metadata.
            self.load_metadata(limited=True)

            # Get pixel-spacing.
            spacing = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0028, 0x0030),
                tag_type="mult_float",
                macro_dcm_seq=(0x0028, 0x9110),
                frame_id=frame_id
            )
            # Fall-back option if no spacing is provided.
            if spacing is None:
                spacing = [1.0, 1.0]

            # 2D images don't have a depth. The z-dimension is length 1 by convention.
            spacing += [1.0]

            self.image_spacing = tuple(spacing[::-1])

    def _complete_image_dimensions(self, force=False):
        if self.image_dimension is None:
            # Load relevant metadata.
            self.load_metadata(limited=True)

            dimensions = tuple([
                1,
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0010), tag_type="int"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0011), tag_type="int")
            ])

            self.image_dimension = dimensions

    def load_data(self, **kwargs):
        if self.image_data is not None:
            return self.image_data

        if self.file_path is not None and not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The image file could not be found at the expected location: {self.file_path}. [{self.describe_self()}]"
            )

        if self.file_path is None:
            raise ValueError(f"A path to a file was expected, but not present. [{self.describe_self()}]")

        # Load metadata.
        self.load_metadata(include_image=True)
        image_data = self.image_metadata.pixel_array.astype(np.float32)

        # Rescaling and intercept are not required for x-ray images. However, since the pixel data is 2D, we need to
        # add a dimension.
        self.image_data = np.expand_dims(image_data, axis=0)
