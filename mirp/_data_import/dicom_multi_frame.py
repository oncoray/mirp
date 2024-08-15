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
        from mirp._data_import.dicom_file_ct import ImageDicomFileCTMultiFrame
        from mirp._data_import.dicom_file_pet import ImageDicomFilePTMultiFrame
        from mirp._data_import.dicom_file_mr import ImageDicomFileMRMultiFrame
        from mirp._data_import.dicom_file_mr_adc import ImageDicomFileMRADCMultiFrame

        if self.modality == "ct":
            file_class = ImageDicomFileCTMultiFrame
        elif self.modality == "pt":
            file_class = ImageDicomFilePTMultiFrame
        elif self.modality == "mr":
            file_class = ImageDicomFileMRMultiFrame
        elif self.modality == "adc":
            file_class = ImageDicomFileMRADCMultiFrame

        else:
            # Multi-frame is not implemented for the following modalities:
            # - RT Dose: lack of DICOM module for RT Dose with multi-frame data.
            raise NotImplementedError(
                f"Multi-frame DICOM not implemented for {self.modality} modality. Contact the devs."
            )

        image = file_class()
        image.update_from_template(template=self)

        return image

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

        # Determine number of frames
        n_frames = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0008), tag_type="int")

        # Determine rescale intercept.
        rescale_intercept = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0028, 0x1052),
            tag_type="float",
            macro_dcm_seq=(0x0028, 0x9145)
        )
        if rescale_intercept is None:
            if n_frames is None:
                rescale_intercept = 0.0
            else:
                rescale_intercept = [
                    get_pydicom_meta_tag(
                        dcm_seq=self.image_metadata,
                        tag=(0x0028, 0x1052),
                        tag_type="float",
                        macro_dcm_seq=(0x0028, 0x9145),
                        frame_id=frame_id,
                        default=0.0
                    )
                    for frame_id in np.arange(self.image_dimension[0])
                ]

        # Determine rescale slope.
        rescale_slope = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0028, 0x1053),
            tag_type="float",
            macro_dcm_seq=(0x0028, 0x9145)
        )
        if rescale_slope is None:
            if n_frames is None:
                rescale_slope = 1.0
            else:
                rescale_slope = [
                    get_pydicom_meta_tag(
                        dcm_seq=self.image_metadata,
                        tag=(0x0028, 0x1052),
                        tag_type="float",
                        macro_dcm_seq=(0x0028, 0x9145),
                        frame_id=frame_id,
                        default=1.0
                    )
                    for frame_id in np.arange(self.image_dimension[0])
                ]

        # Apply slope and intercept.
        if isinstance(rescale_slope, list):
            for ii, b in enumerate(rescale_slope):
                image_data[ii, :, :] = b * image_data[ii, :, :]
        else:
            image_data *= rescale_slope

        if isinstance(rescale_intercept, list):
            for ii, a in enumerate(rescale_intercept):
                image_data[ii, :, :] = image_data[ii, :, :] + a
        else:
            image_data += rescale_intercept

        self.image_data = image_data
