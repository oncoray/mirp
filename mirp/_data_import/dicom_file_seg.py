import warnings

import numpy as np
from pydicom import dcmread

from mirp._data_import.dicom_file import MaskDicomFile
from mirp._data_import.utilities import get_pydicom_meta_tag, has_pydicom_meta_tag
from mirp._masks.base_mask import BaseMask


class MaskDicomFileSEG(MaskDicomFile):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_stackable(self, stack_images: str):
        return False

    def create(self):
        return self

    def _complete_image_origin(self, force=False, frame_id=None):
        return

    def _complete_image_orientation(self, force=False, frame_id=None):
        return

    def _complete_image_spacing(self, force=False, frame_id=None):
        return

    def _complete_image_dimensions(self, force=False):
        return

    def check_mask(self, raise_error=True):
        if self.image_metadata is None:
            raise TypeError("DEV: the image_meta_data attribute has not been set.")

        if not get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0062, 0x0002), test_tag=True):
            if raise_error:
                warnings.warn(
                    f"The current SEG set did not contain any segment sequences. [{self.describe_self()}]",
                )
            return False

        roi_name_present = [
            get_pydicom_meta_tag(
                dcm_seq=current_segment_sequence, tag=(0x0062, 0x0005), tag_type="str", default=None)
            for current_segment_sequence in self.image_metadata[(0x0062, 0x0002)]
        ]

        if len(roi_name_present) == 0 and raise_error:
            warnings.warn(
                f"The current SEG set does not contain any segmentations. [{self.describe_self()}]"
            )
            return False

        if any(x is None for x in roi_name_present) and raise_error:
            warnings.warn(
                f"The current SEG set does not contain any labels. [{self.describe_self()}]"
            )
            return False

        return True

    def load_data(self, **kwargs):
        pass

    def to_object(self, **kwargs) -> None | list[BaseMask]:

        self.load_metadata()
        self.set_object_metadata()
        if not self.check_mask():
            return None

        mask_list = []

        # Find which roi numbers (3006,0022) are associated with which roi names (3004,0024).
        roi_name_present = [
            get_pydicom_meta_tag(
                dcm_seq=current_segment_sequence, tag=(0x0062, 0x0005), tag_type="str", default=None)
            for current_segment_sequence in self.image_metadata[(0x0062, 0x0002)]
        ]
        roi_number_present = [
            get_pydicom_meta_tag(
                dcm_seq=current_segment_sequence, tag=(0x0062, 0x0004), tag_type="int", default=None)
            for current_segment_sequence in self.image_metadata[(0x0062, 0x0002)]
        ]

        # Identify user-provided roi names.
        provided_roi_names = None
        if isinstance(self.roi_name, str):
            provided_roi_names = [self.roi_name]
        elif isinstance(self.roi_name, list):
            provided_roi_names = self.roi_name
        elif isinstance(self.roi_name, dict):
            provided_roi_names = list(self.roi_name.keys())

        if provided_roi_names is not None:
            roi_number_present = [
                x for ii, x in enumerate(roi_number_present) if roi_name_present[ii] in provided_roi_names
            ]
            roi_name_present = [x for x in roi_name_present if x in provided_roi_names]

        if len(roi_number_present) == 0:
            return None

        # Determine which frames corresponds to which roi numbers.
        frame_list = []
        for roi_number in roi_number_present:
            frame_list += [list(self._generate_frame_index(roi_index=roi_number))]

        # Read mask data.
        dcm = dcmread(self.file_path, stop_before_pixels=False, force=True)
        seg_mask_data = dcm.pixel_array

        for ii, roi_number in enumerate(roi_number_present):
            frames = frame_list[ii]

            # Acquire mask spacing
            mask_spacing = self._get_mask_spacing(frames=frames)
            for frame in frames:
                frame_spacing = self._get_mask_spacing(frames=[frame])
                if frame_spacing != mask_spacing:
                    raise ValueError(
                        f"The DICOM SEG file has inconsistent mask spacing [{self.describe_self()}]."
                    )

            # Acquire mask orientation
            mask_orientation = self._get_mask_orientation(frames=frames)
            for frame in frames:
                frame_orientation = self._get_mask_orientation(frames=[frame])
                if not np.array_equal(frame_orientation, mask_orientation):
                    raise ValueError(
                        f"The DICOM SEG file has inconsistent orientation [{self.describe_self()}]."
                    )

            # For each frame, determine slice position relative to the first frame. First determine frame origins. Then
            # Determine slice position
            mask_origin = self._get_mask_origin(frames=frames)
            frame_origins = [
                self._get_mask_origin(frames=[frame]) for frame in frames
            ]
            if all(frame_origin == mask_origin for frame_origin in frame_origins):
                # If origin is not set per frame, but is shared, assume that frames are sequential.
                mask_slice_number = np.arange(len(frames))
            else:
                frame_voxel_origins = [
                    self.to_voxel_coordinates(
                        x=np.array(frame_origin),
                        origin=np.array(mask_origin),
                        orientation=mask_orientation,
                        spacing=np.array(mask_spacing)
                    )
                    for frame_origin in frame_origins
                ]
                mask_slice_number = [int(x[0]) for x in frame_voxel_origins]

            # Determine mask dimensions.
            mask_dimension = tuple([max(mask_slice_number) + 1, seg_mask_data.shape[1], seg_mask_data.shape[2]])

            # Create data and insert frame at the correct slice.
            mask_data = np.zeros(mask_dimension, dtype=seg_mask_data.dtype)
            for jj, frame in enumerate(frames):
                mask_data[mask_slice_number[jj], :, :] = seg_mask_data[frame, :, :]

            current_roi_name = [
                x
                for jj, x in enumerate(roi_name_present)
                if roi_number_present[jj] == roi_number
            ][0]

            if isinstance(self.roi_name, dict):
                current_roi_name = self.roi_name.get(current_roi_name)

            mask_list += [
                BaseMask(
                    roi_name=current_roi_name,
                    sample_name=self.sample_name,
                    image_modality=self.modality,
                    image_data=mask_data,
                    image_spacing=mask_spacing,
                    image_origin=mask_origin,
                    image_orientation=mask_orientation,
                    image_dimensions=mask_dimension,
                    metadata=self.object_metadata
                )
            ]

        if len(mask_list) == 0:
            return None

        return mask_list

    def _generate_frame_index(self, roi_index: int):
        for ii in np.arange(
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0008), tag_type="int")):
            current_roi_number = None
            if has_pydicom_meta_tag(self.image_metadata, tag=(0x5200, 0x9229)):
                if has_pydicom_meta_tag(self.image_metadata[(0x5200, 0x9229)][0], tag=(0x0062, 0x000A)):
                    current_roi_number = get_pydicom_meta_tag(
                        dcm_seq=self.image_metadata[(0x5200, 0x9229)][0][(0x0062, 0x000A)][0],
                        tag=(0x0062, 0x000B),
                        tag_type="int",
                        default=None
                    )

            if current_roi_number is None:
                current_roi_number = get_pydicom_meta_tag(
                    dcm_seq=self.image_metadata[(0x5200, 0x9230)][ii][(0x0062, 0x000A)][0],
                    tag=(0x0062, 0x000B),
                    tag_type="int",
                    default=None
                )

            if current_roi_number is None:
                raise ValueError(
                    f"The current frame ({ii + 1}) has no associated segment number. [{self.describe_self()}]"
                )

            if roi_index == current_roi_number:
                yield ii

    def _get_mask_origin(self, frames: list[int]) -> tuple[float, ...]:
        origin = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0020, 0x0032),
            tag_type="mult_float",
            macro_dcm_seq=(0x0020, 0x9113),
            frame_id=frames[0]
        )[::-1]

        if origin is None:
            raise ValueError(
                f"The DICOM SEG file lacks a information that is required ",
                f"to set the spatial origin of the mask. [{self.describe_self()}]"
            )

        return tuple(origin)

    def _get_mask_spacing(self, frames: list[int]) -> tuple[float, ...]:

        # Get pixel-spacing.
        spacing = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0028, 0x0030),
            tag_type="mult_float",
            macro_dcm_seq=(0x0028, 0x9110),
            frame_id=frames[0]
        )

        if spacing is None:
            raise ValueError(
                f"The DICOM SEG file lacks in-plane mask spacing [{self.describe_self()}]."
            )

        # First try to get spacing between slices.
        z_spacing = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0018, 0x0088),
            tag_type="float",
            macro_dcm_seq=(0x0028, 0x9110),
            frame_id=frames[0]
        )

        # If spacing between slices is not set, get slice thickness.
        if z_spacing is None:
            z_spacing = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0018, 0x0050),
                tag_type="float",
                macro_dcm_seq=(0x0028, 0x9110),
                frame_id=frames[0]
            )

        if z_spacing is None:
            raise ValueError(
                f"The DICOM SEG file lacks slice spacing [{self.describe_self()}]."
            )

        spacing += [z_spacing]

        return tuple(spacing[::-1])

    def _get_mask_orientation(self, frames: list[int]) -> np.array:
        orientation: list[float] = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0020, 0x0037),
            tag_type="mult_float",
            macro_dcm_seq=(0x0020, 0x9116),
            frame_id=frames[0]
        )

        if orientation is None:
            raise ValueError(
                f"The DICOM SEG file lacks information to determine the mask orientation. "
                f"[{self.describe_self()}]"
            )

        # First compute z-orientation.
        # noinspection PyUnreachableCode
        orientation += list(np.cross(orientation[0:3], orientation[3:6]))
        return np.reshape(orientation[::-1], [3, 3], order="F")

    def export_roi_labels(self):

        self.load_metadata()

        # Find which roi numbers (3006,0022) are associated with which roi names (3004,0024).
        labels = [
            get_pydicom_meta_tag(dcm_seq=current_segment_sequence, tag=(0x0062, 0x0005), tag_type="str", default=None)
            for current_segment_sequence in self.image_metadata[(0x0062, 0x0002)]
        ]

        n_labels = max([1, len(labels)])

        if len(labels) == 0:
            labels = [None]

        # Get general attributes.
        parent_attributes = self._get_export_attributes()

        # Add roi labels as attribute.
        attributes = [("roi_label", labels)]
        parent_attributes.update(dict(attributes))

        return parent_attributes
