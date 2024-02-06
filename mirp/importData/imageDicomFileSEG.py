import warnings

import numpy as np
import pandas as pd
from pydicom import dcmread

from mirp.importData.imageDicomFile import MaskDicomFile
from mirp.importData.utilities import get_pydicom_meta_tag, has_pydicom_meta_tag
from mirp.masks.baseMask import BaseMask


class MaskDicomFileSEG(MaskDicomFile):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def is_stackable(self, stack_images: str):
        return False

    def create(self):
        return self

    def _complete_image_origin(self, force=False):
        return

    def _complete_image_orientation(self, force=False):
        return

    def _complete_image_spacing(self, force=False):
        return

    def _complete_image_dimensions(self, force=False):
        return

    def check_mask(self, raise_error=True):
        if self.image_metadata is None:
            raise TypeError("DEV: the image_meta_data attribute has not been set.")

        if not get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0062, 0x0002), test_tag=True):
            if raise_error:
                warnings.warn(
                    f"The SEG set ({self.file_path}) did not contain any segment sequences.",
                )
            return False

        roi_name_present = [
            get_pydicom_meta_tag(
                dcm_seq=current_segment_sequence, tag=(0x0062, 0x0005), tag_type="str", default=None)
            for current_segment_sequence in self.image_metadata[(0x0062, 0x0002)]
        ]

        if len(roi_name_present) == 0 and raise_error:
            warnings.warn(
                f"The current SEG set ({self.file_path}) does not contain any segmentations."
            )
            return False

        if any(x is None for x in roi_name_present) and raise_error:
            warnings.warn(
                f"The current SEG set ({self.file_path}) does not contain any labels."
            )
            return False

        return True

    def load_data(self, **kwargs):
        pass

    def to_object(self, **kwargs) -> None | list[BaseMask]:

        self.load_metadata()
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

            # Acquire mask origin, mask spacing, orientation and dimension.
            mask_origin = self._get_mask_origin(frames=frames)
            mask_spacing = self._get_mask_spacing(frames=frames)
            mask_orientation = self._get_mask_orientation(frames=frames, spacing=mask_spacing)
            mask_dimension = self._get_mask_dimension(
                frames=frames,
                origin=mask_origin,
                spacing=mask_spacing,
                orientation=mask_orientation,
                frame_shape=seg_mask_data.shape
            )

            # Insert frame at the right slices
            mask_data = np.zeros(mask_dimension, dtype=seg_mask_data.dtype)
            for frame_slice_number in frames:
                mask_slice_number = self._get_mask_slice_number(
                    frame=frame_slice_number,
                    origin=mask_origin,
                    spacing=mask_spacing,
                    orientation=mask_orientation
                )

                # Insert frame at the correct slice.
                mask_data[mask_slice_number, :, :] = seg_mask_data[frame_slice_number, :, :]

            current_roi_name = [
                x
                for ii, x in enumerate(roi_name_present)
                if roi_number_present[ii] == roi_number
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
                    image_dimensions=mask_dimension
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
                raise ValueError(f"The current frame ({ii + 1}) has no associated segment number.")

            if roi_index == current_roi_number:
                yield ii

    def _get_mask_origin(self, frames: list[int]) -> tuple[float, ...]:
        mask_origin = None

        # Check if a Shared Functional Groups Sequence exists.
        if has_pydicom_meta_tag(self.image_metadata, tag=(0x5200, 0x9229)):
            if has_pydicom_meta_tag(self.image_metadata[(0x5200, 0x9229)][0], tag=(0x0020, 0x9113)):
                mask_origin = get_pydicom_meta_tag(
                    dcm_seq=self.image_metadata[(0x5200, 0x9229)][0][(0x0020, 0x9113)][0],
                    tag=(0x0020, 0x0032),
                    tag_type="mult_float",
                    default=None
                )

        if mask_origin is not None:
            return tuple(mask_origin[::-1])

        # Check that the Per-Frame Functional Groups Sequence exists.
        if not has_pydicom_meta_tag(self.image_metadata, tag=(0x5200, 0x9230)):
            raise ValueError(
                f"The DICOM SEG file ({self.file_path}) lacks a Per-Frame Functions Groups Sequence that is required ",
                f"to set the spatial origin of the mask."
            )

        mask_position = self._get_origin_position_table(frames=frames)
        mask_position = mask_position.iloc[0]

        return tuple([mask_position.position_z, mask_position.position_y, mask_position.position_x])

    def _get_mask_spacing(self, frames: list[int]) -> tuple[float, ...]:
        mask_spacing = None

        # Check if a Shared Functional Groups Sequence exists.
        if has_pydicom_meta_tag(self.image_metadata, tag=(0x5200, 0x9229)):
            if has_pydicom_meta_tag(self.image_metadata[(0x5200, 0x9229)][0], tag=(0x0028, 0x9110)):
                pixel_spacing = get_pydicom_meta_tag(
                    dcm_seq=self.image_metadata[(0x5200, 0x9229)][0][(0x0028, 0x9110)][0],
                    tag=(0x0028, 0x0030),
                    tag_type="mult_float",
                    default=None
                )

                slice_spacing = get_pydicom_meta_tag(
                    dcm_seq=self.image_metadata[(0x5200, 0x9229)][0][(0x0028, 0x9110)][0],
                    tag=(0x0018, 0x0088),
                    tag_type="float",
                    default=None
                )

                if pixel_spacing is not None and slice_spacing is not None:
                    mask_spacing = pixel_spacing + [slice_spacing]

        if mask_spacing is not None:
            return tuple(mask_spacing[::-1])

        # Isolate Functional Groups Sequence for each frame.
        frame_functional_groups = [
            frame_functional_group
            for ii, frame_functional_group in enumerate(self.image_metadata[(0x5200, 0x9230)])
            if ii in frames
        ]

        if not len(frame_functional_groups) == len(frames):
            raise ValueError(
                f"The DICOM SEG file ({self.file_path}) does not have the same number of per-frame functional group "
                f"sequences ({len(frame_functional_groups)}) as the expected number of frames containing a part of "
                f"the segmentation ({len(frames)}."
            )

        mask_spacing_x = None
        mask_spacing_y = None
        mask_spacing_z = None
        for ii, frame_functional_group in enumerate(frame_functional_groups):
            if not has_pydicom_meta_tag(frame_functional_group, tag=(0x0028, 0x9110)):
                raise ValueError(
                    f"One or more Per-Frame Functional Group Sequences of the DICOM SEG file ({self.file_path}) lack "
                    f"a Pixel Measure Sequence (0028, 9110) to determine the mask spacing."
                )

            pixel_spacing = get_pydicom_meta_tag(
                dcm_seq=frame_functional_group[(0x0028, 0x9110)][0],
                tag=(0x0028, 0x0030),
                tag_type="mult_float",
                default=None
            )

            slice_spacing = get_pydicom_meta_tag(
                dcm_seq=frame_functional_group[(0x0028, 0x9110)][0],
                tag=(0x0018, 0x0088),
                tag_type="float",
                default=None
            )

            if pixel_spacing is None or slice_spacing is None:
                raise ValueError(
                    f"One or more Per-Frame Functional Group Sequences of the DICOM SEG file ({self.file_path}) lack "
                    f"a complete Pixel Measure Sequence (0028, 9110) to determine the mask spacing."
                )

            if mask_spacing_x is None or mask_spacing_y is None or mask_spacing_z is None:
                mask_spacing_x = pixel_spacing[0]
                mask_spacing_y = pixel_spacing[1]
                mask_spacing_z = slice_spacing

            elif not (mask_spacing_x == pixel_spacing[0] and mask_spacing_y == pixel_spacing[1] and mask_spacing_z ==
                      slice_spacing):
                raise ValueError(
                    f"Inconsistent spacing detected for one or more frames of the DICOM SEG file ({self.file_path})."
                )
            else:
                pass

        return tuple([mask_spacing_z, mask_spacing_y, mask_spacing_x])

    def _get_mask_orientation(self, frames: list[int], spacing: tuple[float, ...]) -> np.array:
        mask_orientation = None

        # Check if a Shared Functional Groups Sequence exists.
        if has_pydicom_meta_tag(self.image_metadata, tag=(0x5200, 0x9229)):
            if has_pydicom_meta_tag(self.image_metadata[(0x5200, 0x9229)][0], tag=(0x0028, 0x9110)):
                mask_orientation = get_pydicom_meta_tag(
                    dcm_seq=self.image_metadata[(0x5200, 0x9229)][0][(0x0020, 0x9116)][0],
                    tag=(0x0020, 0x0037),
                    tag_type="mult_float",
                    default=None
                )

        if mask_orientation is not None:
            mask_orientation += list(np.cross(mask_orientation[0:3], mask_orientation[3:6]))
            return np.reshape(mask_orientation[::-1], [3, 3], order="F")

        # Isolate Functional Groups Sequence for each frame.
        frame_functional_groups = [
            frame_functional_group
            for ii, frame_functional_group in enumerate(self.image_metadata[(0x5200, 0x9230)])
            if ii in frames
        ]

        if not len(frame_functional_groups) == len(frames):
            raise ValueError(
                f"The DICOM SEG file ({self.file_path}) does not have the same number of per-frame functional group "
                f"sequences ({len(frame_functional_groups)}) as the expected number of frames containing a part of "
                f"the segmentation ({len(frames)}."
            )

        for ii, frame_functional_group in enumerate(frame_functional_groups):
            if not has_pydicom_meta_tag(frame_functional_group, tag=(0x0020, 0x9116)):
                raise ValueError(
                    f"One or more Per-Frame Functional Group Sequences of the DICOM SEG file ({self.file_path}) lack "
                    f"a Pixel Measure Sequence (0020, 9110) to determine the mask spacing."
                )

            frame_mask_orientation = get_pydicom_meta_tag(
                dcm_seq=frame_functional_group[(0x0020, 0x9116)][0],
                tag=(0x0020, 0x0037),
                tag_type="mult_float",
                default=None
            )

            if frame_mask_orientation is None:
                raise ValueError(
                    f"One or more Per-Frame Functional Group Sequences of the DICOM SEG file ({self.file_path}) lack "
                    f"a complete Pixel Orientation Sequence (0020, 9116) to determine the mask orientation."
                )

            if mask_orientation is None:
                mask_orientation = frame_mask_orientation

            elif not (mask_orientation == frame_mask_orientation):
                raise ValueError(
                    f"Inconsistent orientation detected for one or more frames of the DICOM SEG file "
                    f"({self.file_path})."
                )
            else:
                pass

        mask_orientation += list(np.cross(mask_orientation[0:3], mask_orientation[3:6]))
        return np.reshape(mask_orientation[::-1], [3, 3], order="F")

    def _get_mask_slice_number(
            self,
            frame: int,
            origin: tuple[float, ...],
            spacing: tuple[float, ...],
            orientation: np.ndarray
    ) -> int:
        # Check if frame has its own origin.
        frame_origin = None
        if has_pydicom_meta_tag(self.image_metadata, tag=(0x5200, 0x9230)):
            if has_pydicom_meta_tag(self.image_metadata[(0x5200, 0x9230)][frame], tag=(0x0020, 0x9113)):
                frame_origin = get_pydicom_meta_tag(
                    dcm_seq=self.image_metadata[(0x5200, 0x9230)][frame][(0x0020, 0x9113)][0],
                    tag=(0x0020, 0x0032),
                    tag_type="mult_float",
                    default=None
                )

        if frame_origin is not None:
            # When the frame has a known origin, use this origin to determine the corresponding slice in the voxel
            # coordinate system.
            frame_origin = frame_origin[::-1]

            voxel_origin = self.to_voxel_coordinates(
                x=np.array(frame_origin),
                origin=np.array(origin),
                orientation=orientation,
                spacing=np.array(spacing)
            )

            return int(voxel_origin[0])

        else:
            # When the frame lacks an origin, we should assume that the frame indicates the slice, i.e. there is only
            # a single segment that should be considered.
            return frame

    def _get_origin_position_table(self, frames: list[int]):
        # Placeholders for mask positions.
        mask_position_z = [0.0] * len(frames)
        mask_position_y = [0.0] * len(frames)
        mask_position_x = [0.0] * len(frames)

        # Isolate Functional Groups Sequence for each frame.
        frame_functional_groups = [
            frame_functional_group
            for ii, frame_functional_group in enumerate(self.image_metadata[(0x5200, 0x9230)])
            if ii in frames
        ]

        if not len(frame_functional_groups) == len(frames):
            raise ValueError(
                f"The DICOM SEG file ({self.file_path}) does not have the same number of per-frame functional group "
                f"sequences ({len(frame_functional_groups)}) as the expected number of frames containing a part of "
                f"the segmentation ({len(frames)}."
            )

        for ii, frame_functional_group in enumerate(frame_functional_groups):
            if not has_pydicom_meta_tag(frame_functional_group, tag=(0x0020, 0x9113)):
                raise ValueError(
                    f"One or more Per-Frame Functional Group Sequences of the DICOM SEG file ({self.file_path}) lack "
                    f"a Plane Position Sequence (0020, 9113) to determine the mask origin."
                )

            slice_origin = get_pydicom_meta_tag(
                dcm_seq=frame_functional_group[(0x0020, 0x9113)][0],
                tag=(0x0020, 0x0032),
                tag_type="mult_float",
                default=None
            )

            if slice_origin is None:
                raise ValueError(
                    f"One or more Per-Frame Functional Group Sequences of the DICOM SEG file ({self.file_path}) lack "
                    f"a complete Plane Position Sequence (0020, 9113) to determine the mask origin."
                )
            slice_origin = slice_origin[::-1]

            mask_position_z[ii] = slice_origin[0]
            mask_position_y[ii] = slice_origin[1]
            mask_position_x[ii] = slice_origin[2]

        mask_position = pd.DataFrame({
            "original_object_order": list(range(len(frames))),
            "position_z": mask_position_z,
            "position_y": mask_position_y,
            "position_x": mask_position_x
        }).sort_values(by=["position_z", "position_y", "position_x"], ignore_index=True)

        return mask_position

    def _get_mask_dimension(
            self,
            frames: list[int],
            origin: tuple[float, ...],
            spacing: tuple[float, ...],
            orientation: np.ndarray,
            frame_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        # The tricky part is to determine the number of required slices.
        position_table = self._get_origin_position_table(frames=frames)
        position_table = position_table.iloc[-1]
        max_extent_world = tuple([position_table.position_z, position_table.position_y, position_table.position_x])
        max_extent_voxel = self.to_voxel_coordinates(
            x=np.array(max_extent_world),
            origin=np.array(origin),
            orientation=orientation,
            spacing=np.array(spacing),
        )

        return tuple([int(max_extent_voxel[0]) + 1, frame_shape[1], frame_shape[2]])

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

        return {
            "sample_name": [self.sample_name] * n_labels,
            "dir_path": [self.dir_path] * n_labels,
            "file_path": [self.file_name] * n_labels,
            "roi_label": labels
        }
