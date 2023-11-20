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

            pass

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
        mask_position = mask_position.iloc[0]

        # Order ascending position (DICOM: z increases from feet to head)
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
                    f"a Pixel Measure Sequence (0020, 9110) to determine the mask spacing."
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
                    f"a complete Pixel Measure Sequence (0020, 9110) to determine the mask spacing."
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
