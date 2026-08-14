import warnings

import numpy as np
from pydicom import dcmread
from typing import Generator, Self

from skimage import segmentation

from mirp._data_import.dicom_file import MaskDicomFile
from mirp._data_import.utilities import get_pydicom_meta_tag, has_pydicom_meta_tag
from mirp._data_import.dicom_multi_frame import (ImageDicomMultiFrame, ImageDicomMultiFrameStack,
                                                 ImageDicomMultiFrameIndividual)
from mirp._masks.base_mask import BaseMask


class MaskDicomFileSEG(ImageDicomMultiFrame, MaskDicomFile):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create(self) -> Self:
        # Creates stacks of frames. Each stack of frames is its own image, in the sense that they have their own
        # image origin, orientation and so forth.

        # We need access to full metadata here due to functional groups.
        self.load_metadata()

        # Determine the stack identifiers. For SEG files these are the Referenced Segment Number in Segment
        # Identification Sequence.
        stack_identifiers = self.get_pydicom_func_group_tag(
            tag=(0x0062, 0x000B),
            macro_dcm_seq=(0x0062, 0x000A),
            tag_type="str"
        )

        # Create a copy
        image = self.copy()

        frame_stacks = []
        if stack_identifiers is not None:
            # Frames belong to one or more stacks, identified using stack_identifiers.

            for stack_identifier in set(stack_identifiers):
                frame_ids = [ii for ii in range(len(stack_identifiers)) if stack_identifiers[ii] == stack_identifier]
                frame_stack = ImageDicomMultiFrameStack(
                    stack_id=stack_identifier,
                    frame_ids=frame_ids
                )
                frame_stack.update_from_template(template=image)

                # Update to create stacks of frames.
                frame_stack: MaskDicomFileSEGMultiFrameStack = frame_stack.create()
                frame_stack.set_roi_name()
                frame_stacks += [frame_stack]

        if len(frame_stacks) > 0:
            image.stacks = frame_stacks

        return image

    def to_object(self, **kwargs) -> Generator[None | BaseMask, None, None] | None:
        if self.stacks is None:
            warnings.warn(
                f"The current SEG file did not contain any segmentations. [{self.describe_self()}]",
            )
            return None

        # Identify user-provided roi names.
        provided_roi_names = None
        if isinstance(self.roi_name, str):
            provided_roi_names = [self.roi_name]
        elif isinstance(self.roi_name, list):
            provided_roi_names = self.roi_name
        elif isinstance(self.roi_name, dict):
            provided_roi_names = list(self.roi_name.keys())

        if provided_roi_names is None:
            available_stacks = self.stacks
        else:
            available_stacks = [x for x in self.stacks if x.roi_name in provided_roi_names]
            if len(available_stacks) == 0:
                warnings.warn(
                    f"The current SEG file did not contain any of the required ROIs. "
                    f"Required: one or more of {provided_roi_names}. "
                    f"Available: {[x.roi_name for x in self.stacks]}"
                )

        for stack in available_stacks:
            stack.load_data(**kwargs)
            stack.complete()
            stack.update_image_data()
            stack.set_object_metadata()

            yield BaseMask(
                sample_name=stack.sample_name,
                roi_name=stack.roi_name,
                image_modality=stack.modality,
                image_data=stack.image_data,
                image_spacing=stack.image_spacing,
                image_origin=stack.image_origin,
                image_orientation=stack.image_orientation,
                image_dimensions=stack.image_dimension,
                metadata=stack.object_metadata
            )

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


class MaskDicomFileSEGMultiFrameStack(ImageDicomMultiFrameStack, MaskDicomFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_individual_frame_class():
        return MaskDicomFileSEGMultiFrameIndividual

    def load_data(self, **kwargs):
        if self.frames is None:
            return

        self.load_metadata(include_image=True)
        image = np.zeros(self.image_dimension, dtype=bool)
        for frame in self.frames:
            frame.load_data(**kwargs)
            image[frame.in_stack_position - 1, :, :] = frame.image_data

        self._update_attributes_from_frames()
        self.image_data = image

    def set_roi_name(self):
        segment_sequence = self.image_metadata[(0x0062, 0x0002)]

        # Find segment number.
        segment_number = [
            get_pydicom_meta_tag(
                dcm_seq=x,
                tag=(0x0062, 0x0004),
                tag_type="str"
            ) for x in segment_sequence
        ]

        # Find the roi name for the segment sequence element that corresponds to this stack.
        roi_name = [
            get_pydicom_meta_tag(
                dcm_seq=x,
                tag=(0x0062, 0x0006),
                tag_type="str"
            )
            for ii, x in enumerate(segment_sequence)
            if segment_number[ii] == self.stack_id
        ]

        self.roi_name = roi_name[0]


class MaskDicomFileSEGMultiFrameIndividual(ImageDicomMultiFrameIndividual, MaskDicomFileSEGMultiFrameStack):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(
            self,
            **kwargs
    ):
        # Load data.
        super().load_data(**kwargs)

        # Use Segmentation Type (0062,0001)
        segmentation_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0062, 0x0001),
            tag_type="str",
            default="BINARY"
        )

        if not segmentation_type == "BINARY":
            raise NotImplementedError(
                f"Only BINARY segmentation type is currently supported. Found: {segmentation_type}."
            )

        # Convert to boolean
        if self.image_data is np.ndarray:
            self.image_data = self.image_data.astype(bool)
