import os.path
from keyword import kwlist

import pydicom

import numpy as np

from typing import Any, Self, Generator
from mirp._images.generic_image import GenericImage
from mirp._data_import.dicom_file import ImageDicomFile
from mirp._data_import.utilities import get_pydicom_meta_tag, has_pydicom_meta_tag


class ImageDicomMultiFrame(ImageDicomFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stacks: None | list[ImageDicomMultiFrameStack] = None

    def create(self) -> Self:
        # Creates stacks of frames. Each stack of frames is its own image, in the sense that they have their own
        # image origin, orientation and so forth.

        # We need access to full metadata here due to functional groups.
        self.load_metadata()

        # Determine the stack identifiers.
        stack_identifiers = self.get_pydicom_func_group_tag(
            tag=(0x0020, 0x9056),
            macro_dcm_seq=(0x0020, 0x9111),
            tag_type="str"
        )

        # Create a copy
        image = self.copy()

        frame_stacks = []
        for stack_identifier in set(stack_identifiers):
            frame_ids = [ii for ii in range(len(stack_identifiers)) if stack_identifiers[ii] == stack_identifier]
            frame_stack = ImageDicomMultiFrameStack(
                stack_id=stack_identifier,
                frame_ids=frame_ids
            )
            frame_stack.update_from_template(template=image)

            # Update to create stacks of frames.
            frame_stack = frame_stack.create()
            frame_stacks += [frame_stack]

        if len(frame_stacks) > 0:
            image.stacks = frame_stacks

        return image

    def is_stackable(self, stack_images: str):
        # Multi-frame images might be actually be stackable (concatenated), but ignore that for now.
        return False

    def load_data(self, **kwargs):
        if self.stacks is not None:
            for stack in self.stacks:
                stack.load_data(**kwargs)

    def to_object(self, **kwargs) -> Generator[GenericImage, None, None]:
        if self.stacks is None:
            raise ValueError(f"Stacks of a multiframe DICOM object cannot be empty. {self.describe_self()}")

        for stack in self.stacks:
            for substack in stack.create_real_world_unit_stacks(**kwargs):
                substack.load_data(**kwargs)
                substack.complete()
                substack.update_image_data()
                substack.set_object_metadata()

                yield GenericImage(
                    sample_name=substack.sample_name,
                    image_modality=substack.modality,
                    image_data=substack.image_data,
                    image_spacing=substack.image_spacing,
                    image_origin=substack.image_origin,
                    image_orientation=substack.image_orientation,
                    image_dimensions=substack.image_dimension,
                    metadata=substack.object_metadata
                )

    def _complete_image_origin(self, force=False):
        if self.stacks is not None:
            for stack in self.stacks:
                stack._complete_image_origin(force=force)

    def _complete_image_orientation(self, force=False):
        if self.stacks is not None:
            for stack in self.stacks:
                stack._complete_image_orientation(force=force)

    def _complete_image_spacing(self, force=False):
        if self.stacks is not None:
            for stack in self.stacks:
                stack._complete_image_spacing(force=force)

    def _complete_image_dimensions(self, force=False):
        if self.stacks is not None:
            for stack in self.stacks:
                stack._complete_image_dimensions(force=force)

    def _get_n_frames(self):
        self.load_metadata()
        return get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0008), tag_type="int", default=0)

    def _get_pydicom_func_group_tag(
            self,
            tag: tuple[int, int],
            tag_type: None | str = None,
            default: Any = None,
            macro_dcm_seq: None | tuple[int, int] | list[tuple[int, int]] = None,
            frame_id: None | int | list[int] = None,
            test_tag: bool = False,
            check_all_none: bool = True
    ) -> Any:

        # Ensure that frame_id is a list.
        if isinstance(frame_id, int):
            frame_id = [frame_id]

        share_macro_dcm_tag = (0x5200, 0x9229)
        frame_macro_dcm_tag = (0x5200, 0x9230)

        frame_value = [get_pydicom_meta_tag(
            dcm_seq=self.image_metadata[frame_macro_dcm_tag][frame_id_ii],
            tag=tag,
            tag_type=tag_type,
            macro_dcm_seq=macro_dcm_seq,
            default=None,
            test_tag=test_tag
        ) for frame_id_ii in frame_id]

        if test_tag and all(x == True for x in frame_value):
            return True

        if not all(x is None for x in frame_value):
            return frame_value

        # Attempt to get from shared group.
        share_value = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata[share_macro_dcm_tag][0],
            tag=tag,
            tag_type=tag_type,
            macro_dcm_seq=macro_dcm_seq,
            default=default,
            test_tag=test_tag
        )

        if test_tag:
            return share_value  # True if tag is present, and False if not.

        if share_value is None and check_all_none:
            return None

        return [share_value] * len(frame_id)

    def get_pydicom_func_group_tag(
            self,
            tag: tuple[int, int],
            tag_type: None | str = None,
            default: Any = None,
            macro_dcm_seq: None | tuple[int, int] | list[tuple[int, int]] = None,
            frame_id: None | int | list[int] = None,
            test_tag: bool = False,
            check_all_none: bool = True
    ) -> Any:

        if frame_id is None:
            n_frames = self._get_n_frames()
            if n_frames is None or n_frames == 0:
                if test_tag:
                    return False
                return None

            frame_id = list(range(n_frames))

        return self._get_pydicom_func_group_tag(
            tag = tag,
            tag_type=tag_type,
            default=default,
            macro_dcm_seq=macro_dcm_seq,
            frame_id=frame_id,
            test_tag=test_tag,
            check_all_none=check_all_none
        )

    def _check_is_mr_adc(self):
        # Check for ADC images. ADC can sometimes by identified the ADC value in the Image Type (0008,0008) tag,
        # the frame type tag (0008, 9007) or acquisition contrast (0008, 9209) [though this typically should be
        # DIFFUSION].
        image_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x0008),
            tag_type="mult_str"
        )
        frame_type = self.get_pydicom_func_group_tag(
            tag=(0x0008, 0x9007),
            tag_type="mult_str",
            macro_dcm_seq=(0x0018, 0x9226)
        )
        alt_frame_type = self.get_pydicom_func_group_tag(
            tag=(0x0008, 0x9007),
            tag_type="mult_str",
            macro_dcm_seq=(0x0040, 0x9092)
        )
        acquisition_contrast = self.get_pydicom_func_group_tag(
            tag=(0x0008, 0x9209),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9226)
        )

        if image_type is not None and any(x.lower() == "adc" for x in image_type):
            return True
        elif frame_type is not None and any(x.lower() == "adc" for x in frame_type):
            return True
        elif alt_frame_type is not None and any(x.lower() == "adc" for x in alt_frame_type):
            return True
        elif acquisition_contrast is not None and acquisition_contrast.lower() == "adc":
            return True

        return False

    def _check_is_mr_dce(self):
        # Check for DCE images. DCE can sometimes by identified the DCE value in the Image Type (0008,0008) tag,
        # the frame type tag (0008, 9007) or acquisition contrast (0008, 9209) [though this typically should be
        # DIFFUSION].
        image_type = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x0008),
            tag_type="mult_str"
        )
        frame_type = self.get_pydicom_func_group_tag(
            tag=(0x0008, 0x9007),
            tag_type="mult_str",
            macro_dcm_seq=(0x0018, 0x9226)
        )
        alt_frame_type = self.get_pydicom_func_group_tag(
            tag=(0x0008, 0x9007),
            tag_type="mult_str",
            macro_dcm_seq=(0x0040, 0x9092)
        )
        acquisition_contrast = self.get_pydicom_func_group_tag(
            tag=(0x0008, 0x9209),
            tag_type="str",
            macro_dcm_seq=(0x0018, 0x9226)
        )

        if image_type is not None and any(x.lower() == "dce" for x in image_type):
            return True
        elif frame_type is not None and any(x.lower() == "dce" for x in frame_type):
            return True
        elif alt_frame_type is not None and any(x.lower() == "dce" for x in alt_frame_type):
            return True
        elif acquisition_contrast is not None and acquisition_contrast.lower() == "dce":
            return True

        return False


class ImageDicomMultiFrameStack(ImageDicomMultiFrame):
    def __init__(
            self,
            stack_id: None | str = None,
            frame_ids: None | list[int] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.stack_id = stack_id
        self.frame_ids = frame_ids
        self.frames: list[ImageDicomMultiFrameIndividual] | None = None
        self.real_world_unit: str | None = None

    def create(self):
        # This method is called from ImageDicomMultiFrame.create amd dispatches to modality-specific multi-frame
        # objects. The frame stacks are modality-specific.
        from mirp._data_import.dicom_file_enhanced_ct import ImageDicomFileCTMultiFrameStack
        from mirp._data_import.dicom_file_enhanced_pet import ImageDicomFilePTMultiFrameStack
        from mirp._data_import.dicom_file_enhanced_mr import ImageDicomFileMRMultiFrameStack
        from mirp._data_import.dicom_file_mr_adc import ImageDicomFileMRADCMultiFrameStack
        from mirp._data_import.dicom_file_mr_dce import ImageDicomFileMRDCEMultiFrameStack

        if self.modality == "ct":
            file_class = ImageDicomFileCTMultiFrameStack
        elif self.modality == "pt":
            file_class = ImageDicomFilePTMultiFrameStack
        elif self.modality == "mr":
            file_class = ImageDicomFileMRMultiFrameStack
        elif self.modality == "adc":
            file_class = ImageDicomFileMRADCMultiFrameStack
        elif self.modality == "dce":
            file_class = ImageDicomFileMRDCEMultiFrameStack

        else:
            # Multi-frame is not implemented for the following modalities:
            # - RT Dose: lack of DICOM module for RT Dose with multi-frame data.
            raise NotImplementedError(
                f"Multi-frame DICOM not implemented for {self.modality} modality. Contact the devs."
            )

        stack = file_class()
        stack.update_from_template(self)

        if stack.frame_ids is not None:
            in_stack_position = stack.get_pydicom_func_group_tag(
                tag=(0x0020, 0x9057),
                macro_dcm_seq=(0x0020, 0x9111),
                tag_type="int",
                frame_id=stack.frame_ids
            )
            frames = [None] * max(in_stack_position)
            for ii, frame_id in enumerate(stack.frame_ids):
                individual_frame = stack.create_individual_frame(
                    frame_id=frame_id,
                    in_stack_position=in_stack_position[ii]
                )
                # Stack position is 1-indexed. Ensure that the frames are positioned in order, so that stack
                # information such as image_spacing can be computed.
                frames[in_stack_position[ii] - 1] = individual_frame

            # Check that all frames are positioned as expected.
            if any(x is None for x in frames):
                raise ValueError(
                    f"Not all positions in the DICOM MultiFrame stack were filled. "
                    f"Some frames may be mapped to the same stack position (0020, 9057), or not all frames are "
                    f"present. {self.describe_self()}"
                )

            stack.frames = frames
            stack.frame_ids = [x.frame_id for x in stack.frames]

        return stack

    def create_real_world_unit_stacks(self, **kwargs) -> Generator[Self, None, None] | None:
        if self.frames is None:
            return None

        # Find real world value units. Can be none (in which case rescale intercept and offset are used as a fallback
        # option).
        rw_units, rw_schemes = self._get_real_world_units(**kwargs)
        if rw_units is None:
            rw_units = [None]

        for rw_unit in rw_units:
            # Check that every frame in the stack has the rw_unit.
            rw_unit_present_in_all_frames = all(frame.has_real_world_unit(rw_unit) for frame in self.frames)
            if not rw_unit_present_in_all_frames:
                continue

            substack = self.copy()
            substack.real_world_unit = rw_unit

            yield substack

    def create_individual_frame(
            self,
            frame_id: int,
            in_stack_position: int
    ):
        # Creates a frame for the current class. The _get_individual_frame_class function is defined in inheriting
        # classes and returns the class definitions of the corresponding frames.
        frame_class = self._get_individual_frame_class()
        frame = frame_class(
            frame_id=frame_id,
            in_stack_position=in_stack_position
        )
        frame.update_from_template(self)

        return frame

    @staticmethod
    def _get_individual_frame_class():
        raise NotImplementedError("_get_individual_frame_class method is missing an implementation in inheriting classes.")

    def update_from_template(self, template: Self):
        from copy import deepcopy

        super().update_from_template(template)

        if isinstance(template, ImageDicomMultiFrameStack):
            self.stack_id = deepcopy(template.stack_id)
            self.frame_ids = deepcopy(template.frame_ids)
            self.frames = deepcopy(template.frames)

    def get_pydicom_func_group_tag(
            self,
            tag: tuple[int, int],
            tag_type: None | str = None,
            default: Any = None,
            macro_dcm_seq: None | tuple[int, int] | list[tuple[int, int]] = None,
            frame_id: None | int | list[int] = None,
            test_tag: bool = False,
            check_all_none: bool = True
    ):
        # Ensure that only function groups related to the current object are accessed.
        if self.frame_ids is None or len(self.frame_ids) == 0:
            if test_tag:
                return False
            return None

        return self._get_pydicom_func_group_tag(
            tag=tag,
            tag_type=tag_type,
            default=default,
            macro_dcm_seq=macro_dcm_seq,
            frame_id=self.frame_ids,
            test_tag=test_tag,
            check_all_none=check_all_none
        )

    def _complete_image_origin(self, force=False):
        if self.image_origin is None:
            # Load relevant metadata.
            self.load_metadata()

            origin = self.get_pydicom_func_group_tag(
                tag=(0x0020, 0x0032),
                tag_type="mult_float",
                macro_dcm_seq=(0x0020, 0x9113),
            )[0][::-1]
            self.image_origin = tuple(origin)

    def _complete_image_orientation(self, force=False):
        if self.image_orientation is None:
            # Load relevant metadata.
            self.load_metadata()

            orientation: list[float] = self.get_pydicom_func_group_tag(
                tag=(0x0020, 0x0037),
                tag_type="mult_float",
                macro_dcm_seq=(0x0020, 0x9116)
            )[0]

            # First compute z-orientation.
            # noinspection PyUnreachableCode
            orientation += list(np.cross(orientation[0:3], orientation[3:6]))
            self.image_orientation = np.reshape(orientation[::-1], [3, 3], order="F")

    def _complete_image_spacing(self, force=False):
        if self.image_spacing is None:
            # Load relevant metadata.
            self.load_metadata()

            # Get pixel-spacing.
            spacing = self.get_pydicom_func_group_tag(
                tag=(0x0028, 0x0030),
                tag_type="mult_float",
                macro_dcm_seq=(0x0028, 0x9110)
            )[0]

            # First try to get spacing between slices.
            z_spacing = self.get_pydicom_func_group_tag(
                tag=(0x0018, 0x0088),
                tag_type="float",
                macro_dcm_seq=(0x0028, 0x9110)
            )
            if z_spacing is not None:
                z_spacing = z_spacing[0]

            # Try to compute spacing between slices based on slice origin,
            if z_spacing is None:
                frame_origins = self.get_pydicom_func_group_tag(
                    tag=(0x0020, 0x0032),
                    tag_type="mult_float",
                    macro_dcm_seq=(0x0020, 0x9113),
                )
                if len(frame_origins) > 1:
                    z_spacing = np.sqrt(np.sum(np.power(np.array(frame_origins[0]) - np.array(frame_origins[1]), 2.0)))

            # Try to use slice thickness.
            if z_spacing is None:
                z_spacing = self.get_pydicom_func_group_tag(
                    tag=(0x0018, 0x0050),
                    tag_type="float",
                    macro_dcm_seq=(0x0028, 0x9110)
                )
                if z_spacing is not None:
                    z_spacing = z_spacing[0]

            # If slice thickness is not set, use a default value.
            if z_spacing is None:
                z_spacing = 1.0

            spacing += [z_spacing]

            self.image_spacing = tuple(spacing[::-1])

    def _complete_image_dimensions(self, force=False):
        if self.image_dimension is None and self.frames is not None:
            # Load relevant metadata.
            self.load_metadata()

            dimensions = tuple([
                len(self.frames),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0010), tag_type="int"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0011), tag_type="int")
            ])

            self.image_dimension = dimensions

    def _get_n_frames(self):
        if self.frame_ids is None:
            return 0

        return len(self.frame_ids)

    def load_data(self, **kwargs):
        if self.frames is None:
            return

        image = np.zeros(self.image_dimension, dtype=np.float32)
        for frame in self.frames:
            frame.load_data(**kwargs)
            image[frame.in_stack_position-1, :, :] = frame.image_data

        self.image_data = image

    def _get_real_world_units(self, **kwargs):
        # Metadata is required to assess units (0008,0100) and coding scheme designator (0008,0102) from
        # the Measurement Units Code Sequence (0040,08EA) in a Real World Value Mapping Sequence (0040,9096).
        # Multiple Real World Value Mapping Sequences may be present for each Frame or Image.
        self.load_metadata()

        _get_real_world_unit


class ImageDicomMultiFrameIndividual(ImageDicomMultiFrame):
    def __init__(
            self,
            frame_id: int,
            in_stack_position: int,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.frame_id = frame_id
        self.in_stack_position = in_stack_position

    def load_data(self, **kwargs):
        # Only loads data. Convert pixel values using Real World Value Mapping Sequences in downstream methods.
        if self.image_data is not None:
            return

        if self.file_path is not None and not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The image file could not be found at the expected location: {self.file_path}. [{self.describe_self()}]"
            )

        if self.file_path is None:
            raise ValueError(f"A path to a file was expected, but not present. [{self.describe_self()}]")

        # Load metadata.
        self.load_metadata(include_image=True)

        image_data = self.image_metadata.pixel_array.astype(np.float32)[self.in_stack_position-1, :, :]

        # Do not perform any transformations to pixel values here -- use data from Real World Value Mapping Sequences
        # instead.
        self.image_data = image_data

    def get_pydicom_func_group_tag(
            self,
            tag: tuple[int, int],
            tag_type: None | str = None,
            default: Any = None,
            macro_dcm_seq: None | tuple[int, int] | list[tuple[int, int]] = None,
            frame_id: None | int | list[int] = None,
            test_tag: bool = False,
            check_all_none: bool = True
    ):
        # Ensure that only function groups related to the current object are accessed.
        if self.frame_id is None:
            if test_tag:
                return False
            return None

        value = self._get_pydicom_func_group_tag(
            tag=tag,
            tag_type=tag_type,
            default=default,
            macro_dcm_seq=macro_dcm_seq,
            frame_id=self.frame_id,
            test_tag=test_tag,
            check_all_none=check_all_none
        )

        if isinstance(value, list) and len(value) == 1:
            value = value[0]

        return value

    def _get_real_world_unit(self, **kwargs) -> None | list[str]:
        # Metadata is required to assess units (0008,0100) and coding scheme designator (0008,0102) from
        # the Measurement Units Code Sequence (0040,08EA) in a Real World Value Mapping Sequence (0040,9096).
        # Multiple Real World Value Mapping Sequences may be present for each Frame or Image.
        self.load_metadata()

        real_world_value_mapping_sequences = self._get_real_world_mapping_sequence()
        if real_world_value_mapping_sequences is None:
            return None

        coding_values = []
        for real_world_value_mapping_sequence in real_world_value_mapping_sequences:
            measurement_units_coding_value = get_pydicom_meta_tag(
                dcm_seq=real_world_value_mapping_sequence,
                macro_dcm_seq=(0x040, 0x08EA),
                tag=(0x0008, 0x0100),
                tag_type="str"
            )
            measurement_units_coding_scheme = get_pydicom_meta_tag(
                dcm_seq=real_world_value_mapping_sequence,
                macro_dcm_seq=(0x040, 0x08EA),
                tag=(0x0008, 0x0102),
                tag_type="str"
            )
            # Skip if the coding scheme is not DCM (DICOM) or UCUM (Unified Code for Units of Measure).
            if measurement_units_coding_scheme is None or \
                    measurement_units_coding_scheme not in ["DCM", "UCUM"]:
                continue

            coding_values += [measurement_units_coding_value]

        if len(coding_values) == 0:
            return None
        return coding_values

    def _has_real_world_unit(self, x: str | None) -> bool:
        if x is None:
            return True

        available_real_world_units = self._get_real_world_unit()
        if available_real_world_units is None:
            return False

        return x in available_real_world_units

    def _get_real_world_mapping_sequence(self) -> pydicom.DataElement | None:
        rw_sequences = self.get_pydicom_func_group_tag(
            tag=(0x0040, 0x9096),
            tag_type="pydicom"
        )

        if rw_sequences is None:
            return None
        return rw_sequences
