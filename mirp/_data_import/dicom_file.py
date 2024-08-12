import sys
import datetime
import os.path
import hashlib
import numpy as np

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from typing import Any

from pydicom import dcmread
from warnings import warn
from copy import deepcopy

from mirp._data_import.generic_file import ImageFile, MaskFile
from mirp._data_import.utilities import supported_image_modalities, stacking_dicom_image_modalities, \
    supported_mask_modalities, get_pydicom_meta_tag, convert_dicom_time, has_pydicom_meta_tag


class ImageDicomFile(ImageFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # These are set using the 'complete' method.
        self.series_instance_uid: None | str = None
        self.frame_of_reference_uid: None | str = None
        self.sop_instance_uid: None | str = None

    def is_stackable(self, stack_images: str):
        """
        Is the image potentially stackable?
        :param stack_images: One of auto, yes or no. Ignored for DICOM imaging, as stackability can be determined
        from metadata.
        :return: boolean value, here true.
        """
        return True

    def get_identifiers(self, as_hash=False) -> dict[str, Any] | bytes:
        """
        General identifiers for images. Compared to other
        :return: a dictionary with identifiers.
        """

        identifier_data = dict({
            "modality": [self.modality],
            "file_type": [self.file_type],
            "sample_name": [self.get_sample_name()],
            "dir_path": [self.get_dir_path()],
            "series_instance_uid": [self.series_instance_uid],
            "frame_of_reference_uid": [self.frame_of_reference_uid]
        })

        if as_hash:
            return hashlib.sha256(str(identifier_data).encode(), usedforsecurity=False).digest()
        else:
            return identifier_data

    def check(self, raise_error=False, remove_metadata=True) -> bool:

        # Metadata needs to be read from files, and should therefore be skipped if not available.
        if self.file_path is None:
            return True

        # Checks requires metadata.
        self.load_metadata(limited=True)

        # Perform general checks.
        if not super().check(raise_error=raise_error):
            if remove_metadata:
                self.remove_metadata()
            return False

        if remove_metadata:
            self.remove_metadata()

        return True

    def _check_modality(self, raise_error: bool) -> bool:
        dicom_modality = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x0060), tag_type="str")

        # Check for ADC images
        if self._check_is_mr_adc():
            dicom_modality = "adc"

        support_modalities = supported_image_modalities(self.modality)
        if dicom_modality.lower() not in support_modalities:
            if raise_error:
                raise ValueError(
                    f"The current DICOM file does not have the expected modality. "
                    f"Found: {dicom_modality.lower()}. Expected: one of {', '.join(support_modalities)}. "
                    f"[{self.describe_self()}]"
                )

            return False

        return True

    def _check_sample_name(self, raise_error: bool) -> bool:
        if self.sample_name is not None:
            allowed_sample_name = self.sample_name
            if not isinstance(allowed_sample_name, list):
                allowed_sample_name = [allowed_sample_name]

            # Consider the following DICOM elements: study description, series description, patient name,
            # patient id, study id.
            dicom_sample_name = [
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x1030), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x103E), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0010, 0x0010), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0010, 0x0020), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0020, 0x0010), tag_type="str")
            ]

            dicom_sample_name = [
                current_dicom_sample_name for current_dicom_sample_name in dicom_sample_name
                if current_dicom_sample_name is not None
            ]

            if len(dicom_sample_name) > 0:
                matching_sample_name = set(dicom_sample_name).intersection(set(allowed_sample_name))
                if len(matching_sample_name) == 0:
                    if raise_error:
                        raise ValueError(
                            f"The current DICOM file does not have a matching sample name among potential identifiers. "
                            f"Potential identifiers: {', '.join(dicom_sample_name)}; "
                            f"Expected identifiers: {', '.join(allowed_sample_name)}. "
                            f"[{self.describe_self()}]"
                        )
                    else:
                        return False

                return True
        else:
            return True

    def create(self):
        """
        DICOM-files have different routines depending on the modality. These are then dispatched to different classes
        using this method.
        :return: an object of a subclass of ImageDicomFile.
        """

        # Import locally to prevent circular references.
        from mirp._data_import.dicom_file_ct import ImageDicomFileCT
        from mirp._data_import.dicom_file_mr import ImageDicomFileMR
        from mirp._data_import.dicom_file_mr_adc import ImageDicomFileMRADC
        from mirp._data_import.dicom_file_pet import ImageDicomFilePT
        from mirp._data_import.dicom_file_rtdose import ImageDicomFileRTDose
        from mirp._data_import.dicom_multi_frame import ImageDicomMultiFrame

        # Load metadata so that the modality tag can be read.
        self.load_metadata(limited=True)
        modality = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x0060), tag_type="str").lower()

        if modality is None:
            raise TypeError(f"Modality attribute could not be obtained from DICOM file. [{self.describe_self()}]")

        if self._check_is_mr_adc():
            modality = "adc"

        if modality == "ct":
            file_class = ImageDicomFileCT
        elif modality == "pt":
            file_class = ImageDicomFilePT
        elif modality == "mr":
            file_class = ImageDicomFileMR
        elif modality == "adc":
            file_class = ImageDicomFileMRADC
        elif modality == "rtdose":
            file_class = ImageDicomFileRTDose
        else:
            # This will return a base class, which will fail to pass the modality check.
            return None

        if has_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x5200, 0x9299)) or \
                has_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x5200, 0x9230)):
            file_class = ImageDicomMultiFrame

        # Instantiate subclass, and update using current object.
        # Modality is updated here to reflect the choices made above.
        image = file_class()
        image.update_from_template(template=self)
        image.modality = modality

        # Set metadata of image.
        image.image_metadata = self.image_metadata
        image.is_limited_metadata = self.is_limited_metadata

        # Multi-frame images need additional work.
        if isinstance(image, ImageDicomMultiFrame):
            image = image.create()

        return image

    def update_from_template(self, template: Self):
        if not isinstance(template, ImageDicomFile):
            raise TypeError(
                f"The new class object should inherit from an ImageDicomFile object. Found: {type(template)}"
            )

        # Attributes from the template Image Dicom File.
        self.file_path = deepcopy(template.file_path)
        self.dir_path = deepcopy(template.dir_path)
        self.sample_name = deepcopy(template.sample_name)
        self.file_name = deepcopy(template.file_name)
        self.modality = deepcopy(template.modality)
        self.image_name = deepcopy(template.image_name)
        self.file_type = deepcopy(template.file_type)
        self.image_data = deepcopy(template.image_data)
        self.image_origin = deepcopy(template.image_origin)
        self.image_orientation = deepcopy(template.image_orientation)
        self.image_spacing = deepcopy(template.image_spacing)
        self.image_dimension = deepcopy(template.image_dimension)
        self.frame_of_reference_uid = deepcopy(template.frame_of_reference_uid)
        self.image_metadata = deepcopy(template.image_metadata)
        self.is_limited_metadata = deepcopy(template.is_limited_metadata)
        self.series_instance_uid = deepcopy(template.series_instance_uid)
        self.sop_instance_uid = deepcopy(template.sop_instance_uid)
        self.associated_masks = template.associated_masks

    def complete(self, remove_metadata=False, force=False):

        # complete loads metadata.
        super().complete(remove_metadata=False, force=force)

        # Set SOP instance UID.
        if self.sop_instance_uid is None:
            self.load_metadata(limited=True)
            self.sop_instance_uid = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0008, 0x0018),
                tag_type="str"
            )

        # Set Frame of Reference UID (if any)
        self._complete_frame_of_reference_uid()

        # Set series UID
        if self.series_instance_uid is None:
            self.load_metadata(limited=True)
            self.series_instance_uid = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0020, 0x000E),
                tag_type="str"
            )

        if remove_metadata:
            self.remove_metadata()

    def _complete_modality(self):
        if self.modality is None:
            self.load_metadata(limited=True)
            self.modality = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x0060), tag_type="str")

        if self.modality is None:
            warn(f"Could not ascertain the modality from the DICOM file. [{self.describe_self()}]", UserWarning)

            self.modality = "generic"
        else:
            self.modality = self.modality.lower()

    def _complete_sample_name(self):
        # Set sample name -- note that if sample_name is a (single) string, it is neither replaced nor updated.
        if self.sample_name is None or isinstance(self.sample_name, list):

            # Load relevant metadata.
            self.load_metadata(limited=True)

            # Consider the following DICOM elements:
            # patient id, study id, patient name, series description and study description.
            # These are explicitly ordered by relevance for setting the sample name.
            dicom_sample_name = [
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0010, 0x0020), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0020, 0x0010), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0010, 0x0010), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x1030), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x103E), tag_type="str")
            ]

            dicom_sample_name = [
                current_dicom_sample_name for current_dicom_sample_name in dicom_sample_name
                if current_dicom_sample_name is not None
            ]

            if self.sample_name is None and len(dicom_sample_name) > 0:
                # Set sample name using the most relevant DICOM tag, if no sample names are present.
                self.sample_name = dicom_sample_name[0]

            elif len(self.sample_name) > 0 and len(dicom_sample_name) > 0:
                # Set sample name using the intersection of allowed sample names and DICOM-based names.
                matching_sample_names = set(self.sample_name).intersection(set(dicom_sample_name))
                if len(matching_sample_names) > 0:
                    self.sample_name = list(matching_sample_names)[0]
                else:
                    self.sample_name = None

    def _complete_image_origin(self, force=False, frame_id=None):
        if self.image_origin is None:
            # Origin needs to be determined at the stack-level for slice-based dicom, not for each slice.
            if self.modality in stacking_dicom_image_modalities() and not force:
                return

            # Load relevant metadata.
            self.load_metadata(limited=True)

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
            # Orientation needs to be determined at the stack-level for slice-based dicom, not for each slice.
            if self.modality in stacking_dicom_image_modalities() and not force:
                return

            # Load relevant metadata.
            self.load_metadata(limited=True)

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
            # Image spacing needs to be determined at the stack-level for slice-based dicom, not for each slice.
            if self.modality in stacking_dicom_image_modalities() and not force:
                return

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
            # Image dimension needs to be determined at the stack-level for slice-based dicom, not for each slice.
            if self.modality in stacking_dicom_image_modalities() and not force:
                return

            # Load relevant metadata.
            self.load_metadata(limited=True)

            dimensions = tuple([
                1,
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0010), tag_type="int"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x0011), tag_type="int")
            ])

            self.image_dimension = dimensions

    def _complete_frame_of_reference_uid(self):
        if self.frame_of_reference_uid is None:
            self.load_metadata(limited=True)
            self.frame_of_reference_uid = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0020, 0x0052),
                tag_type="str"
            )

    def associate_with_mask(
            self,
            mask_list,
            association_strategy=None):
        if mask_list is None or len(mask_list) == 0 or association_strategy is None:
            return None

        # Match on frame of reference UID:
        if "frame_of_reference" in association_strategy and self.frame_of_reference_uid is not None:
            matching_mask_list = [
                mask_file for mask_file in mask_list
                if self.frame_of_reference_uid == mask_file.frame_of_reference_uid
            ]

            if len(matching_mask_list) > 0:
                self.associated_masks = matching_mask_list
                return

        return super().associate_with_mask(mask_list=mask_list, association_strategy=association_strategy)

    def load_metadata(self, limited=False, include_image=False):
        if include_image:
            limited = False

        # Limited metadata exists and limited metadata is sufficient.
        if self.image_metadata is not None and self.is_limited_metadata and limited:
            return

        # A full image metadata set exists.
        if self.image_metadata is not None and not self.is_limited_metadata:
            return

        if self.file_path is None or not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The image file could not be found at the expected location: {self.file_path}. "
                f"[{self.describe_self()}]")

        if limited:
            dcm = dcmread(
                self.file_path,
                stop_before_pixels=True,
                force=True,
                specific_tags=self._get_limited_metadata_tags()
            )
        else:
            dcm = dcmread(
                self.file_path,
                stop_before_pixels=not include_image,
                force=True
            )

        self.image_metadata = dcm
        self.is_limited_metadata = limited

    def load_data(self, **kwargs):
        raise NotImplementedError(
            "DEV: The load_data method should be specified for subclasses of ImageDicomFile. A generic method does not "
            "exist."
        )

    def load_data_generic(self) -> np.ndarray:
        """
        This is the generic method for loading pixel data from DICOM images that is shared across many modalities.
        :return:
        """
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

        # Update data with scale and intercept. These may change per slice.
        rescale_intercept = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x1052), tag_type="float", default=0.0)
        rescale_slope = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x1053), tag_type="float", default=1.0)
        image_data = image_data * rescale_slope + rescale_intercept

        return image_data

    def set_object_metadata(self):
        """
        Updates the object metadata that is passed to native image and mask classes in to_object.
        """
        metadata = []
        super().set_object_metadata()

        # Ensure that metadata are present.
        self.load_metadata(limited=False)

        # Study date
        study_date = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x0020),
            tag_type="str"
        )

        if study_date is not None and not study_date == "":
            metadata += [("study_date", study_date)]

        # Study description
        study_description = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x1030),
            tag_type="str"
        )

        if study_description is not None and not study_description == "":
            metadata += [("study_description", study_description)]

        # Series description
        series_description = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x103E),
            tag_type="str"
        )

        if series_description is not None and not series_description == "":
            metadata += [("series_description", series_description)]

        # Series instance UID
        if self.series_instance_uid is not None:
            metadata += [("series_instance_uid", self.series_instance_uid)]

        # Update object_metadata
        self.object_metadata.update(dict(metadata))

    def _check_is_mr_adc(self):
        # Check for ADC images. ADC can sometimes by identified the ADC value in the Image Type (0008,0008) tag,
        # and otherwise through the diffusion b-value in (0018,9087).
        image_type = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x0008), tag_type="mult_str")
        if image_type is not None and any(x.lower() == "adc" for x in image_type):
            return True
        elif get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0018, 0x9087), tag_type="float"):
            return True

        return False

    @staticmethod
    def _get_limited_metadata_tags():
        # Limited tags are read to populate basic
        return [
            (0x0008, 0x0008),  # image type
            (0x0008, 0x0018),  # SOP instance UID
            (0x0008, 0x0060),  # modality
            (0x0008, 0x1030),  # study description
            (0x0008, 0x103E),  # series description
            (0x0010, 0x0010),  # patient name
            (0x0010, 0x0020),  # patient id
            (0x0018, 0x0050),  # slice thickness
            (0x0018, 0x9087),  # diffusion b-value
            (0x0020, 0x000E),  # series instance UID
            (0x0020, 0x0010),  # study id
            (0x0020, 0x0032),  # origin
            (0x0020, 0x0037),  # orientation
            (0x0020, 0x0052),  # frame of reference UID
            (0x0028, 0x0008),  # number of frames
            (0x0028, 0x0010),  # pixel rows
            (0x0028, 0x0011),  # pixel columns
            (0x0028, 0x0030),  # pixel spacing
            (0x3004, 0x000C),  # grid frame offset vector
            (0x5200, 0x9229),  # shared functional groups sequence
            (0x5200, 0x9230)   # per-frame functional groups sequence
        ]

    def _get_acquisition_start_time(self) -> datetime.datetime:
        self.load_metadata()

        # Start of image acquisition. Prefer Acquisition Datetime (0x0008, 0x002A).
        acquisition_ref_time = get_pydicom_meta_tag(
            dcm_seq=self.image_metadata,
            tag=(0x0008, 0x002A),
            tag_type="str"
        )
        acquisition_ref_time = convert_dicom_time(datetime_str=acquisition_ref_time)

        # Fall back to Acquisition Date (0x0008, 0x002A) and Acquisition Time (0x0008, 0x0032).
        if acquisition_ref_time is None:
            acquisition_start_date = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0008, 0x0022),
                tag_type="str"
            )
            acquisition_start_time = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0008, 0x0032),
                tag_type="str"
            )
            acquisition_ref_time = convert_dicom_time(
                date_str=acquisition_start_date,
                time_str=acquisition_start_time
            )

        # Fall back to Private GE Acquisition DateTime (0x0009, 0x100d).
        if acquisition_ref_time is None:
            acquisition_ref_time = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0009, 0x100d),
                tag_type="str"
            )
            acquisition_ref_time = convert_dicom_time(datetime_str=acquisition_ref_time)

        # Fall back to Series Date and Series Time (
        if acquisition_ref_time is None:
            acquisition_start_date = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0008, 0x0021),
                tag_type="str"
            )
            acquisition_start_time = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0008, 0x0031),
                tag_type="str"
            )
            acquisition_ref_time = convert_dicom_time(
                date_str=acquisition_start_date,
                time_str=acquisition_start_time
            )

        # Final check.
        if acquisition_ref_time is None:
            raise ValueError(
                f"Acquisition start time cannot be determined from DICOM metadata."
            )

        return acquisition_ref_time

    def _get_export_attributes(self) -> dict[str, Any]:
        attributes = []
        self.load_metadata()

        study_description = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x1030), tag_type="str")
        if study_description is not None and study_description != "":
            attributes += [("study_description", study_description)]

        series_description = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x103E), tag_type="str")
        if series_description is not None and series_description != "":
            attributes += [("series_description", series_description)]

        # Try to find the acquisition time (which may be equal to series time).
        try:
            acquisition_time = self._get_acquisition_start_time()
        except ValueError:
            acquisition_time = None
        if acquisition_time is not None:
            attributes += [("acquisition_time", acquisition_time)]

        if self.series_instance_uid is not None:
            attributes += [("series_instance_uid", self.series_instance_uid)]
        if self.frame_of_reference_uid is not None:
            attributes += [("frame_of_reference_uid", self.frame_of_reference_uid)]

        parent_attributes = super()._get_export_attributes()
        parent_attributes.update(dict(attributes))

        return parent_attributes


class MaskDicomFile(ImageDicomFile, MaskFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _check_modality(self, raise_error: bool) -> bool:
        dicom_modality = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x0060), tag_type="str")
        support_modalities = supported_mask_modalities(self.modality)
        if dicom_modality.lower() not in support_modalities:
            if raise_error:
                raise ValueError(
                    f"The current DICOM file does not have the expected modality. "
                    f"Found: {dicom_modality.lower()}. "
                    f"Expected: {', '.join(support_modalities)}. "
                    f"[{self.describe_self()}]"
                )

            return False

        return True

    def load_data(self, **kwargs):
        raise NotImplementedError(
            "DEV: The load_data method should be specified for subclasses of MaskDicomFile. A generic method does not "
            "exist."
        )

    def create(self):
        """
        DICOM-files have different routines depending on the modality. These are then dispatched to different classes
        using this method.
        :return: an object of a subclass of ImageDicomFile.
        """

        # Import locally to prevent circular references.
        from mirp._data_import.dicom_file_rtstruct import MaskDicomFileRTSTRUCT
        from mirp._data_import.dicom_file_seg import MaskDicomFileSEG

        # Load metadata so that the modality tag can be read.
        self.load_metadata(limited=True)

        modality = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x0060), tag_type="str").lower()

        if modality is None:
            raise TypeError(f"Modality attribute could not be obtained from the DICOM file. [{self.describe_self()}]")

        if modality == "rtstruct":
            file_class = MaskDicomFileRTSTRUCT
        elif modality == "seg":
            file_class = MaskDicomFileSEG
        else:
            return None

        mask = file_class(
            file_path=self.file_path,
            dir_path=self.dir_path,
            sample_name=self.sample_name,
            file_name=self.file_name,
            image_modality=self.modality,
            image_name=self.image_name,
            image_file_type=self.file_type,
            image_data=self.image_data,
            image_origin=self.image_origin,
            image_orientation=self.image_orientation,
            image_spacing=self.image_spacing,
            image_dimensions=self.image_dimension,
            roi_name=self.roi_name
        )

        # Set metadata of mask.
        mask.image_metadata = self.image_metadata
        mask.is_limited_metadata = self.is_limited_metadata

        return mask

    def _get_limited_metadata_tags(self):
        tags = super()._get_limited_metadata_tags()

        tags += [
            (0x3006, 0x0020),  # Structure set roi sequence
            (0x0028, 0x0008)  # number of frames
        ]
