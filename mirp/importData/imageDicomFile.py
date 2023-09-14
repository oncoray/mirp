import os.path
import hashlib
import numpy as np

from typing import Union, List, Tuple, Dict

from pydicom import dcmread
from warnings import warn

from mirp.importData.imageGenericFile import ImageFile, MaskFile
from mirp.importData.utilities import supported_image_modalities, stacking_dicom_image_modalities, \
    supported_mask_modalities, get_pydicom_meta_tag


class ImageDicomFile(ImageFile):
    def __init__(
            self,
            file_path: Union[None, str] = None,
            dir_path: Union[None, str] = None,
            sample_name: Union[None, str, List[str]] = None,
            file_name: Union[None, str] = None,
            image_name: Union[None, str] = None,
            image_modality: Union[None, str] = None,
            image_file_type: Union[None, str] = None,
            image_data: Union[None, np.ndarray] = None,
            image_origin: Union[None, Tuple[float, ...]] = None,
            image_orientation: Union[None, np.ndarray] = None,
            image_spacing: Union[None, Tuple[float, ...]] = None,
            image_dimensions: Union[None, Tuple[int, ...]] = None,
            **kwargs):

        super().__init__(
            file_path=file_path,
            dir_path=dir_path,
            sample_name=sample_name,
            file_name=file_name,
            image_name=image_name,
            image_modality=image_modality,
            image_file_type=image_file_type,
            image_data=image_data,
            image_origin=image_origin,
            image_orientation=image_orientation,
            image_spacing=image_spacing,
            image_dimensions=image_dimensions,
            **kwargs
        )

        # These are set using the 'complete' method.
        self.series_instance_uid: Union[None, str] = None
        self.frame_of_reference_uid: Union[None, str] = None
        self.sop_instance_uid: Union[None, str] = None

    def is_stackable(self, stack_images: str):
        """
        Is the image potentially stackable?
        :param stack_images: One of auto, yes or no. Ignored for DICOM imaging, as stackability can be determined
        from metadata.
        :return: boolean value, here true.
        """
        return True

    def get_identifiers(self, as_hash=False) -> Union[Dict, bytes]:
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
        self.load_metadata()

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
        support_modalities = supported_image_modalities(self.modality)
        if dicom_modality.lower() not in support_modalities:
            if raise_error:
                raise ValueError(
                    f"The current DICOM file {self.file_path} does not have the expected modality. "
                    f"Found: {dicom_modality.lower()}. Expected: {', '.join(support_modalities)}")

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
                            f"The current DICOM file {self.file_path} does not have a matching sample name among "
                            f"potential identifiers. Potential identifiers: {', '.join(dicom_sample_name)}; "
                            f"Expected identifiers: {', '.join(allowed_sample_name)}."
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
        from mirp.importData.imageDicomFileCT import ImageDicomFileCT
        from mirp.importData.imageDicomFileMR import ImageDicomFileMR
        from mirp.importData.imageDicomFilePT import ImageDicomFilePT

        # Load metadata so that the modality tag can be read.
        self.load_metadata()

        modality = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x0060), tag_type="str").lower()

        if modality is None:
            raise TypeError(f"Modality attribute could not be obtained from DICOM file ({self.file_path})")

        if modality == "ct":
            file_class = ImageDicomFileCT
        elif modality == "pt":
            file_class = ImageDicomFilePT
        elif modality == "mr":
            file_class = ImageDicomFileMR
        else:
            # This will return a base class, which will fail to pass the modality check.
            return self

        return file_class(
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
            image_dimensions=self.image_dimension
        )

    def complete(self, remove_metadata=False, force=False):

        # complete loads metadata.
        super().complete(remove_metadata=False, force=force)

        # Set SOP instance UID.
        self.sop_instance_uid = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x0018), tag_type="str")

        # Set Frame of Reference UID (if any)
        self._complete_frame_of_reference_uid()

        # Set series UID
        self.series_instance_uid = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0020, 0x000E), tag_type="str")

        if remove_metadata:
            self.remove_metadata()

    def _complete_modality(self):
        if self.modality is None:
            self.modality = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x0060), tag_type="str")

        if self.modality is None:
            warn(f"Could not ascertain the modality from the DICOM file ({self.file_path}).", UserWarning)

            self.modality = "generic"
        else:
            self.modality = self.modality.lower()

    def _complete_sample_name(self):
        # Set sample name -- note that if sample_name is a (single) string, it is neither replaced nor updated.
        if self.sample_name is None or isinstance(self.sample_name, list):
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

    def _complete_image_origin(self, force=False):
        if self.image_origin is None:
            # Origin needs to be determined at the stack-level for slice-based dicom, not for each slice.
            if self.modality in stacking_dicom_image_modalities() and not force:
                return

            origin = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0020, 0x0032), tag_type="mult_float")[::-1]
            self.image_origin = tuple(origin)

    def _complete_image_orientation(self, force=False):
        if self.image_orientation is None:
            # Origin needs to be determined at the stack-level for slice-based dicom, not for each slice.
            if self.modality in stacking_dicom_image_modalities() and not force:
                return

            orientation = get_pydicom_meta_tag(
                dcm_seq=self.image_orientation,
                tag=(0x0020, 0x0037),
                tag_type="mult_float")
            orientation += [0.0, 0.0, 1.0]

            self.image_orientation = np.reshape(orientation[::-1], [3, 3])

    def _complete_image_spacing(self, force=False):
        if self.image_spacing is None:
            # Image spacing needs to be determined at the stack-level for slice-based dicom, not for each slice.
            if self.modality in stacking_dicom_image_modalities() and not force:
                return

            # Get pixel-spacing.
            spacing = get_pydicom_meta_tag(dcm_seq=self.image_spacing, tag=(0x0028, 0x0030), tag_type="mult_float")

            # Get slice thickness.
            z_spacing = get_pydicom_meta_tag(
                dcm_seq=self.image_metadata,
                tag=(0x0018, 0x0050),
                tag_type="float",
                default=1.0)
            spacing += [z_spacing]

            self.image_spacing = tuple(spacing[::-1])

    def _complete_image_dimensions(self, force=False):
        if self.image_dimension is None:
            # Image dimension needs to be determined at the stack-level for slice-based dicom, not for each slice.
            if self.modality in stacking_dicom_image_modalities() and not force:
                return

            dimensions = tuple([
                1,
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x010), tag_type="int"),
                get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0028, 0x011), tag_type="int")
            ])

            self.image_dimension = dimensions

    def _complete_frame_of_reference_uid(self):
        if self.frame_of_reference_uid is None:
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

    def load_metadata(self):
        if self.image_metadata is not None:
            return

        if self.file_path is None or not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The image file could not be found at the expected location: {self.file_path}")

        dcm = dcmread(
            self.file_path,
            stop_before_pixels=True,
            force=True)

        self.image_metadata = dcm

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
                f"The image file could not be found at the expected location: {self.file_path}"
            )

        if self.file_path is None:
            raise ValueError(f"A path to a file was expected, but not present.")

        # Load metadata.
        self.load_metadata()

        # Read
        dcm = dcmread(self.file_path, stop_before_pixels=False, force=True)
        image_data = dcm.pixel_array.astype(np.float32)

        # Update data with scale and intercept. These may change per slice.
        rescale_intercept = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0028, 0x1052), tag_type="float", default=0.0)
        rescale_slope = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0028, 0x1053), tag_type="float", default=1.0)
        image_data = image_data * rescale_slope + rescale_intercept

        return image_data


class MaskDicomFile(ImageDicomFile, MaskFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _check_modality(self, raise_error: bool) -> bool:
        dicom_modality = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x0060), tag_type="str")
        support_modalities = supported_mask_modalities(self.modality)
        if dicom_modality.lower() not in support_modalities:
            if raise_error:
                raise ValueError(
                    f"The current DICOM file {self.file_path} does not have the expected modality. "
                    f"Found: {dicom_modality.lower()}. Expected: {', '.join(support_modalities)}")

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
        from mirp.importData.imageDicomFileRTSTRUCT import MaskDicomFileRTSTRUCT
        from mirp.importData.imageDicomFileSEG import MaskDicomFileSEG

        # Load metadata so that the modality tag can be read.
        self.load_metadata()

        modality = get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0008, 0x0060), tag_type="str").lower()

        if modality is None:
            raise TypeError(f"Modality attribute could not be obtained from DICOM file ({self.file_path})")

        if modality == "rtstruct":
            file_class = MaskDicomFileRTSTRUCT
        elif modality == "seg":
            file_class = MaskDicomFileSEG
        else:
            # This will return a base class, which will fail to pass the modality check.
            return self

        return file_class(
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
