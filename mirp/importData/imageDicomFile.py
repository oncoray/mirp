from typing import Union, List

from pydicom import dcmread
from warnings import warn

from mirp.importData.imageGenericFile import ImageFile
from mirp.importData.utilities import supported_image_modalities
from mirp.imageMetaData import get_pydicom_meta_tag


class ImageDicomFile(ImageFile):
    def __init__(
            self,
            file_path: Union[None, str] = None,
            dir_path: Union[None, str] = None,
            sample_name: Union[None, str, List[str]] = None,
            file_name: Union[None, str] = None,
            image_name: Union[None, str] = None,
            modality: Union[None, str] = None,
            image_file_type: Union[None, str] = None):

        super().__init__(
            file_path=file_path,
            dir_path=dir_path,
            sample_name=sample_name,
            file_name=file_name,
            image_name=image_name,
            image_modality=modality,
            image_file_type=image_file_type)

        # These are set using the 'complete' method.
        self.series_instance_uid: Union[None, str] = None
        self.frame_of_reference_uid: Union[None, str] = None
        self.sop_instance_uid: Union[None, str] = None

    def check(self, raise_error=False):

        # Perform general checks.
        if not super().check(raise_error=raise_error):
            return False

        # Read DICOM file.
        dcm = dcmread(
            self.file_path,
            stop_before_pixels=True,
            force=True)

        # Check that modality is matching.
        dicom_modality = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str")
        support_modalities = supported_image_modalities(self.modality)
        if dicom_modality.lower() not in support_modalities:
            if raise_error:
                raise ValueError(
                    f"The current DICOM file {self.file_path} does not have the expected modality. "
                    f"Found: {dicom_modality.lower()}. Expected: {', '.join(support_modalities)}")

            return False

        # Check sample name.
        if self.sample_name is not None:

            allowed_sample_name = self.sample_name
            if not isinstance(allowed_sample_name, list):
                allowed_sample_name = [allowed_sample_name]

            # Consider the following DICOM elements: study description, series description, patient name,
            # patient id, study id.
            dicom_sample_name = [
                get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x1030), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x103E), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0010, 0x0010), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0010, 0x0020), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x0010), tag_type="str")
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

    def complete(self):

        # Read DICOM file.
        dcm = dcmread(
            self.file_path,
            stop_before_pixels=True,
            force=True)

        # Attempt to set the modality attribute.
        if self.modality is None:
            self.modality = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str")

        if self.modality is None:
            warn(f"Could not ascertain the modality from the DICOM file ({self.file_path}).", UserWarning)

            self.modality = "generic"
        else:
            self.modality = self.modality.lower()

        # Set SOP instance UID.
        self.sop_instance_uid = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0018), tag_type="str")

        # Set Frame of Reference UID (if any)
        self.frame_of_reference_uid = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x0052), tag_type="str")

        # Set series UID
        self.series_instance_uid = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x000E), tag_type="str")

        # Set sample name -- note that if sample_name is a (single) string, it is neither replaced nor updated.
        if self.sample_name is None or isinstance(self.sample_name, list):
            # Consider the following DICOM elements:
            # patient id, study id, patient name, series description and study description.
            # These are explicitly ordered by relevance for setting the sample name.
            dicom_sample_name = [
                get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0010, 0x0020), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x0010), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0010, 0x0010), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x1030), tag_type="str"),
                get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x103E), tag_type="str")
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
