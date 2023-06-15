import os.path
import warnings

import numpy as np

from typing import Union, List
from mirp.importData.importImage import ImageFile
from mirp.importData.imageDicomFile import MaskDicomFile
from mirp.imageMetaData import get_pydicom_meta_tag, has_pydicom_meta_tag


class MaskDicomFileRTSTRUCT(MaskDicomFile):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # RTSTRUCT does not have its own image information.
        self.image_origin = None
        self.image_orientation = None
        self.image_spacing = None
        self.image_dimensions = None

    def is_stackable(self, stack_images: str):
        return False

    def create(self):
        return self

    def _complete_frame_of_reference_uid(self):
        if self.frame_of_reference_uid is None:
            # Try to obtain a frame of reference UID
            if has_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0020, 0x0052)):
                if get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x0020, 0x0052), tag_type="str") is not None:
                    self.frame_of_reference_uid = get_pydicom_meta_tag(
                        dcm_seq=self.image_metadata,
                        tag=(0x0020, 0x0052),
                        tag_type="str")

            # For RT structure sets, the FOR UID may be tucked away in the Structure Set ROI Sequence
            if self.frame_of_reference_uid is None and \
                    has_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x3006, 0x0020)):
                structure_set_roi_sequence = self.image_metadata[(0x3006, 0x0020)]

                for structure_set_roi_element in structure_set_roi_sequence:
                    if has_pydicom_meta_tag(dcm_seq=structure_set_roi_element, tag=(0x3006, 0x0024)):
                        self.frame_of_reference_uid = get_pydicom_meta_tag(
                            dcm_seq=structure_set_roi_element,
                            tag=(0x3006, 0x0024),
                            tag_type="str")
                        break

    def check_mask(self, raise_error=True):
        if self.image_metadata is None:
            raise TypeError("DEV: the image_meta_data attribute has not been set.")

        roi_name_present = [
            get_pydicom_meta_tag(
                dcm_seq=current_structure_set_roi_sequence, tag=(0x3006, 0x0026), tag_type="str", default=None)
            for current_structure_set_roi_sequence in self.image_metadata[(0x3006, 0x0020)]
        ]

        if len(roi_name_present) == 0 and raise_error:
            warnings.warn(
                f"The current RT-structure set ({self.file_path}) does not contain any ROI contours."
            )

        if any(x is None for x in roi_name_present) and raise_error:
            warnings.warn(
                f"The current RT-structure set ({self.file_path}) lacks one or more ROI names."
            )

        return True

    def load_data(self, **kwargs):
        pass

    def to_object(self, image: Union[None, ImageFile], **kwargs):

        from mirp.imageClass import ImageClass
        from mirp.roiClass import RoiClass
        if image is None:
            raise TypeError(
                f"Converting an RT-structure to a segmentation mask requires that the corresponding image is set. "
                f"No image was provided ({self.file_path})."
            )
        else:
            image.complete()

        self.load_metadata()
        self.check_mask()

        # Check if a Structure Set ROI Sequence exists (3006, 0020)
        if not get_pydicom_meta_tag(dcm_seq=self.image_metadata, tag=(0x3006, 0x0020), test_tag=True):
            warnings.warn(
                f"The RT-structure set ({self.file_path}) did not contain any ROI sequences.")
            return

        # Find which roi numbers (3006,0022) are associated with which roi names (3004,0024).
        roi_name_present = [
            get_pydicom_meta_tag(
                dcm_seq=current_structure_set_roi_sequence, tag=(0x3006, 0x0026), tag_type="str", default=None)
            for current_structure_set_roi_sequence in self.image_metadata[(0x3006, 0x0020)]
        ]
        roi_number_present = [
            get_pydicom_meta_tag(
                dcm_seq=current_structure_set_roi_sequence, tag=(0x3006, 0x0026), tag_type="str", default=None)
            for current_structure_set_roi_sequence in self.image_metadata[(0x3006, 0x0020)]
        ]

        # Identify which roi names (and numbers) should be kept by comparing against the roi_name attribute.
        for ii, current_roi_name in enumerate(roi_name_present):

            if current_roi_name is None and len(roi_name_present) == 1:
                if  isinstance(self.roi_name, str):
                    roi_name_present[ii] = self.roi_name
                elif


        # Initialise a data list
        contour_data_list = []

        # Obtain segmentation
        for roi_contour_sequence in self.image_metadata[(0x3006, 0x0039)]:
            for contour_sequence in roi_contour_sequence[(0x3006, 0x0040)]:

                # Check if the geometric type exists (3006, 0042)
                if not get_pydicom_meta_tag(dcm_seq=contour_sequence, tag=(0x3006, 0x0042), test_tag=True):
                    continue

                # Check if the geometric type equals "CLOSED_PLANAR"
                if get_pydicom_meta_tag(
                        dcm_seq=contour_sequence,
                        tag=(0x3006, 0x0042),
                        tag_type="str") != "CLOSED_PLANAR":
                    continue

                # Check if contour data exists (3006, 0050)
                if not get_pydicom_meta_tag(dcm_seq=contour_sequence, tag=(0x3006, 0x0050), test_tag=True):
                    continue

                contour_data = np.array(
                    get_pydicom_meta_tag(dcm_seq=contour_sequence, tag=(0x3006, 0x0050), tag_type="mult_float"),
                    dtype=np.float64)
                contour_data = contour_data.reshape((-1, 3))

                # Determine if there is an offset (3006, 0045)
                contour_offset = np.array(
                    get_pydicom_meta_tag(
                        dcm_seq=contour_sequence,
                        tag=(0x3006, 0x0045),
                        tag_type="mult_float",
                        default=[0.0, 0.0, 0.0]
                    ),
                    dtype=np.float64)

                # Remove the offset from the data
                contour_data -= contour_offset

                # Obtain the sop instance uid.
                if get_pydicom_meta_tag(dcm_seq=contour_sequence, tag=(0x3006, 0x0016), test_tag=True):
                    sop_instance_uid = get_pydicom_meta_tag(dcm_seq=contour_sequence[(0x3006, 0x0016)][0],
                                                            tag=(0x0008, 0x1155), tag_type="str", default=None)
                else:
                    sop_instance_uid = None

                # Store as contour.
                contour = ContourClass(contour=contour_data, sop_instance_uid=sop_instance_uid)

                # Add contour data to the contour list
                contour_data_list += [contour]

        if len(contour_data_list) > 0:
            # Create a new ROI object.
            roi_obj = RoiClass(name="+".join(deparsed_roi), contour=contour_data_list, metadata=dcm)

            # Convert contour into segmentation object
            roi_obj.create_mask_from_contours(img_obj=image_object,
                                              disconnected_segments="keep_as_is",
                                              settings=settings)

            return roi_obj