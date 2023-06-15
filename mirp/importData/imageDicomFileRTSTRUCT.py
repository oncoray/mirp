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

    def load_data(
            self,
            associated_image: Union[None, ImageFile] = None,
            roi_name: Union[None, str, List[str]] = None,
            **kwargs):
        if self.image_data is not None:
            return

        if self.file_path is not None and not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The image file could not be found at the expected location: {self.file_path}"
            )

        if self.file_path is None:
            raise ValueError(f"A path to a file was expected, but not present.")

        if not isinstance(associated_image, ImageFile):
            raise TypeError(f"Processing RTSTRUCT files requires the associated image file to be set.")

        if isinstance(roi_name, str):
            roi_name = [roi_name]

        # Load metadata.
        self.load_metadata()

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

        # Update roi name in case it is missing.
        for ii, current_roi_name in enumerate(roi_name_present):
            if current_roi_name is None:
                roi_name_present[ii] = "region_" + str(roi_number_present[ii])

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

    def to_object(self, **kwargs):

        from mirp.imageClass import ImageClass
        from mirp.roiClass import RoiClass

        self.load_data()
        self.complete()
        self.update_image_data()

        return RoiClass(
            name=self.roi_name,
            contour=None,
            roi_mask=ImageClass(
                voxel_grid=self.image_data,
                origin=self.image_origin,
                spacing=self.image_spacing,
                orientation=self.image_orientation,
                modality=self.modality
            )
        )