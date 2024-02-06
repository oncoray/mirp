import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any

from mirp.importData.imageDicomFile import ImageDicomFile
from mirp.importData.imageGenericFileStack import ImageFileStack, MaskFileStack
from mirp.importData.utilities import get_pydicom_meta_tag


class ImageDicomFileStack(ImageFileStack):

    def __init__(
            self,
            image_file_objects: List[ImageDicomFile],
            **kwargs
    ):
        super().__init__(image_file_objects, **kwargs)

        self.series_instance_uid: None | str = None
        self.frame_of_reference_uid: None | str = None

        # Placeholder for sop_instance_uid of objects contained in the stack.
        self.sop_instance_uid: None | list[str] = None

        # Add type hint.
        self.image_file_objects: list[ImageDicomFile] = self.image_file_objects

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

    def complete(self, remove_metadata=False, force=False):
        """
        Fills out missing attributes in an image stack. Image parameters in DICOM stacks, by design,
        are fully determined by the origin of all slices in the stack. This method then sorts the image file objects
        by origin, and uses their relative positions to determine slice spacing and the orientation vector.
        :param remove_metadata: Whether metadata (DICOM headers) should be removed after completing information.
        :param force: Whether attributes are forced to update or not.
        :return: nothing, attributes are updated in place.
        """

        # Load metadata of every slice.
        self.load_metadata()

        self._complete_modality()
        self._complete_sample_name()

        position_table = self._get_origin_position_table()

        # Sort image file objects.
        self.image_file_objects = [
            self.image_file_objects[position_table.original_object_order[ii]]
            for ii in range(len(position_table))
        ]

        if self.image_origin is None:
            # Set image origin.
            self.image_origin = tuple(get_pydicom_meta_tag(
                dcm_seq=self.image_file_objects[0].image_metadata,
                tag=(0x0020, 0x0032),
                tag_type="mult_float",
                default=np.array([0.0, 0.0, 0.0])
            )[::-1])

        # Set image spacing
        # Compute the distance between the origins of the slices. This is the slice spacing.
        image_slice_spacing = np.sqrt(
            np.power(np.diff(position_table.position_x.values), 2.0) +
            np.power(np.diff(position_table.position_y.values), 2.0) +
            np.power(np.diff(position_table.position_z.values), 2.0))

        # Find the smallest slice spacing.
        min_slice_spacing = np.min(image_slice_spacing)

        # Find how much other slices differ.
        image_slice_spacing_multiplier = image_slice_spacing / min_slice_spacing

        if np.any(image_slice_spacing_multiplier > 1.2):
            warnings.warn(
                f"Inconsistent distance between slice origins of subsequent slices: {np.unique(image_slice_spacing)}. "
                "Slices cannot be aligned correctly. This is likely due to missing slices. "
                "MIRP will attempt to interpolate the missing slices and their ROI masks for volumetric analysis.",
                UserWarning)

        # Determine image slice spacing.
        image_slice_spacing = np.around(np.mean(image_slice_spacing[image_slice_spacing_multiplier <= 1.2]), 5)

        # Warn the user if there is a mismatch between slice thickness and the actual slice spacing.
        image_slice_thickness = get_pydicom_meta_tag(
            dcm_seq=self.image_file_objects[0].image_metadata,
            tag=(0x0018, 0x0050),
            tag_type="float")

        if not np.around(image_slice_thickness - image_slice_spacing, decimals=3) == 0.0:
            warnings.warn(
                f"Mismatch between slice thickness ({image_slice_thickness}) and actual slice spacing"
                f" ({image_slice_spacing}). The actual slice spacing will be used.",
                UserWarning)

        image_pixel_spacing = get_pydicom_meta_tag(
            dcm_seq=self.image_file_objects[0].image_metadata,
            tag=(0x0028, 0x0030),
            tag_type="mult_float")

        # Set image spacing.
        if self.image_spacing is None:
            self.image_spacing = tuple([image_slice_spacing, image_pixel_spacing[1], image_pixel_spacing[0]])

        # Read orientation (Xx, Xy, Xz, Yx, Yy, Yz) from metadata.
        image_orientation = get_pydicom_meta_tag(
            dcm_seq=self.image_file_objects[0].image_metadata,
            tag=(0x0020, 0x0037),
            tag_type="mult_float")

        # Add (Zx, Zy, Zz)
        z_orientation = np.array([
            np.around(np.min(np.diff(position_table.position_x.values)), 5),
            np.around(np.min(np.diff(position_table.position_y.values)), 5),
            np.around(np.min(np.diff(position_table.position_z.values)), 5)
        ]) / image_slice_spacing

        image_orientation += list(z_orientation)

        # Revert to z, y, x order and reshape to matrix.
        if self.image_orientation is None:
            self.image_orientation = np.reshape(image_orientation[::-1], [3, 3])

        # Set image dimensions. First, find the number of rows (y) and columns (x) in the data set.
        n_x = get_pydicom_meta_tag(
            dcm_seq=self.image_file_objects[0].image_metadata,
            tag=(0x0028, 0x011),
            tag_type="int")
        n_y = get_pydicom_meta_tag(
            dcm_seq=self.image_file_objects[0].image_metadata,
            tag=(0x0028, 0x010),
            tag_type="int")

        if self.image_dimension is None:
            self.image_dimension = tuple([len(position_table), n_y, n_x])

        if self.sop_instance_uid is None:
            self.sop_instance_uid = [image.sop_instance_uid for image in self.image_file_objects]
        if self.frame_of_reference_uid is None:
            self.frame_of_reference_uid = self.image_file_objects[0].frame_of_reference_uid
        if self.series_instance_uid is None:
            self.series_instance_uid = self.image_file_objects[0].series_instance_uid

        # Check if the complete data passes verification.
        self.check(raise_error=True, remove_metadata=False)

        if remove_metadata:
            self.remove_metadata()

    def _get_origin_position_table(self) -> pd.DataFrame:
        # Placeholders for slice positions.
        image_position_z = [0.0] * len(self.image_file_objects)
        image_position_y = [0.0] * len(self.image_file_objects)
        image_position_x = [0.0] * len(self.image_file_objects)

        for ii, image_object in enumerate(self.image_file_objects):
            slice_origin = get_pydicom_meta_tag(
                dcm_seq=image_object.image_metadata,
                tag=(0x0020, 0x0032),
                tag_type="mult_float",
                default=np.array([0.0, 0.0, 0.0]))[::-1]

            image_position_z[ii] = slice_origin[0]
            image_position_y[ii] = slice_origin[1]
            image_position_x[ii] = slice_origin[2]

        # Order ascending position (DICOM: z increases from feet to head)
        return pd.DataFrame({
            "original_object_order": list(range(len(self.image_file_objects))),
            "position_z": image_position_z,
            "position_y": image_position_y,
            "position_x": image_position_x
        }).sort_values(by=["position_z", "position_y", "position_x"], ignore_index=True)

    def export_metadata(self) -> Optional[Dict[str, Any]]:
        metadata = super().export_metadata()
        additional_metadata = self.image_file_objects[0].export_metadata(only_self=True)
        metadata.update(additional_metadata)

        return metadata


class MaskDicomFileStack(ImageDicomFileStack, MaskFileStack):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
