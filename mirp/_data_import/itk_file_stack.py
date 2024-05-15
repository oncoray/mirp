import copy
import warnings
import itk
import numpy as np
import pandas as pd

from mirp._data_import.generic_file import MaskFile
from mirp._data_import.itk_file import ImageITKFile
from mirp._data_import.generic_file_stack import ImageFileStack, MaskFileStack


class ImageITKFileStack(ImageFileStack):

    def __init__(
            self,
            image_file_objects: list[ImageITKFile],
            **kwargs
    ):
        super().__init__(image_file_objects, **kwargs)

    def complete(self, remove_metadata=False, force=False):
        """
        Fills out missing attributes in an image stack. Image parameters in ITK stacks are fully determined by the
        origin of all slices in the stack. This method then sorts the image file objects
        by origin, and uses their relative positions to determine slice spacing and the orientation vector. ITK-image
        stacks, e.g. stacks of NIfTI or NRRD files should be very very rare, since such formats are specifically
        designed to address some of the weakness of the DICOM standard slice-based format.
        :param remove_metadata: Whether metadata should be removed after completing information.
        :param force: Whether attributes are forced to update or not.
        :return: nothing, attributes are updated in place.
        """
        # Load metadata of every slice.
        self.load_metadata()

        self._complete_modality()
        self._complete_sample_name()

        # Placeholders for slice positions.
        image_position_z = [0.0] * len(self.image_file_objects)
        image_position_y = [0.0] * len(self.image_file_objects)
        image_position_x = [0.0] * len(self.image_file_objects)

        for ii, image_object in enumerate(self.image_file_objects):
            slice_origin = np.array(image_object.image_metadata.GetOrigin())[::-1]

            image_position_z[ii] = slice_origin[0]
            image_position_y[ii] = slice_origin[1]
            image_position_x[ii] = slice_origin[2]

        # Order ascending position (DICOM: z increases from feet to head)
        position_table = pd.DataFrame({
            "original_object_order": list(range(len(self.image_file_objects))),
            "position_z": image_position_z,
            "position_y": image_position_y,
            "position_x": image_position_x
        }).sort_values(by=["position_z", "position_y", "position_x"])

        # Set image spacing. Compute the distance between the origins of the slices. This is the slice spacing.
        image_slice_spacing = np.sqrt(
            np.power(np.diff(position_table.position_x.values), 2.0) +
            np.power(np.diff(position_table.position_y.values), 2.0) +
            np.power(np.diff(position_table.position_z.values), 2.0))

        # Find the smallest slice spacing.
        min_slice_spacing = np.min(image_slice_spacing)
        if min_slice_spacing == 0.0:
            warnings.warn(
                "Images files contain overlapping origins. Attempting to sort image files by numeric name "
                f"patterns. [{self.describe_self()}]",
                UserWarning
            )
            self.sort_image_objects_by_file()

            image_object = copy.deepcopy(self.image_file_objects[0])
            image_object.complete()

            if self.image_origin is None:
                self.image_origin = image_object.image_origin

            if self.image_spacing is None:
                self.image_spacing = image_object.image_spacing

            if self.image_orientation is None:
                self.image_orientation = image_object.image_orientation

            if self.image_dimension is None:
                self.image_dimension = tuple([
                    len(self.image_file_objects), image_object.image_dimension[1], image_object.image_dimension[2]
                ])

        else:
            # Sort image file objects.
            self.image_file_objects = [
                self.image_file_objects[position_table.original_object_order[ii]]
                for ii in range(len(position_table))
            ]

        # Set image origin.
        if self.image_origin is None:
            self.image_origin = tuple(np.array(self.image_file_objects[0].image_metadata.GetOrigin())[::-1])

        # Find how much other slices differ.
        image_slice_spacing_multiplier = image_slice_spacing / min_slice_spacing

        if np.any(image_slice_spacing_multiplier > 1.2):
            warnings.warn(
                f"Inconsistent distance between slice origins of subsequent slices: {np.unique(image_slice_spacing)}. "
                f"Slices cannot be aligned correctly. This is likely due to missing slices. "
                f"MIRP will attempt to interpolate the missing slices and their ROI masks for volumetric analysis. "
                f"[{self.describe_self()}]",
                UserWarning
            )

            # Update slice positions.
            self.slice_positions = list(np.cumsum(np.insert(np.around(image_slice_spacing, 5), 0, 0.0)))

        # Determine image slice spacing.
        image_slice_spacing = np.around(np.mean(image_slice_spacing[image_slice_spacing_multiplier <= 1.2]), 5)

        # Set image spacing.
        image_spacing = np.array(self.image_file_objects[0].image_metadata.GetSpacing())[::-1]
        if self.image_spacing is None:
            if image_spacing[0] == image_slice_spacing:
                self.image_spacing = tuple(image_spacing)
            else:
                self.image_spacing = tuple([image_slice_spacing, image_spacing[1], image_spacing[2]])

        # Read orientation metadata.
        image_orientation = np.reshape(np.ravel(itk.array_from_matrix(
            self.image_file_objects[0].image_metadata.GetDirection()))[::-1], [3, 3])

        # Add (Zx, Zy, Zz)
        z_orientation = np.array([
            np.around(np.min(np.diff(position_table.position_z.values)), 5),
            np.around(np.min(np.diff(position_table.position_y.values)), 5),
            np.around(np.min(np.diff(position_table.position_x.values)), 5)
        ]) / image_slice_spacing

        # Replace z-orientation and set image_orientation.
        image_orientation[0, :] = z_orientation
        if self.image_orientation is None:
            self.image_orientation = image_orientation

        # Set dimension
        image_dimension = np.array(self.image_metadata.GetSize())[::-1]

        if self.image_dimension:
            self.image_dimension = tuple([len(position_table), image_dimension[1], image_dimension[2]])

        # Check if the complete data passes verification.
        self.check(raise_error=True, remove_metadata=False)


class MaskITKFileStack(ImageITKFileStack, MaskFileStack):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def complete(self, remove_metadata=False, force=False):

        super().complete(remove_metadata=False, force=force)

        image_object: MaskFile = copy.deepcopy(self.image_file_objects[0])
        image_object.complete(remove_metadata=False)

        self.roi_name = image_object.roi_name
