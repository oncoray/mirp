import copy
import itertools
import warnings
from typing import Union, List

import numpy as np
import pandas as pd

from mirp.importData.imageGenericFile import ImageFile
from mirp.importData.imageDicomFile import ImageDicomFile
from mirp.importData.imageITKFile import ImageITKFile
from mirp.importData.imageNumpyFile import ImageNumpyFile
from mirp.importData.utilities import supported_file_types


class ImageFileStack(ImageFile):
    def is_stackable(self, stack_images: str):
        return False

    def __init__(
            self,
            image_file_objects: Union[List[ImageFile], List[ImageDicomFile], List[ImageITKFile], List[ImageNumpyFile]],
            dir_path: Union[None, str] = None,
            sample_name: Union[None, str] = None,
            image_name: Union[None, str, List[str]] = None,
            image_modality: Union[None, str] = None,
            image_file_type: Union[None, str] = None,
            **kwargs):

        if dir_path is None:
            dir_path = image_file_objects[0].dir_path

        if sample_name is None:
            sample_name = image_file_objects[0].sample_name

        if image_name is None:
            image_name = image_file_objects[0].image_name

        if image_modality is None:
            image_modality = image_file_objects[0].modality

        if image_file_type is None:
            image_file_type = image_file_objects[0].file_type

        if len(image_file_objects) == 1:
            raise ValueError(f"DEV: More than one file is expected for file stacks.")

        # Aspects regarding the image itself are set based on the stack itself.
        super().__init__(
            file_path=None,
            dir_path=dir_path,
            sample_name=sample_name,
            file_name=None,
            image_name=image_name,
            image_modality=image_modality,
            image_file_type=image_file_type,
            image_data=None,
            image_origin=None,
            image_orientation=None,
            image_spacing=None,
            image_dimensions=None
        )

        self.image_file_objects = image_file_objects
        self.slice_positions: Union[None, List[float]] = None

    def create(self):
        # Import locally to avoid potential circular references.
        from mirp.importData.imageDicomFileStack import ImageDicomFileStack
        from mirp.importData.imageITKFileStack import ImageITKFileStack
        from mirp.importData.imageNumpyFileStack import ImageNumpyFileStack

        if all(isinstance(image_file_object, ImageDicomFile) for image_file_object in self.image_file_objects):
            file_stack_class = ImageDicomFileStack
            file_type = "dicom"

        elif all(isinstance(image_file_object, ImageITKFile) for image_file_object in self.image_file_objects):
            file_stack_class = ImageITKFileStack
            file_type = self.image_file_objects[0].file_type

        elif all(isinstance(image_file_object, ImageNumpyFile) for image_file_object in self.image_file_objects):
            file_stack_class = ImageNumpyFileStack
            file_type = "numpy"

        else:
            raise TypeError(f"The list of image objects does not consist of a known object type.")

        image_file_stack = file_stack_class(
            image_file_objects=self.image_file_objects,
            dir_path=self.dir_path,
            sample_name=self.sample_name,
            image_name=self.image_name,
            image_modality=self.modality,
            image_file_type=file_type
        )

        return image_file_stack

    def complete(self, remove_metadata=True, force=False):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of complete. Please specify "
            f"implementation for subclasses."
        )

    def _complete_sample_name(self):
        if self.sample_name is None:
            image_object = copy.deepcopy(self.image_file_objects[0])
            image_object._complete_sample_name()

            self.sample_name = image_object.sample_name

    def _complete_modality(self):
        if self.modality is None:
            image_object = copy.deepcopy(self.image_file_objects[0])
            image_object._complete_modality()

            self.modality = image_object.modality

    def _complete_image_origin(self, force=False):
        # Image origin and other image-related aspects are set using the complete method of subclasses.
        pass

    def _complete_image_orientation(self, force=False):
        pass

    def _complete_image_spacing(self, force=False):
        pass

    def _complete_image_dimensions(self, force=False):
        pass

    def sort_image_objects_by_file(self):
        """
        Strip sample name and any image name from filenames. Then isolate numeric values.
        sequences of numeric values. We follow the following rules:
        1. Check if all files have a numeric value in their name, otherwise, use the original order.
        2. Check that all files only have a single range of numeric values (otherwise, it might hard to arrange and
        identify sequences).
        3. Sort and check that sequences are truly sequential, i.e. have a difference of one.
        :return: nothing, changes are made in-place.
        """

        file_name_numeric = [image_object._get_numeric_sequence_from_file() for image_object in self.image_file_objects]
        if any(current_file_name_numeric is None for current_file_name_numeric in file_name_numeric):
            warnings.warn(
                f"Cannot form stacks from numpy slices based on the file name as numeric values are missing "
                "from one or more files. The original file order is used.", UserWarning
            )
            return

        if any(len(current_file_name_numeric) > 1 for current_file_name_numeric in file_name_numeric):
            warnings.warn(
                f"Cannot form stacks from numpy slices based on the file name as more than one sequence of numeric "
                f"values are present in the name of one or more files. This excludes the sample name (if known) and "
                f"any identifiers for image data. The original file order is used.", UserWarning
            )
            return

        # Flatten array and convert to integer values.
        file_name_numeric = list(itertools.chain.from_iterable(file_name_numeric))
        file_name_numeric = [int(current_file_name_numeric) for current_file_name_numeric in file_name_numeric]

        if len(file_name_numeric) == 1:
            return

        # Check that all numbers are sequential.
        if not np.all(np.diff(np.sort(np.array(file_name_numeric))) == 1):
            warnings.warn(
                f"Cannot form stacks from numpy slices based on the file name as numbers are not fully sequential for"
                f" all files. The original file order is used.", UserWarning
            )
            return

        position_table = pd.DataFrame({
            "original_object_order": list(range(len(self.image_file_objects))),
            "order_id": file_name_numeric,
        }).sort_values(by=["order_id"])

        # Sort image file objects.
        self.image_file_objects = [
            self.image_file_objects[position_table.original_object_order[ii]]
            for ii in range(len(position_table))
        ]

    def load_metadata(self):
        # Load metadata for underlying files in the order indicated by self.image_file_objects.
        for image_file_object in self.image_file_objects:
            image_file_object.load_metadata()

    def load_data(self, **kwargs):
        # Load data for underlying files in the order indicated by self.image_file_objects.
        for image_file_object in self.image_file_objects:
            image_file_object.load_data()
