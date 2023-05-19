import itk
import os.path
import numpy as np

from typing import Union, List, Tuple

from mirp.importData.imageGenericFile import ImageFile


class ImageITKFile(ImageFile):
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
            image_origin: Union[None, Tuple[float]] = None,
            image_orientation: Union[None, np.ndarray] = None,
            image_spacing: Union[None, Tuple[float]] = None,
            image_dimensions: Union[None, Tuple[int]] = None,
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
            image_dimensions=image_dimensions
        )

    def is_stackable(self, stack_images: str) -> bool:
        """
        Is the image potentially stackable?
        :param stack_images: One of auto, yes or no. By default (auto), images are not stackable. Images might be
        stackable if an image object represents a single slice.
        :return: boolean value
        """
        if stack_images == "auto":
            return False
        elif stack_images == "yes":
            if self.image_dimension is None:
                raise ValueError(
                    "The image_dimension argument is expected to be set. Call load_metadata to set this attribute."
                )

            if len(self.image_dimension) < 3:
                return True
            elif self.image_dimension[0] == 1:
                return True
            else:
                return False

        elif stack_images == "no":
            return False
        else:
            raise ValueError(
                f"The stack_images argument is expected to be one of yes, auto, or no. Found: {stack_images}."
            )

    def _complete_image_origin(self):
        if self.image_orientation is None:
            origin = np.array(self.image_metadata.GetOrigin())[::-1]
            self.image_origin = tuple(origin)

    def _complete_image_orientation(self):
        if self.image_orientation is None:
            orientation = np.reshape(np.ravel(itk.array_from_matrix(self.image_metadata.GetDirection()))[::-1], [3, 3])
            self.image_orientation = orientation

    def _complete_image_spacing(self):
        if self.image_spacing is None:
            spacing = np.array(self.image_metadata.GetSpacing())[::-1]
            self.image_spacing = tuple(spacing)

    def _complete_image_dimensions(self):
        if self.image_dimension is None:
            dimensions = np.array(self.image_metadata.GetSize())[::-1]
            self.image_dimension = tuple(dimensions)

    def load_metadata(self):
        if self.image_metadata is not None:
            pass

        if self.file_path is None or not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The image file could not be found at the expected location: {self.file_path}")

        # Default reader is from itk.
        reader = itk.ImageFileReader()
        reader.SetFileName(self.file_path)
        reader.ReadImageInformation()

        self.image_metadata = reader

    def load_data(self):
        if self.image_data is not None:
            pass

        if self.file_path is None or not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The image file could not be found at the expected location: {self.file_path}")

        # Load the image
        itk_img = itk.imread(os.path.join(self.file_path))
        self.image_data = itk.GetArrayFromImage(itk_img).astype(np.float32)
