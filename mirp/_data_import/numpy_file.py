import os.path
import numpy as np

from mirp._data_import.generic_file import ImageFile, MaskFile


class ImageNumpyFile(ImageFile):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_stackable(self, stack_images: str):
        """
        Is the image potentially stackable?
        :param stack_images: One of auto, yes or no. By default (auto), images are stackable if an image object
        represents a single slice.
        :return: boolean value.
        """
        if stack_images in ["auto", "yes"]:
            if self.image_dimension is None:
                raise ValueError(
                    f"The image_dimension argument is expected to be set. Call load_metadata to set this attribute. "
                    f"[{self.describe_self()}]"
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
                f"The stack_images argument is expected to be one of yes, auto, or no. Found: {stack_images}. "
                f"[{self.describe_self()}]"
            )

    def check(self, raise_error=False, remove_metadata=True) -> bool:

        # Perform general checks.
        if not super().check(raise_error=raise_error):
            return False

        # Read Numpy file.
        image_data = np.load(file=self.file_path, mmap_mode="r")

        # Check that the contents are in fact a ndarray.
        if not isinstance(image_data, np.ndarray):
            if raise_error:
                raise TypeError(
                    f"The current numpy dataset {self.file_path} does not have the expected content. "
                    f"Found: {type(image_data)}. Expected: numpy.ndarray. [{self.describe_self()}]"
                )

            return False

        # Check dimensions.
        if not 0 < image_data.ndim <= 3:
            if raise_error:
                raise ValueError(
                    f"The current numpy dataset has as an unexpected number of dimensions: "
                    f"Found: {image_data.ndim}. Expected: 1, 2, or 3 dimensions. [{self.describe_self()}]"
                )

            return False

        return True

    def _complete_image_origin(self, force=False):
        if self.image_origin is None:
            self.load_metadata()
            n_dim = len(self.image_metadata.shape)
            self.image_origin = tuple([0.0] * n_dim)

    def _complete_image_orientation(self, force=False):
        if self.image_orientation is None:
            self.load_metadata()
            n_dim = len(self.image_metadata.shape)
            self.image_orientation = np.identity(n_dim, dtype=float)

    def _complete_image_spacing(self, force=False):
        if self.image_spacing is None:
            self.load_metadata()
            n_dim = len(self.image_metadata.shape)
            self.image_spacing = tuple([1.0] * n_dim)

    def _complete_image_dimensions(self, force=False):
        if self.image_dimension is None:
            self.load_metadata()
            self.image_dimension = tuple(self.image_metadata.shape)

    def create(self):
        return self

    def load_metadata(self, **kwargs):
        if self.image_metadata is not None:
            return

        if self.file_path is None or not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The image file could not be found at the expected location: {self.file_path}. "
                f"[{self.describe_self()}]"
            )

        # `Lazy load the data
        self.image_metadata = np.load(file=self.file_path, mmap_mode="r")

    def load_data(self, **kwargs):
        if self.image_data is not None:
            self.update_image_data()
            return

        if self.file_path is None or not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The image file could not be found at the expected location: {self.file_path}. "
                f"[{self.describe_self()}]"
            )

        self.image_data = np.load(self.file_path).astype(np.float32)
        self.update_image_data()


class MaskNumpyFile(ImageNumpyFile, MaskFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        if self.image_data is not None:
            self.update_image_data()
            return

        if self.file_path is None or not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The mask file could not be found at the expected location: {self.file_path}. [{self.describe_self()}]"
            )

        self.image_data = np.load(self.file_path).astype(int)
        self.update_image_data()
