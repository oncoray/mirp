import numpy as np

from typing import Union

from mirp.importData.imageGenericFile import ImageFile


class ImageNumpyFile(ImageFile):
    def __init__(
            self,
            file_path: Union[None, str] = None,
            dir_path: Union[None, str] = None,
            sample_name: Union[None, str] = None,
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

    def check(self, raise_error=False):

        # Perform general checks.
        if not super().check(raise_error=raise_error):
            return False

        # Read Numpy file.
        image_data = np.load(file=self.file_path)

        # Check that the contents are in fact a ndarray.
        if not isinstance(image_data, np.ndarray):
            if raise_error:
                raise TypeError(f"The current numpy dataset {self.file_path} does not have the expected content. "
                                f"Found: {type(image_data)}. Expected: numpy.ndarray")

            return False

        # Check dimensions.
        if not 0 < image_data.ndim <= 3:
            if raise_error:
                raise ValueError(f"The current numpy dataset has as an unexpected number of dimensions: "
                                 f"Found: {image_data.ndim}. Expected: 1, 2, or 3 dimensions")

            return False

        return True
