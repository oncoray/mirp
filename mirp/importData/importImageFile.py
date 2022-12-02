import os
import os.path

import pandas as pd

from typing import Union
from fnmatch import fnmatch
from mirp.importData.utilities import supported_file_types


class ImageFile:

    def __init__(self,
                 file_path: Union[None, str],
                 sample_name: Union[None, str] = None,
                 suggested_sample_name: Union[None, str] = "",
                 image_name: Union[None, str] = None,
                 modality: Union[None, str] = None,
                 file_type: Union[None, str] = None):

        # Sanity check.
        if isinstance(sample_name, list):
            raise ValueError("The sample_name argument should be None or a string.")

        self.file_path = file_path
        self.sample_name = sample_name
        self.image_name = image_name
        self.modality = modality
        self.file_type = file_type

        # Set file name as candidate sample name.
        deparsed_file_name = ""
        if isinstance(file_path, str):
            deparsed_file_name = os.path.basename(file_path)
            deparsed_file_name = os.path.splitext(deparsed_file_name)[0]

            if image_name is not None:
                deparsed_file_name = deparsed_file_name.replace(image_name, "")

            deparsed_file_name = deparsed_file_name.strip(" _^*")

        if isinstance(suggested_sample_name, str):
            self.suggested_sample_name = dict({"original": sample_name,
                                               "dir_name": suggested_sample_name,
                                               "file_name": deparsed_file_name})

        elif isinstance(suggested_sample_name, dict):
            self.suggested_sample_name = suggested_sample_name

        else:
            self.suggested_sample_name = None

    def set_sample_name(self,
                        sample_name: str):
        self.sample_name = sample_name

    def create(self):
        # Import locally to avoid potential circular references.
        from mirp.importData.importImageDicomFile import ImageDicomFile
        from mirp.importData.importImageNiftiFile import ImageNiftiFile
        from mirp.importData.importImageNrrdFile import ImageNrrdFile
        from mirp.importData.importImageNumpyFile import ImageNumpyFile

        file_extensions = supported_file_types(file_type=self.file_type)

        if any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("dicom")):

            # Create DICOM-specific file.
            image_file = ImageDicomFile(
                file_path=self.file_path,
                sample_name=self.sample_name,
                suggested_sample_name=self.suggested_sample_name,
                image_name=self.image_name,
                modality=self.modality,
                file_type="dicom")

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("nifti")):

            # Create Nifti file.
            image_file = ImageNiftiFile(
                file_path=self.file_path,
                sample_name=self.sample_name,
                suggested_sample_name=self.suggested_sample_name,
                image_name=self.image_name,
                modality=self.modality,
                file_type="nifti")

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("nrrd")):

            # Create NRRD file.
            image_file = ImageNrrdFile(
                file_path=self.file_path,
                sample_name=self.sample_name,
                suggested_sample_name=self.suggested_sample_name,
                image_name=self.image_name,
                modality=self.modality,
                file_type="nrrd")

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("numpy")):

            # Create Numpy file.
            image_file = ImageNumpyFile(
                file_path=self.file_path,
                sample_name=self.sample_name,
                suggested_sample_name=self.suggested_sample_name,
                image_name=self.image_name,
                modality=self.modality,
                file_type="numpy")

        else:
            raise NotImplementedError(f"The provided image type is not implemented: {self.file_type}")

        return image_file

    def check(self, raise_error=False):
        # Check if file_path is set. Otherwise, none of the generic checks below can be used.
        if self.file_path is None:
            return True

        # Dispatch to subclass based on file_path.
        file_extensions = supported_file_types(self.file_type)

        # Check that the file type is correct.
        if not any(self.file_path.lower().endswith(ii) for ii in file_extensions):
            if raise_error:
                raise ValueError(f"The file type does not correspond to a known, implemented image type: {self.file_path}.")

            return False

        # Check that the file exists.
        if not os.path.exists(self.file_path):
            if raise_error:
                raise FileNotFoundError(f"The image file could not be found at the expected location: {self.file_path}")

            return False

        # Check that the file name contains image_name.
        if self.image_name is not None:
            if not fnmatch(os.path.basename(self.file_path), self.image_name):
                if raise_error:
                    raise ValueError(f"The file name of the image file {self.file_path} does not match the expected pattern:"
                                     f" {self.image_name}")

            return False

        return True

    def reset_sample_name(self):
        self.sample_name = self.suggested_sample_name["original"]

    def get_identifiers(self, style):
        if style == "basic":
            return pd.DataFrame.from_dict(dict({"modality": [self.modality],
                                                "sample_name": [self.sample_name]}))

        elif style == "extended":
            return pd.DataFrame.from_dict(dict({"modality": [self.modality],
                                                "file_type": [self.file_type],
                                                "sample_name": [self.sample_name]}))

        else:
            raise ValueError(f"The style parameter should be either 'basic' or 'extended'.")

    def complete(self):
        # Set modality
        if self.modality is None:
            self.modality = "generic"
