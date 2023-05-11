import os
import os.path

import pandas as pd

from typing import Union
from fnmatch import fnmatch
from mirp.importData.utilities import supported_file_types


class ImageFile:

    def __init__(
            self,
            file_path: Union[None, str] = None,
            dir_path: Union[None, str] = None,
            sample_name: Union[None, str] = None,
            file_name: Union[None, str] = None,
            image_name: Union[None, str] = None,
            modality: Union[None, str] = None,
            file_type: Union[None, str] = None):

        # Sanity check.
        if isinstance(sample_name, list):
            raise ValueError("The sample_name argument should be None or a string.")

        self.file_path: Union[None, str] = file_path
        self.sample_name: Union[None, str] = sample_name
        self.image_name: Union[None, str] = image_name
        self.modality: Union[None, str] = modality
        self.file_type: Union[None, str] = file_type

        # Attempt to set the file name, if this is not externally provided.
        if isinstance(file_path, str) and file_name is None:
            file_name = os.path.basename(file_path)

            file_extension = None
            for current_file_extension in supported_file_types(self.file_type):
                if file_name.endswith(current_file_extension):
                    file_extension = current_file_extension

            if file_extension is not None:
                file_name = file_name.replace(file_extension, "")

            file_name = file_name.strip(" _^*")

        # Attempt to set the directory path, if this is not externally provided.
        if isinstance(file_path, str) and dir_path is None:
            dir_path = os.path.dirname(file_path)

        self.file_name: Union[None, str] = file_name
        self.dir_path: Union[None, str] = dir_path

    def set_sample_name(
            self,
            sample_name: str):

        self.sample_name = sample_name

    def get_sample_name(self):
        if self.sample_name is None:
            return "unset_sample_name__"

        return self.sample_name

    def get_file_name(self):
        if self.file_name is None:
            return "unset_file_name__"

        return self.file_name

    def get_dir_path(self):
        if self.dir_path is None:
            return "unset_dir_path__"

        return self.dir_path

    def create(self):
        # Import locally to avoid potential circular references.
        from mirp.importData.imageDicomFile import ImageDicomFile
        from mirp.importData.imageNiftiFile import ImageNiftiFile
        from mirp.importData.imageNrrdFile import ImageNrrdFile
        from mirp.importData.imageNumpyFile import ImageNumpyFile

        file_extensions = supported_file_types(file_type=self.file_type)

        if any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("dicom")):

            # Create DICOM-specific file.
            image_file = ImageDicomFile(
                file_path=self.file_path,
                dir_path=self.dir_path,
                sample_name=self.sample_name,
                file_name=self.file_name,
                image_name=self.image_name,
                modality=self.modality,
                file_type="dicom")

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("nifti")):

            # Create Nifti file.
            image_file = ImageNiftiFile(
                file_path=self.file_path,
                dir_path=self.dir_path,
                sample_name=self.sample_name,
                file_name=self.file_name,
                image_name=self.image_name,
                modality=self.modality,
                file_type="nifti")

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("nrrd")):

            # Create NRRD file.
            image_file = ImageNrrdFile(
                file_path=self.file_path,
                dir_path=self.dir_path,
                sample_name=self.sample_name,
                file_name=self.file_name,
                image_name=self.image_name,
                modality=self.modality,
                file_type="nrrd")

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("numpy")):

            # Create Numpy file.
            image_file = ImageNumpyFile(
                file_path=self.file_path,
                dir_path=self.dir_path,
                sample_name=self.sample_name,
                file_name=self.file_name,
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

    def get_identifiers(self):

        return pd.DataFrame.from_dict(dict({
            "modality": [self.modality],
            "file_type": [self.file_type],
            "sample_name": [self.get_sample_name()],
            "file_name": [self.get_file_name()],
            "dir_path": [self.get_dir_path()]}))

    def complete(self):
        # Set modality
        if self.modality is None:
            self.modality = "generic"


class MaskFile:

    def __init__(
            self,
            file_path: Union[None, str] = None,
            dir_path: Union[None, str] = None,
            sample_name: Union[None, str] = None,
            file_name: Union[None, str] = None,
            mask_name: Union[None, str] = None,
            modality: Union[None, str] = None,
            file_type: Union[None, str] = None):

        # Sanity check.
        if isinstance(sample_name, list):
            raise ValueError("The sample_name argument should be None or a string.")

        self.file_path: Union[None, str] = file_path
        self.sample_name: Union[None, str] = sample_name
        self.mask_name: Union[None, str] = mask_name
        self.modality: Union[None, str] = modality
        self.file_type: Union[None, str] = file_type

        # Attempt to set the file name, if this is not externally provided.
        if isinstance(file_path, str) and file_name is None:
            file_name = os.path.basename(file_path)

            file_extension = None
            for current_file_extension in supported_file_types(self.file_type):
                if file_name.endswith(current_file_extension):
                    file_extension = current_file_extension

            if file_extension is not None:
                file_name = file_name.replace(file_extension, "")

            file_name = file_name.strip(" _^*")

        # Attempt to set the directory path, if this is not externally provided.
        if isinstance(file_path, str) and dir_path is None:
            dir_path = os.path.dirname(file_path)

        self.file_name: Union[None, str] = file_name
        self.dir_path: Union[None, str] = dir_path

    def set_sample_name(
            self,
            sample_name: str):

        self.sample_name = sample_name

    def get_sample_name(self):
        if self.sample_name is None:
            return "unset_sample_name__"

        return self.sample_name

    def get_file_name(self):
        if self.file_name is None:
            return "unset_file_name__"

        return self.file_name

    def get_dir_path(self):
        if self.dir_path is None:
            return "unset_dir_path__"

        return self.dir_path

    def create(self):
        # Import locally to avoid potential circular references.
        from mirp.importData.imageDicomFile import MaskDicomFile
        from mirp.importData.imageNiftiFile import MaskNiftiFile
        from mirp.importData.imageNrrdFile import MaskNrrdFile
        from mirp.importData.imageNumpyFile import MaskNumpyFile

        file_extensions = supported_file_types(file_type=self.file_type)

        if any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("dicom")):

            # Create DICOM-specific file.
            image_file = MaskDicomFile(
                file_path=self.file_path,
                dir_path=self.dir_path,
                sample_name=self.sample_name,
                file_name=self.file_name,
                mask_name=self.mask_name,
                modality=self.modality,
                file_type="dicom")

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("nifti")):

            # Create Nifti file.
            image_file = MaskNiftiFile(
                file_path=self.file_path,
                dir_path=self.dir_path,
                sample_name=self.sample_name,
                file_name=self.file_name,
                image_name=self.mask_name,
                modality=self.modality,
                file_type="nifti")

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("nrrd")):

            # Create NRRD file.
            image_file = MaskNrrdFile(
                file_path=self.file_path,
                dir_path=self.dir_path,
                sample_name=self.sample_name,
                file_name=self.file_name,
                image_name=self.mask_name,
                modality=self.modality,
                file_type="nrrd")

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("numpy")):

            # Create Numpy file.
            image_file = MaskNumpyFile(
                file_path=self.file_path,
                dir_path=self.dir_path,
                sample_name=self.sample_name,
                file_name=self.file_name,
                image_name=self.mask_name,
                modality=self.modality,
                file_type="numpy")

        else:
            raise NotImplementedError(f"The provided mask type is not implemented: {self.file_type}")

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
                raise ValueError(
                    f"The file type does not correspond to a known, implemented maks type: {self.file_path}.")

            return False

        # Check that the file exists.
        if not os.path.exists(self.file_path):
            if raise_error:
                raise FileNotFoundError(f"The image file could not be found at the expected location: {self.file_path}")

            return False

        # Check that the file name contains mask_name. This is relevant for non-dicom files.
        if self.mask_name is not None:
            if not fnmatch(os.path.basename(self.file_path), self.mask_name):
                if raise_error:
                    raise ValueError(
                        f"The file name of the image file {self.file_path} does not match the expected pattern: {self.mask_name}")

            return False

        return True

    def get_identifiers(self):

        return pd.DataFrame.from_dict(dict({
            "modality": [self.modality],
            "file_type": [self.file_type],
            "sample_name": [self.get_sample_name()],
            "file_name": [self.get_file_name()],
            "dir_path": [self.get_dir_path()]}))

    def complete(self):
        # Set modality
        if self.modality is None:
            self.modality = "generic_mask"