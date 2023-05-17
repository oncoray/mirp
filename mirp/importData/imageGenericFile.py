import os
import os.path

import itk
import numpy as np
import pandas as pd

from typing import Union, List, Tuple
from mirp.importData.utilities import supported_file_types, match_file_name, bare_file_name


class ImageFile:

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

        self.file_path: Union[None, str] = file_path
        self.sample_name: Union[None, str, List[str]] = sample_name
        self.image_name: Union[None, str] = image_name
        self.modality: Union[None, str] = image_modality
        self.file_type: Union[None, str] = image_file_type

        # Add image data
        self.image_data = image_data
        self.image_origin = image_origin
        self.image_orientation = image_orientation
        self.image_spacing = image_spacing
        self.image_dimension = image_dimensions

        # Add metadata
        self.image_metadata = None

        # Check incoming image data.
        _ = self._check_image_data()

        # Attempt to set the file name, if this is not externally provided.
        if isinstance(file_path, str) and file_name is None:
            file_name = os.path.basename(file_path)

            file_extension = None
            for current_file_extension in supported_file_types(self.file_type):
                if file_name.endswith(current_file_extension):
                    file_extension = current_file_extension
                    break

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
        from mirp.importData.imageITKFile import ImageITKFile
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
                image_file_type="dicom")

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions):
            if any(self.file_path.lower().endswith(ii) for ii in supported_file_types("nifti")):
                file_type = "nifti"
            elif any(self.file_path.lower().endswith(ii) for ii in supported_file_types("nrrd")):
                file_type = "nrrd"
            else:
                raise ValueError(f"DEV: specify file_type")

            # Create ITK file.
            image_file = ImageITKFile(
                file_path=self.file_path,
                dir_path=self.dir_path,
                sample_name=self.sample_name,
                file_name=self.file_name,
                image_name=self.image_name,
                image_modality=self.modality,
                image_file_type=file_type,
                image_data=self.image_data,
                image_origin=self.image_origin,
                image_orientation=self.image_orientation,
                image_spacing=self.image_spacing,
                image_dimensions=self.image_dimension)

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
                image_file_type="numpy")

        else:
            raise NotImplementedError(f"The provided image type is not implemented: {self.file_type}")

        return image_file

    def check(self, raise_error=False):

        # Check image data first.
        image_check = self._check_image_data(raise_error=raise_error)
        if not image_check:
            return False

        # Check if file_path is set. Otherwise, none of the generic checks below can be used.
        if self.file_path is None:
            return True

        # Dispatch to subclass based on file_path.
        allowed_file_extensions = supported_file_types(self.file_type)

        # Check that the file type is correct.
        if not self.file_path.lower().endswith(tuple(allowed_file_extensions)):
            if raise_error:
                raise ValueError(
                    f"The file type does not correspond to a known, implemented image type: {self.file_type}.")

            return False

        # Check that the file exists.
        if not os.path.exists(self.file_path):
            if raise_error:
                raise FileNotFoundError(
                    f"The image file could not be found at the expected location: {self.file_path}")

            return False

        # Check that the file name contains image_name.
        if self.image_name is not None:
            if not match_file_name(self.file_path, pattern=self.image_name, file_extension=allowed_file_extensions):
                if raise_error:
                    raise ValueError(
                        f"The file name of the image file {os.path.basename(self.file_path)} does not match "
                        f"the expected pattern: {self.image_name}")

            return False

        # Check that image file contains a sample name, if multiple sample names are present. To assess the filename,
        # we first strip the extension. Optionally we split the filename on the image name pattern, reducing the
        # filename into parts that should contain the sample name.
        if isinstance(self.sample_name, list) and len(self.sample_name) > 1:
            if self._get_sample_name_from_file() is None:
                if raise_error:
                    raise ValueError(
                        f"The file name of the image file {os.path.basename(self.file_path)} does not contain "
                        f"any of the expected patterns: {', '.join(self.sample_name)}")
                else:
                    return False

        return True

    def _check_image_data(self, raise_error=False):
        if self.image_data is None:
            return True

        # Check that image_data has the expected type.
        if not isinstance(self.image_data, np.ndarray):
            if raise_error:
                raise TypeError(
                    f"The image_data argument expects None or a numpy ndarray. Found object with class "
                    f"{type(self.image_data).name}"
                )
            else:
                return False

        # Check that image_data has up to 3 dimensions.
        data_shape = self.image_data.shape
        if not 1 <= len(data_shape) <= 3:
            if raise_error:
                raise ValueError(
                    f"MIRP supports image data up to 3 dimensions. The current numpy array has a dimension of "
                    f"{len(data_shape)} ({data_shape})."
                )
            else:
                return False

        # Check that the shape of image_data matches that of image_dimensions.
        if self.image_dimension is not None and not np.array_equal(self.image_dimension, data_shape):
            if raise_error:
                raise ValueError(
                    f"The shape of the image data itself and the purported shape (image_dimensions) are different. The "
                    f"current numpy array has dimensions ({data_shape}), where ({self.image_dimension}) is expected."
                )
            else:
                return False

        # Check that image_origin has the correct number of dimensions.
        if self.image_origin is not None and not len(data_shape) == len(self.image_origin):
            if raise_error:
                raise ValueError(
                    f"The dimensions of the image data itself ({len(data_shape)} and the dimensions of the origin ("
                    f"image_origin; {len(self.image_origin)}) are different."
                )
            else:
                return False

        # Check that image_orientation has the correct number of dimensions, and is correctly formatted.
        if self.image_orientation is not None:
            if not np.all(np.equal(self.image_orientation, len(data_shape))):
                if raise_error:
                    raise ValueError(
                        f"The orientation matrix should be square, with a dimension equal to the dimensions "
                        f"the image data itself ({len(data_shape)}. Found: {self.image_orientation.shape}."
                    )
                else:
                    return False

        # Check that image orientation has a l2-norm of 1.0.
        if self.image_orientation is not None:
            l2_norm = np.around(np.linalg.norm(self.image_orientation, ord=2), decimals=6)
            if not l2_norm == 1.0:
                if raise_error:
                    raise ValueError(
                        f"The orientation matrix should be square with an l2-norm of 1.0. Found: {l2_norm}."
                    )
                else:
                    return False

        # Check that spacing has the correct number of dimensions.
        if self.image_spacing is not None:
            if not len(self.image_spacing) == len(data_shape):
                if raise_error:
                    raise ValueError(
                        f"The dimensions of the image data itself ({len(data_shape)} and the dimensions of the voxel "
                        f"spacing (image_spacing; {len(self.image_spacing)}) are different."
                    )
                else:
                    return False

        # Check that spacing contains strictly positive values.
        if self.image_spacing is not None:
            if np.any(np.array(self.image_spacing) <= 0.0):
                if raise_error:
                    raise ValueError(
                        f"Image spacing should be strictly positive. Found: {self.image_spacing}."
                    )
                else:
                    return False

    def _get_sample_name_from_file(self) -> Union[None, str]:
        allowed_file_extensions = supported_file_types(self.file_type)

        if isinstance(self.sample_name, list) and len(self.sample_name) > 1:
            file_name = bare_file_name(x=self.file_name, file_extension=allowed_file_extensions)
            if self.image_name is not None:
                image_id_name = self.image_name
                if not isinstance(self.image_name, list):
                    image_id_name = [image_id_name]

                # Find the id that is present in the filename.
                matching_image_id = None
                for current_image_id_name in image_id_name:
                    if current_image_id_name in file_name:
                        matching_image_id = current_image_id_name
                        break

                if matching_image_id is not None:
                    # Handle wildcards in the image id.
                    matching_image_id.replace("?", "*")
                    matching_image_id = matching_image_id.split("*")
                    matching_image_id = [x for x in matching_image_id if x != ""]

                    if len(matching_image_id) == 0:
                        file_name_parts = [file_name]
                    else:
                        # Find the parts of the file name that do not contain the image identifier.
                        blocked_start = file_name.find(matching_image_id[0])
                        blocked_end = file_name.find(matching_image_id[-1]) + len(matching_image_id[-1])
                        file_name_parts = [""]
                        if blocked_start > 0:
                            file_name_parts.append(file_name[0:blocked_start])

                        if blocked_end < len(file_name):
                            file_name_parts.append(file_name[blocked_end:len(file_name)])
                else:
                    file_name_parts = [file_name]

            else:
                file_name_parts = [file_name]

            # Check if any sample name is present.
            matching_name = None
            matching_frac = 0.0
            for current_file_name_part in file_name_parts:
                for current_sample_name in self.sample_name:
                    if current_sample_name in current_file_name_part:
                        # Prefer the most complete matches.
                        if len(current_sample_name) / len(current_file_name_part) > matching_frac:
                            matching_frac = len(current_sample_name) / len(current_file_name_part)
                            matching_name = current_sample_name

                            if matching_frac == 1.0:
                                return matching_name

            return matching_name

        else:
            return None

    def get_identifiers(self):

        return pd.DataFrame.from_dict(dict({
            "modality": [self.modality],
            "file_type": [self.file_type],
            "sample_name": [self.get_sample_name()],
            "file_name": [self.get_file_name()],
            "dir_path": [self.get_dir_path()]}))

    def complete(self):
        # Load metadata.
        self.load_metadata()

        self._complete_modality()
        self._complete_sample_name()
        self._complete_image_dimensions()
        self._complete_image_origin()
        self._complete_image_orientation()
        self._complete_image_spacing()

        # Remove metadata. This allows file connections to be garbage collected.
        self.remove_metadata()

        # Check if the complete data passes verification.
        self.check()

    def _complete_modality(self):
        # Set modality.
        if self.modality is None:
            self.modality = "generic"

    def _complete_sample_name(self):
        # Set sample name.
        if isinstance(self.sample_name, list):
            file_sample_name = self._get_sample_name_from_file()

            if file_sample_name is None and len(self.sample_name) == 1:
                self.sample_name = self.sample_name[0]

            elif file_sample_name is not None:
                self.sample_name = file_sample_name

            else:
                self.sample_name = None

    def _complete_image_origin(self):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_sample_origin. Please specify "
            f"implementation for subclasses."
        )

    def _complete_image_orientation(self):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_sample_orientation. Please specify "
            f"implementation for subclasses."
        )

    def _complete_image_spacing(self):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_sample_spacing. Please specify "
            f"implementation for subclasses."
        )

    def _complete_image_dimensions(self):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_sample_dimensions. Please specify "
            f"implementation for subclasses."
        )

    def load_metadata(self):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of load_metadata. Please specify "
            f"implementation for subclasses."
        )

    def remove_metadata(self):
        self.image_metadata = None

    def load_data(self):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of load_data. Please specify "
            f"implementation for subclasses."
        )
    