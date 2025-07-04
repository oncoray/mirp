import copy
import fnmatch
import hashlib
import itertools
import os
import os.path
import re
import sys
import warnings

import numpy as np

from typing import Any
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from mirp._images.base_image import BaseImage
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._data_import.utilities import supported_file_types, match_file_name, bare_file_name, compute_file_distance


class ImageFile(BaseImage):

    def __init__(
            self,
            file_path: None | str = None,
            dir_path: None | str = None,
            sample_name: None | str | list[str] = None,
            file_name: None | str = None,
            image_name: None | str = None,
            image_modality: None | str = None,
            image_file_type: None | str = None,
            image_data: None | np.ndarray = None,
            image_origin: None | tuple[float, float, float] = None,
            image_orientation: None | np.ndarray = None,
            image_spacing: None | tuple[float, float, float] = None,
            image_dimensions: None | tuple[int, int, int] = None,
            **kwargs
    ):

        super().__init__(
            image_modality=image_modality,
            image_origin=image_origin,
            image_orientation=image_orientation,
            image_spacing=image_spacing,
            image_dimensions=image_dimensions,
            **kwargs
        )

        self.file_path = file_path
        self.sample_name = sample_name
        self.image_name = image_name

        if isinstance(image_file_type, str):
            image_file_type = image_file_type.lower()
        self.file_type = image_file_type

        # Add image data
        self.image_data = image_data

        # Check incoming image data.
        _ = self._check_image_data()

        self.update_image_data()

        # Check updated image data.
        _ = self._check_image_data()

        # Add metadata
        self.image_metadata = None
        self.is_limited_metadata = False

        # Attempt to set the file name, if this is not externally provided.
        if isinstance(file_path, str) and file_name is None:
            file_name = os.path.basename(file_path)

        # Attempt to set the directory path, if this is not externally provided.
        if isinstance(file_path, str) and dir_path is None:
            dir_path = os.path.dirname(file_path)

        self.file_name = file_name
        self.dir_path = dir_path
        self.associated_masks = None

        # Object metadata that will be exported when converted to native format.
        self.object_metadata: dict[str, str] = dict()

    def is_stackable(self, stack_images: str):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of is_stackable. Please specify "
            f"implementation for subclasses."
        )

    def copy(self):
        return copy.deepcopy(self)

    def on_file_system(self):
        return self.file_path is None or self.dir_path is None

    def get_identifiers(self, as_hash=False) -> dict | bytes:
        """
        General identifiers for images. Note that image_origin is not included, as this should be different for every
        slice in a volume.
        :param as_hash: boolean flag. When true returns a SHA256 hash of the identifier data.
        :return: a dictionary with identifiers.
        """

        identifier_data = dict({
            "modality": [self.modality],
            "file_type": [self.file_type],
            "sample_name": [self.get_sample_name(allow_none=False)],
            "dir_path": [self.get_dir_path(allow_none=False)],
            "image_dimensions": [self.get_image_dimension(as_str=True)],
            "image_spacing": [self.get_image_spacing(as_str=True)],
            "image_orientation": [self.get_image_orientation(as_str=True)]
        })

        if as_hash:
            return hashlib.sha256(str(identifier_data).encode(), usedforsecurity=False).digest()
        else:
            return identifier_data

    def associate_with_mask(
            self,
            mask_list,
            association_strategy: None | set[str] = None
    ):
        import math

        if mask_list is None or len(mask_list) == 0 or association_strategy is None:
            return

        # Match on sample name.
        if "sample_name" in association_strategy and self.sample_name is not None:
            matching_mask_list = [
                mask_file for mask_file in mask_list
                if self.sample_name == mask_file.sample_name
            ]

            if len(matching_mask_list) > 0:
                self.associated_masks = matching_mask_list
                return

        # Match on path distance. Then masks are filtered based on distance to the image, with masks with minimum
        # distance to the image being kept.
        if "file_distance" in association_strategy and self.dir_path is not None:
            file_distance = [
                compute_file_distance(self.dir_path, mask_file.dir_path)
                for mask_file in mask_list
            ]

            if any(math.isfinite(current_distance) for current_distance in file_distance):
                min_distance = min(file_distance)
                matching_mask_list = [
                    mask_file for ii, mask_file in enumerate(mask_list)
                    if file_distance[ii] == min_distance
                ]

                self.associated_masks = matching_mask_list
                return

        # Match on file name similarity. The image is matched against the most similar mask.
        if "file_name_similarity" in association_strategy and self.file_name is not None:
            ...

        return

    def set_sample_name(self, sample_name: str):

        self.sample_name = sample_name

    def get_sample_name(self, allow_none = True):
        if self.sample_name is None:
            if allow_none:
                return None
            else:
                return "unset_sample_name"

        return self.sample_name

    def get_dir_path(self, allow_none = True):
        if self.dir_path is None:
            if allow_none:
                return None
            else:
                return "unset_dir_path"

        return self.dir_path

    def get_file_name(self, allow_none = True):
        if self.file_name is None:
            if allow_none:
                return None
            else:
                return "unset_file_name"

        return self.file_name

    def get_image_dimension(self, as_str: bool = False):
        if not as_str:
            return self.image_dimension

        if self.image_dimension is None:
            return "unset_image_dimension"

        return str(self.image_dimension)

    def get_image_spacing(self, as_str: bool = False):
        if not as_str:
            return self.image_spacing

        if self.image_spacing is None:
            return "unset_image_spacing"

        return str(self.image_spacing)

    def get_image_origin(self, as_str: bool = False):
        if not as_str:
            return self.image_origin

        if self.image_origin is None:
            return "unset_image_origin"

        return str(self.image_origin)

    def get_image_orientation(self, as_str: bool = False):
        if not as_str:
            return self.image_orientation

        if self.image_orientation is None:
            return "unset_image_orientation"

        return str(np.ravel(self.image_orientation))

    def set_modality(self, modality: None | str):
        from mirp._data_import.utilities import supported_image_modalities
        if modality is None:
            return

        if not isinstance(modality, str):
            raise ValueError(f"modality is expected to be a character string. Found: {modality} [{self.describe_self()}]")

        if modality == "generic":
            raise ValueError(f"modality cannot be 'generic' [{self.describe_self()}]")

        modality = supported_image_modalities(modality)
        if self.modality is None or self.modality == "generic":
            self.modality = modality[0]

    def create(self):
        # Import locally to avoid potential circular references.
        from mirp._data_import.dicom_file import ImageDicomFile
        from mirp._data_import.itk_file import ImageITKFile
        from mirp._data_import.numpy_file import ImageNumpyFile

        file_extensions = supported_file_types(file_type=self.file_type)

        if any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("dicom")):

            file_class = ImageDicomFile
            file_type = "dicom"

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("itk")):
            if any(self.file_path.lower().endswith(ii) for ii in supported_file_types("nifti")):
                file_type = "nifti"
            elif any(self.file_path.lower().endswith(ii) for ii in supported_file_types("nrrd")):
                file_type = "nrrd"
            else:
                raise ValueError(f"DEV: specify file_type")

            file_class = ImageITKFile

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("numpy")):

            file_class = ImageNumpyFile
            file_type = "numpy"

        else:
            raise NotImplementedError(
                f"The provided image type is not implemented: {self.file_type} [{self.describe_self()}]"
            )

        image_file = file_class(
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
            image_dimensions=self.image_dimension
        ).create()

        return image_file

    def check(self, raise_error=False, remove_metadata=False):

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
                    f"The file type does not correspond to a known, implemented image type: {self.file_type}. "
                    f"[{self.describe_self()}]"
                )

            return False

        # Check that the file exists.
        if not os.path.exists(self.file_path):
            if raise_error:
                raise FileNotFoundError(
                    f"The image file could not be found at the expected location: {self.file_path} "
                    f"[{self.describe_self()}]"
                )

            return False

        # Check that the file name contains image_name.
        if self.image_name is not None:
            if not match_file_name(self.file_path, pattern=self.image_name, file_extension=allowed_file_extensions):
                if raise_error:
                    raise ValueError(
                        f"The file name of the image file {os.path.basename(self.file_path)} does not match "
                        f"the expected pattern: {self.image_name} [{self.describe_self()}]"
                    )

            return False

        # Check modality.
        modality_check = self._check_modality(raise_error=raise_error)
        if not modality_check:
            return False

        # Check sample name.
        sample_name_check = self._check_sample_name(raise_error=raise_error)
        if not sample_name_check:
            return False

        return True

    def _check_image_data(self, raise_error=False) -> bool:
        if self.image_data is None:
            return True

        # Check that image_data has the expected type.
        if not isinstance(self.image_data, np.ndarray):
            if raise_error:
                raise TypeError(
                    f"The image_data argument expects None or a numpy ndarray. Found object with class "
                    f"{type(self.image_data).name} [{self.describe_self()}]"
                )
            else:
                return False

        # Check that image_data has up to 3 dimensions.
        data_shape = self.image_data.shape
        if not 1 <= len(data_shape) <= 3:
            if raise_error:
                raise ValueError(
                    f"MIRP supports image data up to 3 dimensions. The current numpy array has a dimension of "
                    f"{len(data_shape)} ({data_shape}). [{self.describe_self()}]"
                )
            else:
                return False

        # Check that the shape of image_data matches that of image_dimensions.
        if self.image_dimension is not None and not np.array_equal(self.image_dimension, data_shape):
            if raise_error:
                raise ValueError(
                    f"The shape of the image data itself and the purported shape (image_dimensions) are different. The "
                    f"current numpy array has dimensions ({data_shape}), where ({self.image_dimension}) is expected."
                    f"[{self.describe_self()}]"
                )
            else:
                return False

        # Check that image_origin has the correct number of dimensions.
        if self.image_origin is not None and not len(data_shape) == len(self.image_origin):
            if raise_error:
                raise ValueError(
                    f"The dimensions of the image data itself ({len(data_shape)} and the dimensions of the origin ("
                    f"image_origin; {len(self.image_origin)}) are different. [{self.describe_self()}]"
                )
            else:
                return False

        # Check that image_orientation has the correct number of dimensions, and is correctly formatted.
        if self.image_orientation is not None:
            if not np.all(np.equal(self.image_orientation.shape, len(data_shape))):
                if raise_error:
                    raise ValueError(
                        f"The orientation matrix should be square, with a dimension equal to the dimensions "
                        f"the image data itself ({len(data_shape)}. Found: {self.image_orientation.shape}."
                        f"[{self.describe_self()}]"
                    )
                else:
                    return False

        # Check that image orientation has a l2-norm of 1.0.
        if self.image_orientation is not None:
            l2_norm = np.around(np.linalg.norm(self.image_orientation, ord=2), decimals=3)
            if not l2_norm == 1.0:
                if raise_error:
                    raise ValueError(
                        f"The orientation matrix should be square with an l2-norm of 1.0. Found: {l2_norm}."
                        f"[{self.describe_self()}]"
                    )
                else:
                    return False

        # Check that spacing has the correct number of dimensions.
        if self.image_spacing is not None:
            if not len(self.image_spacing) == len(data_shape):
                if raise_error:
                    raise ValueError(
                        f"The dimensions of the image data itself ({len(data_shape)}) and the dimensions of the voxel "
                        f"spacing (image_spacing; {len(self.image_spacing)}) are different. [{self.describe_self()}]"
                    )
                else:
                    return False

        # Check that spacing contains strictly positive values.
        if self.image_spacing is not None:
            if np.any(np.array(self.image_spacing) <= 0.0):
                if raise_error:
                    raise ValueError(
                        f"Image spacing should be strictly positive. Found: {self.image_spacing}. "
                        f"[{self.describe_self()}]"
                    )
                else:
                    return False

    def _check_sample_name(self, raise_error: bool) -> bool:
        # Check that image file contains a sample name, if multiple sample names are present. To assess the filename,
        # we first strip the extension. Optionally we split the filename on the image name pattern, reducing the
        # filename into parts that should contain the sample name.
        if isinstance(self.sample_name, list) and len(self.sample_name) > 1:
            if self.get_sample_name_from_file() is None:
                if raise_error:
                    raise ValueError(
                        f"The file name of the image file {os.path.basename(self.file_path)} does not contain "
                        f"any of the expected patterns: {', '.join(self.sample_name)} [{self.describe_self()}]"
                    )
                else:
                    return False

        return True

    def _check_modality(self, raise_error: bool) -> bool:
        return True

    def get_sample_name_from_file(self, must_succeed=False) -> None | str:
        allowed_file_extensions = supported_file_types(self.file_type)

        # Do not obtain sample name from the file name if a file name has already been set.
        if isinstance(self.sample_name, str):
            return None

        if self.file_name is None:
            return None

        # Select the most appropriate sample name.
        if isinstance(self.sample_name, list) and len(self.sample_name) > 1:
            file_name = bare_file_name(x=self.file_name, file_extension=allowed_file_extensions)
            if self.image_name is not None:
                image_id_name = self.image_name
                if not isinstance(image_id_name, list):
                    image_id_name = [image_id_name]

                # Find the id that is present in the filename.
                matching_image_id = None
                for current_image_id_name in image_id_name:
                    # Replace markers for the sample name (#).
                    current_image_id_name = current_image_id_name.replace("#", "*")
                    if fnmatch.fnmatch(file_name, current_image_id_name):
                        matching_image_id = current_image_id_name
                        break

                if matching_image_id is not None:
                    # Handle wildcards in the image id.
                    matching_image_id = matching_image_id.replace("?", "*")

                    # Split on wildcards.
                    matching_image_id = matching_image_id.split("*")

                    # Identify parts of the image id that are not empty after splitting.
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

        elif must_succeed:
            return bare_file_name(x=self.file_name, file_extension=allowed_file_extensions)
        else:
            return None

    def _get_numeric_sequence_from_file(self) -> None | list[str]:
        allowed_file_extensions = supported_file_types(self.file_type)
        file_name = bare_file_name(x=self.file_name, file_extension=allowed_file_extensions)

        file_name_parts = [file_name]

        # Remove image name.
        if self.image_name is not None:
            image_id_name = self.image_name
            if not isinstance(image_id_name, list):
                image_id_name = [image_id_name]

            # Find the id that is present in the filename.
            matching_image_id = None
            for current_image_id_name in image_id_name:
                if current_image_id_name in file_name:
                    matching_image_id = current_image_id_name
                    break

            if matching_image_id is not None:
                # Handle wildcards in the image id.
                matching_image_id = matching_image_id.replace("?", "*")
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

        # Remove empty strings (if any).
        file_name_parts = [
            current_file_name_part for current_file_name_part in file_name_parts
            if len(current_file_name_part) > 0
        ]

        if len(file_name_parts) == 0:
            return None

        # Remove sample name.
        if self.sample_name is not None:
            if isinstance(self.sample_name, list):
                raise TypeError(
                    f"The sample_name attribute cannot be a list for extracting numeric sequences. "
                    f"[{self.describe_self()}]"
                )

            file_name_parts = [
                current_file_name_part.split(sep=self.sample_name)
                for current_file_name_part in file_name_parts
            ]

            # Flatten list
            file_name_parts = list(itertools.chain.from_iterable(file_name_parts))

        # Remove empty strings (if any).
        file_name_parts = [
            current_file_name_part for current_file_name_part in file_name_parts
            if len(current_file_name_part) > 0
        ]

        if len(file_name_parts) == 0:
            return None

        # Isolate sequences of numeric values using regex. Then remove parts where no numeric value was found.
        file_name_parts = [re.findall(r'\d+', current_file_name_part) for current_file_name_part in file_name_parts]
        file_name_parts = [
            current_file_name_part for current_file_name_part in file_name_parts
            if len(current_file_name_part) > 0
        ]

        if len(file_name_parts) == 0:
            return None

        # Flatten list and return.
        return list(itertools.chain.from_iterable(file_name_parts))

    def complete(self, remove_metadata=False, force=False):
        self._complete_modality()
        self._complete_sample_name()
        self._complete_image_dimensions(force=force)
        self._complete_image_origin(force=force)
        self._complete_image_orientation(force=force)
        self._complete_image_spacing(force=force)

        # Check if the complete data passes verification.
        self.check(raise_error=True, remove_metadata=False)

        # Remove metadata. This allows file connections to be garbage collected.
        if remove_metadata:
            self.remove_metadata()

    def _complete_modality(self):
        # Set modality.
        if self.modality is None:
            self.modality = "generic"

    def _complete_sample_name(self):
        # Set sample name.
        if isinstance(self.sample_name, list):
            file_sample_name = self.get_sample_name_from_file()

            if file_sample_name is None and len(self.sample_name) == 1:
                self.sample_name = self.sample_name[0]

            elif file_sample_name is not None:
                self.sample_name = file_sample_name

            else:
                self.sample_name = None

    def _complete_image_origin(self, force=False, frame_id: None | int = None):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_sample_origin. Please specify "
            f"implementation for subclasses."
        )

    def _complete_image_orientation(self, force=False, frame_id: None | int = None):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_sample_orientation. Please specify "
            f"implementation for subclasses."
        )

    def _complete_image_spacing(self, force=False, frame_id: None | int = None):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_sample_spacing. Please specify "
            f"implementation for subclasses."
        )

    def _complete_image_dimensions(self, force=False):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_sample_dimensions. Please specify "
            f"implementation for subclasses."
        )

    def load_metadata(self, limited=False, include_image=False):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of load_metadata. Please specify "
            f"implementation for subclasses."
        )

    def remove_metadata(self):
        self.image_metadata = None

    def load_data(self, **kwargs):
        """
        Loads data from file as numpy array, and attaches it to the current object by setting the image_data attribute.
        :param kwargs: Unused arguments.
        :return:
        """
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of load_data. Please specify "
            f"implementation for subclasses."
        )

    def stack_slices(self):
        """
        Stacks slices and sets the image_data attribute. Does nothing unless the image consists of stacks.
        :return:
        """
        pass

    def update_image_data(self):
        if self.image_data is None:
            return

        if not isinstance(self.image_data, np.ndarray):
            raise TypeError(
                f"The image_data argument expects None or a numpy ndarray. Found object with class "
                f"{type(self.image_data).name}. [{self.describe_self()}]"
            )

        # If the image is already 3D, we forgo adding dimensions.
        if len(self.image_data.shape) == 3:
            return

        if not 1 <= len(self.image_data.shape) <= 3:
            raise ValueError(
                f"MIRP supports image data up to 3 dimensions. The current numpy array has a dimension of "
                f"{len(self.image_data.shape)} ({self.image_data.shape}). [{self.describe_self()}]"
            )

        dims_to_add = 3 - len(self.image_data.shape)

        # Add new dimensions to image.
        if dims_to_add == 2:
            self.image_data = self.image_data[np.newaxis, np.newaxis, :]
        elif dims_to_add == 1:
            self.image_data = self.image_data[np.newaxis, :, :]

        # Update image dimensions attribute.
        if self.image_dimension is not None and not len(self.image_dimension) == 3:
            image_dimension = list(self.image_dimension)
            if dims_to_add == 2:
                image_dimension.insert(0, 1)
                image_dimension.insert(0, 1)
            elif dims_to_add == 1:
                image_dimension.insert(0, 1)

            self.image_dimension = tuple(image_dimension)

        # Update image origin attribute.
        if self.image_origin is not None and not len(self.image_origin) == 3:
            image_origin = list(self.image_origin)
            if dims_to_add == 2:
                image_origin.insert(0, 0)
                image_origin.insert(0, 0)
            elif dims_to_add == 1:
                image_origin.insert(0, 0)

            self.image_origin = tuple(image_origin)

        # Update image orientation attribute.
        if self.image_orientation is not None and not np.all(np.equal(self.image_orientation.shape, 3)):
            image_orientation = np.identity(3)
            if dims_to_add == 2:
                image_orientation[1:2, 1:2] = self.image_orientation
            elif dims_to_add == 1:
                image_orientation[1:3, 1:3] = self.image_orientation

            self.image_orientation = image_orientation

        # Update image spacing attribute.
        if self.image_spacing is not None and not len(self.image_spacing) == 3:
            image_spacing: list[float] = list(self.image_spacing)
            if dims_to_add == 2:
                image_spacing.insert(0, image_spacing[0])
                image_spacing.insert(0, 1.0)
            if dims_to_add == 1:
                image_spacing.insert(0, 1.0)

            self.image_spacing = tuple(image_spacing)

    def to_object(self, **kwargs) -> GenericImage:

        self.load_data(**kwargs)
        self.complete()
        self.stack_slices()
        self.update_image_data()
        self.set_object_metadata()

        return GenericImage(
            sample_name=self.sample_name,
            image_modality=self.modality,
            image_data=self.image_data,
            image_spacing=self.image_spacing,
            image_origin=self.image_origin,
            image_orientation=self.image_orientation,
            image_dimensions=self.image_dimension,
            metadata=self.object_metadata
        )

    def set_object_metadata(self):
        """
        Updates the object metadata that is passed to native image and mask classes in to_object.
        """
        attributes = []
        # Add file name to object metadata.
        if self.file_name is not None:
            attributes += [("file_name", self.file_name)]

        # Add directory to object metadata.
        if self.dir_path is not None:
            attributes += [("dir_path", self.dir_path)]

        # Add file type to object metadata.
        if self.file_type is not None:
            attributes += [("file_type", self.file_type)]

        self.object_metadata.update(dict(attributes))

    def export_metadata(self, **kwargs) -> None | dict[str, Any]:
        # Get general attributes.
        parent_attributes = self._get_export_attributes()

        # Add metadata.
        metadata = []
        if self.image_spacing is not None:
            metadata += [
                ("spacing_z", self.image_spacing[0]),
                ("spacing_y", self.image_spacing[1]),
                ("spacing_x", self.image_spacing[2])
            ]
        parent_attributes.update(dict(metadata))

        return parent_attributes

    def _get_export_attributes(self) -> dict[str, Any]:
        attributes = []
        if self.sample_name is not None:
            attributes += [("sample_name", self.sample_name)]
        if self.modality is not None:
            attributes += [("modality", self.modality)]
        if self.dir_path is not None:
            attributes += [("dir_path", self.dir_path)]
        if self.file_name is not None:
            attributes += [("file_name", self.file_name)]

        return dict(attributes)

    def describe_self(self):
        return f"{self._get_export_attributes()}"

    def check_associated_masks(self):
        if self.associated_masks is None:
            return

        for mask in self.associated_masks:
            self._check_associated_mask_image_data(mask=mask)

    def _check_associated_mask_image_data(self, mask: Self):
        """Check whether image and mask plausibly share the same frame of reference."""

        problem_list = []
        # Mismatch in grid dimension
        if isinstance(self.image_dimension, tuple) and isinstance(mask.image_dimension, tuple):
            if not np.array_equal(self.image_dimension, mask.image_dimension):
                problem_list += [
                    f"different dimensions: \n\t\timage: {self.image_dimension}\n\t\tmask: {mask.image_dimension}"
                ]

        # Mismatch in origin
        if isinstance(self.image_origin, tuple) and isinstance(mask.image_origin, tuple):
            if not np.allclose(self.image_origin, mask.image_origin):
                problem_list += [
                    f"different origin: \n\t\timage: {self.image_origin}\n\t\tmask: {mask.image_origin}"
                ]

        # Mismatch in spacing
        if isinstance(self.image_spacing, tuple) and isinstance(mask.image_spacing, tuple):
            if not np.allclose(self.image_spacing, mask.image_spacing):
                problem_list += [
                    f"different spacing: \n\t\timage: {self.image_spacing}\n\t\tmask: {mask.image_spacing}"
                ]

        # Mismatch in orientation
        if isinstance(self.image_orientation, np.ndarray) and isinstance(mask.image_orientation, np.ndarray):
            if not np.allclose(self.image_orientation, mask.image_orientation):
                problem_list += [
                    f"different orientation: \n\t\timage: {np.ravel(self.image_orientation)}\n\t\tmask: "
                    f"{np.ravel(mask.image_orientation)}"
                ]

        if len(problem_list) > 0:
            warnings.warn(
                f"Image {self.describe_self()} and mask {mask.describe_self()} may not have the same frame of "
                f"reference. Please check if segmentation masks are placed correctly:\n\t" + "\n\t".join(problem_list),
                UserWarning
            )


class MaskFile(ImageFile):

    def __init__(
            self,
            roi_name: None | str | list[str] | dict[str, str] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Curate input.
        if roi_name is None or isinstance(roi_name, str) or \
            (isinstance(roi_name, list) and all(isinstance(x, str) for x in roi_name)) or \
                (isinstance(roi_name, dict) and all(isinstance(x, str) for x in roi_name.values()) and all(
                    isinstance(x, str) for x in roi_name.keys())):
            self.roi_name = roi_name
        else:
            raise TypeError(
                f"ROI names are expected to be a string, a list of strings, or a dictionary of strings. Found:"
                f" {roi_name} with type {type(roi_name)}. [{self.describe_self()}]")

    def is_stackable(self, stack_images: str):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of is_stackable. Please specify "
            f"implementation for subclasses."
        )

    def _complete_modality(self):
        # Set modality.
        if self.modality is None:
            self.modality = "generic_mask"

    def _complete_image_origin(self, force=False, frame_id=None):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_image_origin. Please specify "
            f"implementation for subclasses."
        )

    def _complete_image_orientation(self, force=False, frame_id=None):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_image_orientation. Please specify "
            f"implementation for subclasses."
        )

    def _complete_image_spacing(self, force=False, frame_id=None):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_image_spacing. Please specify "
            f"implementation for subclasses."
        )

    def _complete_image_dimensions(self, force=False):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of _complete_image_dimensions. Please specify "
            f"implementation for subclasses."
        )

    def load_metadata(self, **kwargs):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of load_metadata. Please specify "
            f"implementation for subclasses."
        )

    def load_data(self, **kwargs):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of load_data. Please specify "
            f"implementation for subclasses."
        )

    def set_modality(self, modality: None | str):
        from mirp._data_import.utilities import supported_mask_modalities
        if modality is None:
            return

        if not isinstance(modality, str):
            raise ValueError(
                f"modality is expected to be a character string. Found: {modality}. [{self.describe_self()}]"
            )

        if modality == "generic_mask":
            raise ValueError(f"modality cannot be 'generic_mask'. [{self.describe_self()}]")

        modality = supported_mask_modalities(modality)
        if self.modality is None or self.modality == "generic_mask":
            self.modality = modality[0]

    def create(self):
        # Import locally to avoid potential circular references.
        from mirp._data_import.dicom_file import MaskDicomFile
        from mirp._data_import.itk_file import MaskITKFile
        from mirp._data_import.numpy_file import MaskNumpyFile

        file_extensions = supported_file_types(file_type=self.file_type)

        if any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("dicom")):

            file_class = MaskDicomFile
            file_type = "dicom"

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("itk")):
            if any(self.file_path.lower().endswith(ii) for ii in supported_file_types("nifti")):
                file_type = "nifti"
            elif any(self.file_path.lower().endswith(ii) for ii in supported_file_types("nrrd")):
                file_type = "nrrd"
            else:
                raise ValueError(f"DEV: specify file_type")

            file_class = MaskITKFile

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("numpy")):

            file_class = MaskNumpyFile
            file_type = "numpy"

        else:
            raise NotImplementedError(
                f"The provided mask type is not implemented: {self.file_type}. [{self.describe_self()}]"
            )

        image_file = file_class(
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
            image_dimensions=self.image_dimension,
            roi_name=self.roi_name
        ).create()

        return image_file

    def check_mask(self, raise_error=True):
        if self.image_data is None:
            raise TypeError(f"DEV: the image_data attribute has not been set. [{self.describe_self()}]")

        if np.issubdtype(self.image_data.dtype, bool):

            if not np.any(self.image_data):
                if raise_error:
                    warnings.warn(
                        f"No regions of interest were formed. "
                        f"The mask object only contains background values (False). "
                        f"No voxels were found with True values to identify segmentation masks. "
                        f"[{self.describe_self()}]",
                        UserWarning
                    )
            return True

        if np.issubdtype(self.image_data.dtype, np.integer):
            if np.any(self.image_data < 0):
                if raise_error:
                    raise ValueError(
                        f"Labels in a mask should be 0 or positive integers. Negative values were found. "
                        f"Note that 0 is interpreted as background. [{self.describe_self()}]"
                    )
                return False

            if len(np.unique(self.image_data)) > 10:
                if raise_error:
                    warnings.warn(
                        f"More than 10 labels were found. Please check that this is correct. "
                        f"[{self.describe_self()}]",
                        UserWarning
                    )

            if np.all(self.image_data == 0):
                if raise_error:
                    warnings.warn(
                        f"No regions of interest were formed. The mask object only contains "
                        f"background values (0). No voxels were found with positive integers to identify segmentation "
                        f"masks. [{self.describe_self()}]",
                        UserWarning
                    )

        else:
            raise TypeError(f"DEV: the image_data contains non-integer data ({self.image_data.dtype})")

        return True

    def to_object(self, **kwargs) -> None | list[BaseMask]:

        self.load_data()
        self.complete()
        self.stack_slices()
        self.update_image_data()
        self.set_object_metadata()
        self.check_mask(raise_error=True)

        mask_list = []
        if np.issubdtype(self.image_data.dtype, bool):
            if not np.any(self.image_data):
                return None

            if isinstance(self.roi_name, str):
                roi_name = self.roi_name
            elif isinstance(self.roi_name, list):
                if len(self.roi_name) == 1:
                    roi_name = self.roi_name[0]
                else:
                    warnings.warn(
                        f"The name of the region of interest could not be unambiguously determined. "
                        f"One of the following should be used, but it is not clear which: {self.roi_name}. "
                        f"[{self.describe_self()}]",
                        UserWarning
                    )
                    roi_name = "region_1"
            elif isinstance(self.roi_name, dict):
                if len(self.roi_name) == 1:
                    roi_name = list(self.roi_name.values())[0]
                else:
                    warnings.warn(
                        f"The name of the region of interest could not be unambiguously determined. "
                        f"One of the following should be used, but it is not clear which: {self.roi_name}. "
                        f"[{self.describe_self()}]",
                        UserWarning
                    )
                    roi_name = "region_1"
            else:
                roi_name = "region_1"

            mask_list += [
                BaseMask(
                    roi_name=roi_name,
                    sample_name=self.sample_name,
                    image_modality=self.modality,
                    image_data=self.image_data,
                    image_spacing=self.image_spacing,
                    image_origin=self.image_origin,
                    image_orientation=self.image_orientation,
                    image_dimensions=self.image_dimension,
                    metadata=self.object_metadata
                )
            ]

        else:

            # Identify available labels that are non-background.
            labels = list(set(np.unique(self.image_data)) - {0})

            if len(labels) == 0:
                return None

            # Check which labels should be kept.
            filtered_labels = []
            for current_label in labels:
                if self.roi_name is None:
                    filtered_labels += [current_label]
                elif isinstance(self.roi_name, str):
                    if self.roi_name == str(current_label) or self.roi_name == "region_" + str(current_label):
                        filtered_labels += [current_label]
                elif isinstance(self.roi_name, list):
                    if str(current_label) in self.roi_name or "region_" + str(current_label) in self.roi_name:
                        filtered_labels += [current_label]
                elif isinstance(self.roi_name, dict):
                    if str(current_label) in self.roi_name.keys() or "region_" + str(current_label) in self.roi_name.keys():
                        filtered_labels += [current_label]

            if len(filtered_labels) == 0:
                warnings.warn(
                    f"No regions of interest were formed. "
                    f"The available labels ({', '.join([str(x) for x in labels])}) likely did not match "
                    f"any of the expected labels ({self.roi_name}). "
                    f"[{self.describe_self()}]",
                    UserWarning
                )

                return None

            for current_label in filtered_labels:

                # Find roi name to use for the current object.
                if isinstance(self.roi_name, str):
                    roi_name = self.roi_name
                elif isinstance(self.roi_name, list):
                    roi_name = [x for x in self.roi_name if str(current_label) == x or "region_" + str(current_label) == x][0]
                elif isinstance(self.roi_name, dict):
                    roi_name = self.roi_name.get(str(current_label))
                    if roi_name is None:
                        roi_name = self.roi_name.get("region_" + str(current_label))
                else:
                    roi_name = "region_" + str(current_label)

                mask_list += [
                    BaseMask(
                        roi_name=roi_name,
                        sample_name=self.sample_name,
                        image_modality=self.modality,
                        image_data=self.image_data == current_label,
                        image_spacing=self.image_spacing,
                        image_origin=self.image_origin,
                        image_orientation=self.image_orientation,
                        image_dimensions=self.image_dimension
                    )
                ]

        return mask_list

    def export_metadata(self):
        return

    def export_roi_labels(self) -> dict[str, Any]:

        if self.image_data is None:
            self.load_data()
            self.complete()
            self.stack_slices()

        # Identify available labels that are non-background.
        labels = list(set(np.unique(self.image_data)) - {0})
        n_labels = max([1, len(labels)])

        if len(labels) == 0:
            labels = [None]

        # Get general attributes.
        parent_attributes = self._get_export_attributes()

        # Add roi labels as attribute.
        attributes = [("roi_label", labels)]
        parent_attributes.update(dict(attributes))

        return parent_attributes


class MaskFullImage(MaskFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.roi_name = "full_image_mask"

    def to_object(
            self,
            image: None | ImageFile,
            **kwargs
    ) -> None | list[BaseMask]:
        if image is None:
            raise TypeError(
                f"Creation of a full image mask requires that the corresponding image is set. "
                f"No image was provided. [{self.describe_self()}]"
            )
        else:
            image.complete()

        self._complete_modality()

        return [BaseMask(
            roi_name=self.roi_name,
            sample_name=image.sample_name,
            image_modality=self.modality,
            image_data=np.ones(image.image_dimension, dtype=bool),
            image_spacing=image.image_spacing,
            image_origin=image.image_origin,
            image_orientation=image.image_orientation,
            image_dimensions=image.image_dimension,
            metadata=self.object_metadata
        )]
