import os
import os.path
import copy

from typing import Union, List
from itertools import chain
from mirp.importData.utilities import supported_file_types
from mirp.importData.maskGenericFile import MaskFile


class MaskDirectory:

    def __init__(
            self,
            directory,
            sample_name: Union[None, str, List[str]] = None,
            mask_name: Union[None, str, List[str]] = None,
            sub_folder: Union[None, str] = None,
            modality: Union[None, str] = None,
            file_type: Union[None, str] = None,
            contain_multiple_samples: [bool] = False,
            **kwargs):

        self.mask_directory = directory
        self.sample_name = sample_name
        self.mask_name = mask_name
        self.sub_folder = sub_folder
        self.modality = modality
        self.file_type: Union[None, str] = file_type
        self.contain_multiple_samples = contain_multiple_samples

    def _directory_contains_mask(self):

        # The _create_masks generator only returns something if a valid mask can be found in the indicated directory.
        # When that happens, return True, otherwise return False.
        for _ in self._create_masks():
            return True

        return False

    def _generate_sub_directories(self):
        # Generates subdirectories

        # Copy sample names. These are candidates for the names of subdirectories.
        sample_name = copy.deepcopy(self.sample_name)
        if sample_name is not None and not isinstance(sample_name, list):
            sample_name = [sample_name]

        # Yield new object to check.
        if sample_name is None or len(sample_name) == 1:
            if self.sub_folder is not None:
                current_directory = os.path.join(self.mask_directory, self.sub_folder)
            else:
                current_directory = os.path.join(self.mask_directory)

            if os.path.isdir(self.mask_directory):
                new_object = copy.deepcopy(self)
                new_object.mask_directory = current_directory
                new_object.sample_name = sample_name
                new_object.sub_folder = None

                if new_object._directory_contains_mask():
                    yield new_object

        # Add check to see if any sample names matches a sub_directory.
        any_sample_name_matches_directory = False

        # Try is sample names match directory names.
        if self.sample_name is not None:
            for current_sample_name in sample_name:
                current_directory = os.path.join(self.mask_directory, current_sample_name)

                # Check that the directory actually exists.
                if not os.path.isdir(current_directory):
                    continue

                # The subdirectory should contain the information instead if it is set.
                if self.sub_folder is not None:
                    current_directory = os.path.join(self.mask_directory, current_sample_name)

                    if not os.path.isdir(current_directory):
                        continue

                # Make a copy of the object and update the directory and sample name.
                new_object = copy.deepcopy(self)
                new_object.mask_directory = current_directory
                new_object.sample_name = current_sample_name
                new_object.sub_folder = None

                if new_object._directory_contains_mask():
                    # Identified a good candidate!
                    any_sample_name_matches_directory = True

                    yield new_object

        if not any_sample_name_matches_directory:
            # Here the idea is to iterate over all contents and see if any folders contain the expected content.

            dir_contents = os.listdir(self.mask_directory)
            for current_content in dir_contents:

                # Check if the current content is a directory.
                if not os.path.isdir(os.path.join(self.mask_directory, current_content)):
                    continue

                # Set subdirectory
                if self.sub_folder is not None:
                    current_directory = os.path.join(self.mask_directory, current_content, self.sub_folder)
                else:
                    current_directory = os.path.join(self.mask_directory, current_content)

                if not os.path.isdir(current_directory):
                    continue

                # Create an object to test.
                new_object = copy.deepcopy(self)
                new_object.mask_directory = current_directory
                new_object.sub_folder = None

                # Suggest sample name.
                new_object.sample_name = None
                if sample_name is not None:
                    # We will replace sample name in the calling function, because we will need to check some other
                    # stuff there as well.
                    new_object.sample_name = None

                if new_object._directory_contains_mask():
                    # Identified a good candidate!

                    yield new_object

    def check(self, raise_error=False):

        if not isinstance(self.mask_directory, str):
            if raise_error:
                raise TypeError("A directory containing masks was expected. Found: an object that was not a path to a "
                                "directory.")

            return False

        if not os.path.isdir(self.mask_directory):
            if raise_error:
                raise TypeError("A directory containing masks was expected. Found: a path that was not a path to an "
                                "existing directory.")

            return False

        # Generate paths to directories.
        valid_directories = [self._generate_sub_directories()]

        # Check that valid directories were found.
        if len(valid_directories) == 0:
            if raise_error:
                raise ValueError("A directory containing masks was expected. Found: no valid directories with masks.")

            return False

        # TODO: These checks should be moved externally to prevent issues with merging multiple directory paths.

        # Check the number of sample names and the number of valid directories.
        sample_name = copy.deepcopy(self.sample_name)
        if sample_name is not None and not isinstance(sample_name, list):
            sample_name = [sample_name]

        if sample_name is not None and not self.contain_multiple_samples:
            if not len(sample_name) == len(valid_directories):
                if raise_error:
                    raise ValueError("The number of sample names does not match the number of directories with masks.")

                return False

        return True

    def create_masks(self):

        # Create list of mask objects from valid subdirectories.
        mask_list = [list(valid_directory._create_masks()) for valid_directory in self._generate_sub_directories()]

        # Flatten list.
        return list(chain.from_iterable(mask_list))

    def _create_masks(self):
        # Get all contents in the directory.
        dir_contents = os.listdir(self.mask_directory)

        # Check file extensions.
        file_extensions: List[str] = supported_file_types(file_type=self.file_type)

        for current_item in dir_contents:

            # Skip  items that are directories
            if os.path.isdir(os.path.join(self.mask_directory, current_item)):
                continue

            # Skip files that do not have the right extension.
            if not any(current_item.lower().endswith(ii) for ii in file_extensions):
                continue

            mask_file = MaskFile(
                file_path=os.path.join(self.mask_directory, current_item),
                sample_name=self.sample_name,
                modality=self.modality,
                file_type=self.file_type).create()

            if not mask_file.check(raise_error=False):
                continue

            yield mask_file
