import os
import os.path
import copy

from typing import Union, List
from itertools import chain
from mirp.importData.utilities import supported_file_types
from mirp.importData.importImageFile import ImageFile


class ImageDirectory:

    def __init__(self,
                 directory,
                 sample_name: Union[None, str, List[str]] = None,
                 image_name: Union[None, str] = None,
                 sub_folder: Union[None, str] = None,
                 modality: Union[None, str] = None,
                 file_type: Union[None, str] = None,
                 contain_multiple_samples: [bool] = False):

        self.image_directory = directory
        self.sample_name = sample_name
        self.suggested_sample_name = None
        self.image_name = image_name
        self.sub_folder = sub_folder
        self.modality = modality
        self.file_type: Union[None, str] = file_type
        self.contain_multiple_samples = contain_multiple_samples

    def _directory_contains_images(self):

        # The _create_images generator only returns something if a valid image can be found in the indicated directory.
        # When that happens, return True, otherwise return False.
        for _ in self._create_images():
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
                current_directory = os.path.join(self.image_directory, self.sub_folder)
            else:
                current_directory = os.path.join(self.image_directory)

            if os.path.isdir(self.image_directory):
                new_object = copy.deepcopy(self)
                new_object.image_directory = current_directory
                new_object.sample_name = sample_name
                new_object.sub_folder = None

                if new_object._directory_contains_images():
                    yield new_object

        # Add check to see if any sample names matches a sub_directory.
        any_sample_name_matches_directory = False

        # Try is sample names match directory names.
        if self.sample_name is not None:
            for current_sample_name in sample_name:
                current_directory = os.path.join(self.image_directory, current_sample_name)

                # Check that the directory actually exists.
                if not os.path.isdir(current_directory):
                    continue

                # The subdirectory should contain the information instead if it is set.
                if self.sub_folder is not None:
                    current_directory = os.path.join(self.image_directory, current_sample_name)

                    if not os.path.isdir(current_directory):
                        continue

                # Make a copy of the object and update the directory and sample name.
                new_object = copy.deepcopy(self)
                new_object.image_directory = current_directory
                new_object.sample_name = current_sample_name
                new_object.sub_folder = None

                if new_object._directory_contains_images():
                    # Identified a good candidate!
                    any_sample_name_matches_directory = True

                    yield new_object

        if not any_sample_name_matches_directory:
            # Here the idea is to iterate over all contents and see if any folders contain the expected content.

            dir_contents = os.listdir(self.image_directory)
            for current_content in dir_contents:

                # Check if the current content is a directory.
                if not os.path.isdir(os.path.join(self.image_directory, current_content)):
                    continue

                # Set subdirectory
                if self.sub_folder is not None:
                    current_directory = os.path.join(self.image_directory, current_content, self.sub_folder)
                else:
                    current_directory = os.path.join(self.image_directory, current_content)

                if not os.path.isdir(current_directory):
                    continue

                # Create an object to test.
                new_object = copy.deepcopy(self)
                new_object.image_directory = current_directory
                new_object.sub_folder = None

                # Suggest sample name.
                new_object.suggested_sample_name = current_content
                new_object.sample_name = None
                if sample_name is not None:
                    # We will replace sample name in the calling function, because we will need to check some other
                    # stuff there as well.
                    new_object.sample_name = None

                if new_object._directory_contains_images():
                    # Identified a good candidate!

                    yield new_object

    def check(self, raise_error=False):

        if not isinstance(self.image_directory, str):
            if raise_error:
                raise TypeError("A directory containing images was expected. Found: an object that was not a path to a "
                                "directory.")

            return False

        if not os.path.isdir(self.image_directory):
            if raise_error:
                raise TypeError("A directory containing images was expected. Found: a path that was not a path to an "
                                "existing directory.")

            return False

        # Generate paths to directories.
        valid_directories = [self._generate_sub_directories()]

        # Check that valid directories were found.
        if len(valid_directories) == 0:
            if raise_error:
                raise ValueError("A directory containing images was expected. Found: no valid directories with images.")

            return False

        # TODO: These checks should be moved externally to prevent issues with merging multiple directory paths.

        # Check the number of sample names and the number of valid directories.
        sample_name = copy.deepcopy(self.sample_name)
        if sample_name is not None and not isinstance(sample_name, list):
            sample_name = [sample_name]

        if sample_name is not None and not self.contain_multiple_samples:
            if not len(sample_name) == len(valid_directories):
                if raise_error:
                    raise ValueError("The number of sample names does not match the number of directories with images.")

                return False

        return True

    def create_images(self):

        # Create list of image objects from valid subdirectories.
        image_list = [list(valid_directory._create_images()) for valid_directory in self._generate_sub_directories()]

        # Flatten list.
        return list(chain.from_iterable(image_list))

    def _create_images(self):
        # Get all contents in the directory.
        dir_contents = os.listdir(self.image_directory)

        # Check file extensions.
        file_extensions: List[str] = supported_file_types(file_type=self.file_type)

        for current_item in dir_contents:

            # Skip  items that are directories
            if os.path.isdir(os.path.join(self.image_directory, current_item)):
                continue

            # Skip files that do not have the right extension.
            if not any(current_item.lower().endswith(ii) for ii in file_extensions):
                continue

            image_file = ImageFile(file_path=os.path.join(self.image_directory, current_item),
                                   sample_name=self.sample_name,
                                   suggested_sample_name=self.suggested_sample_name,
                                   modality=self.modality,
                                   file_type=self.file_type).create()

            if not image_file.check(raise_error=False):
                continue

            yield image_file
