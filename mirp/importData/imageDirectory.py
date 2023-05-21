import os
import os.path
import copy

from typing import Union, List
from itertools import chain
from mirp.importData.utilities import supported_file_types, dir_structure_contains_directory, match_file_name
from mirp.importData.imageGenericFile import ImageFile
from mirp.importData.imageGenericFileStack import ImageFileStack


class ImageDirectory:

    def __init__(
            self,
            directory,
            sample_name: Union[None, str, List[str]] = None,
            image_name: Union[None, str] = None,
            image_sub_folder: Union[None, str] = None,
            image_modality: Union[None, str] = None,
            image_file_type: Union[None, str] = None,
            stack_images: str = "auto",
            **kwargs):

        if sample_name is not None and not isinstance(sample_name, list):
            sample_name = [sample_name]

        self.image_directory = directory
        self.sample_name: Union[None, str, List[str]] = sample_name
        self.image_name = image_name
        self.sub_folder = image_sub_folder
        self.modality = image_modality
        self.file_type = image_file_type
        self.stack_images = stack_images

        # image_files are set using create_images.
        self.image_files: Union[None, List[str]] = None

    def check(self, raise_error=False):

        if not isinstance(self.image_directory, str):
            if raise_error:
                raise TypeError(
                    f"A directory containing images was expected. Found: an object that was not a path to a "
                    f"directory.")

            return False

        if not os.path.isdir(self.image_directory):
            if raise_error:
                raise TypeError(
                    f"A directory containing images was expected. Found: a path that was not a path to an existing "
                    f"directory: {self.image_directory}")

            return False

        return True

    def create_images(self):

        # Find potentially valid directory paths.
        path_info = list(os.walk(self.image_directory))

        if len(path_info) == 0:
            ValueError(f"The {self.image_directory} directory is empty, and no images could be found.")

        # Find entries that have associated files.
        path_info = [
            path_info_element for path_info_element in path_info if len(path_info_element[2]) > 0
        ]

        if len(path_info) == 0:
            ValueError(
                f"All directories within the {self.image_directory} directory is empty, and no images could be found.")

        # Find entries where the folder structure matches the sub_folder.
        if self.sub_folder is not None:
            path_info = [
                path_info_element for path_info_element in path_info
                if path_info_element[0].endswith(self.sub_folder)
            ]

            if len(path_info) == 0:
                raise ValueError(
                    f"No directories where found in {self.image_directory} that contained the directory "
                    f"substructure {self.sub_folder}.")

        # Add in sample name placeholder to path-information.
        path_info: List = [list(path_info_element).append(None) for path_info_element in path_info]

        # Find entries where the folder structure contains a sample name. All the sample names must be present to
        # avoid incidental findings.
        if self.sample_name is not None:
            all_samples_selected = True
            sample_name_matches = []
            ignore_dirs = [self.image_directory]
            if self.sub_folder is not None:
                ignore_dirs += [self.sub_folder]

            for sample_name in self.sample_name:
                current_sample_matches = [
                    ii for ii in range(len(path_info))
                    if dir_structure_contains_directory(path_info[ii][0], pattern=sample_name, ignore_dir=ignore_dirs)
                ]

                if len(current_sample_matches) > 0:
                    sample_name_matches += current_sample_matches

                else:
                    all_samples_selected = False
                    break

            # Only filter list if all sample names are uniquely part of their respective paths.
            if all_samples_selected and len(set(sample_name_matches)) == len(sample_name_matches) and len(
                    sample_name_matches) > 0:
                path_info = [path_info[ii] for ii in sample_name_matches]

                # Add suggested sample names.
                for ii in range(len(path_info)):
                    path_info[ii][3] = sample_name_matches[ii]

        # Find entries that include files of the right file-type. First, we keep only those files that are of the
        # correct file type. Then we filter out entries where no files remain. Note that if file_type is not externally
        # set, all supported image file types are considered.
        allowed_file_extensions = supported_file_types(file_type=self.file_type)

        for ii, path_info_element in enumerate(path_info):
            if len(path_info_element[2]) > 0:
                path_info[ii][2] = [
                    image_file for image_file in path_info_element[2]
                    if image_file.endswith(tuple(allowed_file_extensions))
                ]

        # Find entries that still contain associated files of the right type.
        path_info = [path_info_element for path_info_element in path_info if len(path_info_element[2]) > 0]

        if len(path_info) == 0:
            ValueError(
                f"The {self.image_directory} directory (and its subdirectories) do not contain any supported "
                f"image files ({', '.join(allowed_file_extensions)})."
            )

        # Find entries that contain file names that match the image name. First, we keep only those files that contain
        # the file pattern, and then we remove empty directories.
        if self.image_name is not None:
            for ii, path_info_element in enumerate(path_info):
                if len(path_info_element[2]) > 0:
                    path_info[ii][2] = [
                        image_file for image_file in path_info_element[2]
                        if match_file_name(image_file, pattern=self.image_name, file_extension=allowed_file_extensions)
                    ]
            # Find entries that still contain associated image files with the correct name.
            path_info = [path_info_element for path_info_element in path_info if len(path_info_element[2]) > 0]

            if len(path_info) == 0:
                ValueError(
                    f"The {self.image_directory} directory (and its subdirectories) do not contain any supported "
                    f"image files ({', '.join(allowed_file_extensions)}) that contain the name pattern "
                    f"({', '.join(self.image_name)})."
                )

        # Read and parse image content in subdirectories.
        image_list = []

        for path_info_element in path_info:
            # Make a copy of the object and update the directory and sample name.
            image_sub_directory = copy.deepcopy(self)
            image_sub_directory.image_directory = path_info_element[0]
            image_sub_directory.sample_name = path_info_element[3]
            image_sub_directory.sub_folder = None
            image_sub_directory.image_files = path_info_element[2]

            image_sub_directory._create_images()

        # Flatten list.
        self.image_files = list(chain.from_iterable(image_list))

        # Try to stack.
        self.autostack()

    def _create_images(self) -> List[ImageFile]:
        """
        Create image objects from files in the directory.

        :return: list of image file objects.
        """

        # Two initial possibilities: the image_files attribute is None, or is set. In case it is None, we need to
        # filter directory contents based on file type and name. Afterwards, the same routine is used.
        if self.image_files is None:

            image_file_list = []

            # Get all contents in the directory, and filter by file type and name.
            dir_contents = os.listdir(self.image_directory)
            allowed_file_extensions = supported_file_types(file_type=self.file_type)

            for current_item in dir_contents:

                # Skip  items that are directories
                if os.path.isdir(os.path.join(self.image_directory, current_item)):
                    continue

                # Skip files that do not have the right extension.
                if not current_item.lower().endswith(tuple(allowed_file_extensions)):
                    continue

                if self.image_name is not None:
                    # Skip files that do not contain the image_name pattern.
                    if not match_file_name(current_item, pattern=self.image_name, file_extension=allowed_file_extensions):
                        continue

                image_file_list.append(current_item)

            self.image_files = image_file_list

        # From here, the routine is completely the same.
        if len(self.image_files) == 0:
            return []

        image_file_list = []
        for image_file_name in self.image_files:

            # Create image file object.
            image_file = ImageFile(
                file_path=os.path.join(self.image_directory, image_file_name),
                sample_name=self.sample_name,
                image_modality=self.modality,
                image_file_type=self.file_type).create()

            if not image_file.check(raise_error=False):
                continue

            image_file_list.append(image_file)

        return image_file_list

    def autostack(self):
        """
        Form image stacks by combining image file objects of the same class that share identifiers.
        """

        if self.image_files is None or len(self.image_files) == 0:
            pass

        # Isolate all cases that are not stackable anyway.
        image_file_list: List[ImageFile] = [
            image_file_object for image_file_object in self.image_files
            if not image_file_object.is_stackable()
        ]

        # If none of the images are potentially stackable we don't make any alterations.
        if len(image_file_list) == len(self.image_files):
            pass

        image_file_list += list(self._autostack())

        self.image_files = image_file_list

    def _autostack(self):

        # Find stackable objects.
        stackable_image_file_list: List[ImageFile] = [
            image_file_object for image_file_object in self.image_files
            if image_file_object.is_stackable()
        ]

        # Hash identifier data: those files that might be stackable should have the same hash.
        identifier_list = [
            image_file_object.get_identifiers(as_hash=True) for image_file_object in stackable_image_file_list
        ]

        # Create mask to avoid revisiting objects.
        available_objects = [True] * len(identifier_list)

        while any(available_objects):
            # Find next unused identifier.
            identifier = next((
                identifier for ii, identifier in enumerate(identifier_list)
                if available_objects[ii] is True
            ))

            # Identify image file objects that have the same identifier.
            stack_list = [
                image_file_object for ii, image_file_object in enumerate(stackable_image_file_list)
                if available_objects[ii] is True and identifier_list[ii] == identifier
            ]

            # Update availability.
            available_objects = [
                False for ii in range(len(available_objects))
                if available_objects is True and identifier_list[ii] == identifier
            ]

            # If there is only one object to stack, it is not necessary to stack the object.
            if len(stack_list) == 1:
                yield stack_list[0]

            else:
                yield ImageFileStack(image_file_objects=stack_list).create().complete()
