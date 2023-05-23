from typing import Union, List

from mirp.importData.imageGenericFile import ImageFile
from mirp.importData.imageDicomFile import ImageDicomFile
from mirp.importData.imageITKFile import ImageITKFile
from mirp.importData.imageNumpyFile import ImageNumpyFile
from mirp.importData.utilities import supported_file_types


class ImageFileStack(ImageFile):
    def is_stackable(self, stack_images: str):
        return False

    def _complete_image_origin(self, force=False):
        ...

    def _complete_image_orientation(self, force=False):
        ...

    def _complete_image_spacing(self, force=False):
        ...

    def _complete_image_dimensions(self, force=False):
        ...

    def __init__(
            self,
            image_file_objects: Union[List[ImageFile], List[ImageDicomFile], List[ImageITKFile], List[ImageNumpyFile]],
            dir_path: Union[None, str] = None,
            sample_name: Union[None, str] = None,
            image_name: Union[None, str, List[str]] = None,
            image_modality: Union[None, str] = None,
            image_file_type: Union[None, str] = None,
            **kwargs):

        if dir_path is None:
            dir_path = image_file_objects[0].dir_path

        if sample_name is None:
            sample_name = image_file_objects[0].sample_name

        if image_name is None:
            image_name = image_file_objects[0].image_name

        if image_modality is None:
            image_modality = image_file_objects[0].modality

        if image_file_type is None:
            image_file_type = image_file_objects[0].file_type

        # Aspects regarding the image itself are set based on the stack itself.
        super().__init__(
            file_path=None,
            dir_path=dir_path,
            sample_name=sample_name,
            file_name=None,
            image_name=image_name,
            image_modality=image_modality,
            image_file_type=image_file_type,
            image_data=None,
            image_origin=None,
            image_orientation=None,
            image_spacing=None,
            image_dimensions=None
        )

        self.image_file_objects = image_file_objects
        self.slice_positions: Union[None, List[float]] = None

    def create(self):
        # Import locally to avoid potential circular references.
        from mirp.importData.imageDicomFileStack import ImageDicomFileStack
        from mirp.importData.imageITKFileStack import ImageITKFileStack
        from mirp.importData.imageNumpyFileStack import ImageNumpyFileStack

        file_extensions = supported_file_types(file_type=self.file_type)

        if any(self.file_path.lower().endswith(ii) for ii in file_extensions) and \
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("dicom")):
            # Create DICOM-specific file.
            image_file_stack = ImageDicomFileStack(
                image_file_objects=self.image_file_objects,
                dir_path=self.dir_path,
                sample_name=self.sample_name,
                image_name=self.image_name,
                image_modality=self.modality,
                image_file_type="dicom"
            )

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions):
            if any(self.file_path.lower().endswith(ii) for ii in supported_file_types("nifti")):
                file_type = "nifti"
            elif any(self.file_path.lower().endswith(ii) for ii in supported_file_types("nrrd")):
                file_type = "nrrd"
            else:
                raise ValueError(f"DEV: specify file_type")

            # Create ITK file.
            image_file_stack = ImageITKFileStack(
                image_file_objects=self.image_file_objects,
                dir_path=self.dir_path,
                sample_name=self.sample_name,
                image_name=self.image_name,
                image_modality=self.modality,
                image_file_type=file_type
            )

        elif any(self.file_path.lower().endswith(ii) for ii in file_extensions) and\
                any(self.file_path.lower().endswith(ii) for ii in supported_file_types("numpy")):

            # Create Numpy file.
            image_file_stack = ImageNumpyFileStack(
                image_file_objects=self.image_file_objects,
                dir_path=self.dir_path,
                sample_name=self.sample_name,
                image_name=self.image_name,
                image_modality=self.modality,
                image_file_type="numpy"
            )

        else:
            raise NotImplementedError(f"The provided image type is not implemented: {self.file_type}")

        return image_file_stack

    def complete(self, remove_metadata=True, force=False):
        raise NotImplementedError(
            f"DEV: There is (intentionally) no generic implementation of complete. Please specify "
            f"implementation for subclasses."
        )

    def load_metadata(self):
        # Load metadata for underlying files in the order indicated by self.image_file_objects.
        for image_file_object in self.image_file_objects:
            image_file_object.load_metadata()

    def load_data(self):
        # Load data for underlying files in the order indicated by self.image_file_objects.
        for image_file_object in self.image_file_objects:
            image_file_object.load_data()
