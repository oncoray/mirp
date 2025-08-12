import numpy as np

from mirp._data_import.dicom_file import ImageDicomFile
from mirp._data_import.dicom_file_stack import ImageDicomFileStack

class ImageDicomFileVolumeStack(ImageDicomFileStack):

    def __init__(
            self,
            image_file_objects: list[ImageDicomFile],
            **kwargs
    ):
        super().__init__(image_file_objects, **kwargs)

    def complete(self, remove_metadata=False, force=False):
        """
        Fills out missing attributes in an image stack. Image parameters in DICOM stacks, by design,
        are fully determined by the origin of all slices in the stack. This method then sorts the image file objects
        by origin, and uses their relative positions to determine slice spacing and the orientation vector.
        :param remove_metadata: Whether metadata (DICOM headers) should be removed after completing information.
        :param force: Whether attributes are forced to update or not.
        :return: nothing, attributes are updated in place.
        """

        # Load metadata of every slice.
        self.load_metadata(limited=True)

        # Complete metadata of underlying files.
        for image_file_object in self.image_file_objects:
            image_file_object.complete(remove_metadata = False)

        self._complete_modality()
        self._complete_sample_name()

        # For image-related aspects, use the first file as a template.
        if self.image_orientation is None:
            self.image_orientation = self.image_file_objects[0].image_orientation
        if self.image_origin is None:
            self.image_origin = self.image_file_objects[0].image_origin
        if self.image_spacing is None:
            self.image_spacing = self.image_file_objects[0].image_spacing
        if self.image_dimension is None:
            self.image_dimension = self.image_file_objects[0].image_dimension

        # Check if the complete data passes verification.
        self.check(raise_error=True, remove_metadata=False)

        if remove_metadata:
            self.remove_metadata()

    def stack_slices(self):
        if self.image_data is not None:
            return

        image = np.zeros(self.image_dimension, dtype=np.float32)
        for image_file in self.image_file_objects:
            if image_file.image_data is None:
                raise ValueError(
                    "DEV: the image_data attribute of underlying image files are not set. Please call load_data first."
                )
            image += image_file.image_data.astype(np.float32)

        self.image_data = image