import copy

from typing import List

from mirp.importData.imageNumpyFile import ImageNumpyFile
from mirp.importData.imageGenericFileStack import ImageFileStack


class ImageNumpyFileStack(ImageFileStack):

    def __init__(
            self,
            image_file_objects: List[ImageNumpyFile],
            **kwargs
    ):
        super().__init__(image_file_objects, **kwargs)

    def complete(self, remove_metadata=True, force=False):
        """
        Fills out missing attributes in an image stack. Numpy files are actually barebones, and compared to DICOM
        and many ITK formats lacks the origin. This forces us to determine stacking by the file name.
        :param remove_metadata: Whether metadata should be removed after completing information.
        :param force: Whether attributes are forced to update or not.
        :return: nothing, attributes are updated in place.
        """
        # Load metadata of every slice.
        self.load_metadata()

        self._complete_modality()
        self._complete_sample_name()

        # Sort by filename.
        self.sort_image_objects_by_file()

        image_object = copy.deepcopy(self.image_file_objects[0])
        image_object.complete(remove_metadata=False)

        if self.image_origin is None:
            self.image_origin = image_object.image_origin

        if self.image_spacing is None:
            self.image_spacing = image_object.image_spacing

        if self.image_orientation is None:
            self.image_orientation = image_object.image_orientation

        if self.image_dimension is None:
            image_dimension = image_object.image_dimension
            if len(image_dimension) == 3:
                self.image_dimension = tuple([
                    len(self.image_file_objects), image_object.image_dimension[1], image_object.image_dimension[2]
                ])
            elif len(image_dimension) == 2:
                self.image_dimension = tuple([
                    len(self.image_file_objects), image_object.image_dimension[0], image_object.image_dimension[1]
                ])
            else:
                self.image_dimension = tuple([len(self.image_file_objects), 1, image_object.image_dimension[0]])
