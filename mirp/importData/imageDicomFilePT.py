import numpy as np

from typing import Union, Tuple, List

from mirp.imageSUV import SUVscalingObj
from mirp.importData.imageDicomFile import ImageDicomFile


class ImageDicomFilePT(ImageDicomFile):
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

        super().__init__(
            file_path=file_path,
            dir_path=dir_path,
            sample_name=sample_name,
            file_name=file_name,
            image_name=image_name,
            image_modality=image_modality,
            image_file_type=image_file_type,
            image_data=image_data,
            image_origin=image_origin,
            image_orientation=image_orientation,
            image_spacing=image_spacing,
            image_dimensions=image_dimensions
        )

    def is_stackable(self, stack_images: str):
        return True

    def create(self):
        return self

    def load_data(self, **kwargs):
        image_data = self.load_data_generic()

        # TODO: integrate SUV computations locally.
        suv_conversion_object = SUVscalingObj(dcm=self.image_metadata)
        scale_factor = suv_conversion_object.get_scale_factor(suv_normalisation="bw")

        # Convert to SUV
        image_data *= scale_factor

        # Update relevant tags in the metadata
        self.image_metadata = suv_conversion_object.update_dicom_header(dcm=self.image_metadata)
