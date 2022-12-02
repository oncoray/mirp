from typing import Union, Dict

from mirp.importData.importImageFile import ImageFile


class ImageNrrdFile(ImageFile):
    def __init__(self, file_path: str,
                 file_type: str,
                 sample_name: Union[None, str] = None,
                 suggested_sample_name: Union[None, str, dict[str, str, str]] = None,
                 image_name: Union[None, str] = None,
                 modality: Union[None, str] = None):
        super().__init__(file_path=file_path,
                         sample_name=sample_name,
                         suggested_sample_name=suggested_sample_name,
                         image_name=image_name,
                         modality=modality,
                         file_type=file_type)

