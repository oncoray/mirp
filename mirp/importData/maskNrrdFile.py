from typing import Union

from mirp.importData.maskGenericFile import MaskFile


class MaskNrrdFile(MaskFile):
    def __init__(
            self,
            file_path: Union[None, str] = None,
            dir_path: Union[None, str] = None,
            sample_name: Union[None, str] = None,
            file_name: Union[None, str] = None,
            mask_name: Union[None, str] = None,
            modality: Union[None, str] = None,
            file_type: Union[None, str] = None):

        super().__init__(
            file_path=file_path,
            dir_path=dir_path,
            sample_name=sample_name,
            file_name=file_name,
            mask_name=mask_name,
            modality=modality,
            file_type=file_type)
