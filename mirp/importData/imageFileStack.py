from typing import List, Union

import pandas as pd
import numpy as np

from mirp.importData.importImage import ImageFile
from mirp.importData.imageDicomFile import ImageDicomFile


class ImageFileStack:

    def __init__(
            self,
            image_list: Union[List[ImageDicomFile], List[ImageFile]]):

        self.image_list = image_list
        self.sample_name = image_list[0].sample_name
        self.modality = image_list[0].modality

        if isinstance(image_list[0], ImageDicomFile):
            self.frame_of_reference_uid = image_list[0].frame_of_reference_uid

        else:
            self.frame_of_reference_uid = None

    def split(self):
        # Extract identifiers
        image_id_table = pd.concat([image_file.get_identifiers(style="extended")
                                    for image_file in self.image_list], ignore_index=True)

        # Check that all names are set.
        if any(image_id_table.sample_name.isnan()):

            # Attempt to replace missing names by
            ...

    def set_sample_name(self,
                        sample_name: str):
        self.sample_name = sample_name

        # Update image list explicitly.
        if self.image_list is not None:
            self.image_list = [image_file.set_sample_name(sample_name) for image_file in self.image_list]

    def check(self, raise_error=False):

        id_table = pd.concat([image_file.get_identifiers(style="extended")
                              for image_file in self.image_list], ignore_index=True)

        if len(id_table.drop_duplicates()) > 1:
            if raise_error:
                raise ValueError(f"The stack contains image files with different characteristics.")

            return False

        # Check that none of the basic identifiers has information missing.
        if any(image_file.sample_name is None for image_file in self.image_list):
            if raise_error:
                raise ValueError(f"The stack contains image files without a sample name.")

            return False

        if any(image_file.modality for image_file in self.image_list):
            if raise_error:
                raise ValueError(f"The stack contains image files without a modality.")

            return False

        return True
