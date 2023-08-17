import copy
import numpy as np
from typing import Optional

from mirp.images.baseImage import BaseImage


class GenericImage(BaseImage):

    def __init__(
            self,
            image_data: Optional[np.ndarray],
            **kwargs
    ):

        super().__init__(**kwargs)

        self.image_data = image_data

    def copy(self, drop_image=False):
        image = copy.deepcopy(self)

        if drop_image:
            image.drop_image()

        return image

    def drop_image(self):
        self.image_data = None
    