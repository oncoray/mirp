import numpy as np
from mirp.images.genericImage import GenericImage


class PETImage(GenericImage):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_image_data(self):
        if self.image_data is None or self.normalised:
            return

        # Ensure that PET values are not negative.
        self.image_data = self.image_data[self.image_data < 0.0] = 0.0
