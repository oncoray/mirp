import numpy as np
from mirp.images.genericImage import GenericImage


class CTImage(GenericImage):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_image_data(self):
        if self.image_data is None or self.normalised:
            return

        # Ensure that CT values are Hounsfield units.
        self.image_data = np.round(self.image_data)