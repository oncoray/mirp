import numpy as np
from typing import Optional, Tuple, Any
from mirp.images.genericImage import GenericImage


class RTDoseImage(GenericImage):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_default_lowest_intensity():
        return 0.0

    def update_image_data(self):
        if self.image_data is None:
            return

        # Ensure that dose values are not negative.
        self.image_data[self.image_data < 0.0] = 0.0

    def normalise_intensities(
            self,
            normalisation_method: Optional[str] = "none",
            intensity_range: Optional[Tuple[Any, Any]] = None,
            saturation_range: Optional[Tuple[Any, Any]] = None,
            mask: Optional[np.ndarray] = None
    ):
        """
        Normalise intensities. NOTE: this changes the class of the object from RTDose to GenericImage as
        normalisation breaks the one-to-one relationship between intensities and dose.
        """
        image = super().normalise_intensities(
            normalisation_method=normalisation_method,
            intensity_range=intensity_range,
            saturation_range=saturation_range,
            mask=mask
        )

        if image.image_data is None:
            return self

        if normalisation_method is None or normalisation_method == "none":
            return self

        new_image = GenericImage(image_data=image.image_data)
        new_image.update_from_template(template=image)

        return new_image
