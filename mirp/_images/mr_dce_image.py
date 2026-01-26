from mirp._images.generic_image import GenericImage


class MRDCEImage(GenericImage):
    # Diffusion contrast-enhanced MR consists of multiple types of (semi-)quantitative parametric maps. These maps all
    # quantify different characteristics, some of which may contain negative numbers.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.calibrated_units = True

    def normalise_intensities(
            self,
            normalisation_method: None | str = "none",
            **kwargs
    ):
        """
        Normalise intensities. NOTE: this changes the class of the object from MRDCEImage to GenericImage as
        normalisation breaks the one-to-one relationship between intensities in the DCE parameteric map.
        """
        if self.image_data is None:
            return self

        if normalisation_method is None or normalisation_method == "none":
            return self

        new_image = GenericImage(image_data=self.image_data)
        new_image.update_from_template(template=self)
        new_image = new_image.normalise_intensities(
            normalisation_method=normalisation_method,
            **kwargs
        )

        return new_image

    def scale_intensities(self, scale: float):

        if self.image_data is None:
            return self

        if scale == 1.0:
            return self

        # Scaling intensities changes the object class from CTImage to MRDCEImage as this breaks the one-to-one
        # relationship between intensities in the DCE parameteric map.
        new_image = GenericImage(image_data=self.image_data)
        new_image.update_from_template(template=self)
        new_image = new_image.scale_intensities(scale=scale)

        return new_image