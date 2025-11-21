from mirp._images.generic_image import GenericImage


class TransformedImage(GenericImage):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    @staticmethod
    def get_default_ivh_discretisation_method():
        return "fixed_bin_number"
