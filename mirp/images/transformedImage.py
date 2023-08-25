from mirp.images.genericImage import GenericImage


class TransformedImage(GenericImage):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
