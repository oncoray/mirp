from mirp._images.generic_image import GenericImage


class DXImage(GenericImage):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_separate_slices(self, x: bool):
        # Planar images should always separate slices.
        self.separate_slices = True


class CRImage(DXImage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MGImage(DXImage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
