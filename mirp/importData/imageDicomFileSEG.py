from mirp.importData.imageDicomFile import MaskDicomFile


class MaskDicomFileSEG(MaskDicomFile):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        ...
