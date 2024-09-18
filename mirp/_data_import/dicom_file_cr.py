from mirp._data_import.dicom_planar_image import ImageDicomPlanarImage


class ImageDicomFileCR(ImageDicomPlanarImage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)