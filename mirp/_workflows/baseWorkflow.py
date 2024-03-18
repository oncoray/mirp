from mirp._data_import.imageGenericFile import ImageFile


class BaseWorkflow:
    def __init__(
            self,
            image_file: ImageFile,
            write_dir: None | str = None,
            **kwargs
    ):
        super().__init__()
        self.image_file = image_file
        self.write_dir = write_dir
