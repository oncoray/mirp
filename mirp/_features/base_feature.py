from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask


class Feature(object):

    def __init__(self, **kwargs):
        self.name: None | str = None
        self.abbr_name: None | str = None
        self.ibsi_id: None | str = None
        self.value: None | float = None

        # Even though most features are IBSI-compliant, set value to False to avoid surprises in the future.
        self.ibsi_compliant: None | bool = False

    def clear_cache(self):
        pass

    def compute(self, image: GenericImage, mask: BaseMask):
        raise NotImplementedError("compute method should be implemented in subclasses.")
