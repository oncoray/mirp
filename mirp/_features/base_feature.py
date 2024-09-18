from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask


class Feature(object):

    def __init__(self, **kwargs):
        self.name: None | str = None
        self.abbr_name: None | str = None
        self.ibsi_id: None | str = None
        self.value: None | float = None
        self.table_name: None | str = None

        # Even though most features are IBSI-compliant, set value to False to avoid surprises in the future.
        self.ibsi_compliant: bool = False

    def clear_cache(self):
        pass

    def clear_local_cache(self, other):
        pass

    def compute(self, image: GenericImage, mask: BaseMask):
        raise NotImplementedError("compute method should be implemented in subclasses.")

    def create_table_name(self):
        self.table_name = "_".join(self._get_base_table_name_element())

    def update_ibsi_compliance(self):
        pass

    def is_ibsi_compliant(self, image: GenericImage) -> bool:
        # Update IBSI compliance based on attributes.
        self.update_ibsi_compliance()

        # Feature is compliant if it is compliant, and is derived from a image processed in an IBSI-compliant manner.
        return self.ibsi_compliant and image.ibsi_compliant

    def _get_base_table_name_element(self) -> list[str | None]:
        return [self.abbr_name]
