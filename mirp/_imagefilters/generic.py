from mirp.settings.generic import SettingsClass
from mirp._images.generic_image import GenericImage


class GenericFilter:

    def __init__(self, image: GenericImage, settings: SettingsClass, name: str):
        # In-slice (2D) or 3D filtering
        self.separate_slices = image.separate_slices

        # Even though most currently implemented filters are IBSI-compliant, set value to False to avoid surprises in
        # the future.
        self.ibsi_compliant: bool = False
        self.ibsi_id: None | str = None

    def generate_object(self):
        raise NotImplementedError("_generate_object method should be defined in the subclasses")

    def transform(self, image: GenericImage):
        raise NotImplementedError("transform method should be defined in the subclasses.")

    def check_isotropic_image(self, image: GenericImage):
        import warnings

        # Check if isotropic requirements for the filter are fulfilled.
        if not image.is_isotropic(axis_only=self.separate_slices):
            warnings.warn(
                self._not_isotropic_warning_message(),
                UserWarning
            )

    def _not_isotropic_warning_message(self):
        raise NotImplementedError("_not_isotropic_warning_message method should be defined in the subclasses")
