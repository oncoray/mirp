import numpy as np
import copy
from typing import Optional

from mirp.images.genericImage import GenericImage
from mirp.images.maskImage import MaskImage
from mirp.importSettings import SettingsClass


class BaseMask:
    def __init__(
            self,
            roi_name: str,
            **kwargs
    ):
        # Make cooperative.
        super().__init__()

        # Set region of interest.
        self.roi = MaskImage(**kwargs)

        # Define other types of masks.
        self.roi_intensity: Optional[MaskImage] = None
        self.roi_morphology: Optional[MaskImage] = None

        # Set name of the mask.
        self.roi_name = roi_name

    def copy(self, drop_image=False):

        # Create new mask by copying the current mask.
        mask = copy.deepcopy(self)

        if drop_image:
            mask.roi.drop_image()
            if mask.roi_intensity is not None:
                mask.roi_intensity.drop_image()
            if mask.roi_morphology is not None:
                mask.roi_morphology.drop_image()

        # Creates a new copy of the roi
        return mask

    def is_empty(self):
        if self.roi is None:
            return True

        return self.roi.is_empty()

    def interpolate(
            self,
            image: Optional[GenericImage],
            settings: SettingsClass):
        # Skip if image and/or mask is missing
        if self.is_empty():
            return

        if image is None or image.is_empty():
            self.roi.interpolate(settings=settings)
            if self.roi_intensity is not None:
                self.roi_intensity.interpolate(settings=settings)
            if self.roi_morphology is not None:
                self.roi_morphology.interpolate(settings=settings)
        else:
            self.register(image=image, settings=settings)

    def register(
            self,
            image: GenericImage,
            settings: SettingsClass
    ):
        if self.is_empty():
            return

        self.roi.register(image=image, settings=settings)
        if self.roi_intensity is not None:
            self.roi_intensity.register(image=image, settings=settings)
        if self.roi_morphology is not None:
            self.roi_morphology.register(image=image, settings=settings)

    def generate_masks(self):
        """"Generate roi intensity and morphology masks"""

        if self.roi is None:
            self.roi_intensity = None
            self.roi_morphology = None
        else:
            self.roi_intensity = self.roi.copy()
            self.roi_morphology = self.roi.copy()

    def update_roi(self):
        """Update region of interest based on intensity and morphological masks"""

        if self.roi is None or self.roi_intensity is None or self.roi_morphology is None:
            return

        self.roi.set_voxel_grid(
            voxel_grid=np.logical_or(self.roi_intensity.get_voxel_grid(), self.roi_morphology.get_voxel_grid()))

    def decimate(self, by_slice):
        """
        Decimates the roi mask.
        :param by_slice: boolean, 2D (True) or 3D (False)
        :return:
        """
        if self.roi is not None:
            self.roi.decimate(by_slice=by_slice)
        if self.roi_intensity is not None:
            self.roi_intensity.decimate(by_slice=by_slice)
        if self.roi_morphology is not None:
            self.roi_morphology.decimate(by_slice=by_slice)

    def crop(
            self,
            ind_ext_z=None,
            ind_ext_y=None,
            ind_ext_x=None,
            xy_only=False,
            z_only=False):

        # Crop masks.
        if self.roi is not None:
            self.roi.crop(
                ind_ext_z=ind_ext_z,
                ind_ext_y=ind_ext_y,
                ind_ext_x=ind_ext_x,
                xy_only=xy_only,
                z_only=z_only)

        if self.roi_intensity is not None:
            self.roi_intensity.crop(
                ind_ext_z=ind_ext_z,
                ind_ext_y=ind_ext_y,
                ind_ext_x=ind_ext_x,
                xy_only=xy_only,
                z_only=z_only)

        if self.roi_morphology is not None:
            self.roi_morphology.crop(
                ind_ext_z=ind_ext_z,
                ind_ext_y=ind_ext_y,
                ind_ext_x=ind_ext_x,
                xy_only=xy_only,
                z_only=z_only)

    def crop_to_size(self, center, crop_size):
        """"Crops roi to a pre-defined size"""

        # Crop masks to size
        if self.roi is not None:
            self.roi.crop_to_size(center=center, crop_size=crop_size)
        if self.roi_intensity is not None:
            self.roi_intensity.crop_to_size(center=center, crop_size=crop_size)
        if self.roi_morphology is not None:
            self.roi_morphology.crop_to_size(center=center, crop_size=crop_size)
