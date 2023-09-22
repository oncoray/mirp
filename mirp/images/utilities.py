import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union

from mirp.images.genericImage import GenericImage
from mirp.images.maskImage import MaskImage
from mirp.masks.baseMask import BaseMask


class InteractivePlot:

    def __init__(
            self,
            axes: plt.Axes,
            image: GenericImage,
            mask: Optional[Union[MaskImage, BaseMask]] = None):

        # Determine if a mask should be shown.
        show_mask = mask is not None and not mask.is_empty()

        if show_mask and isinstance(mask, BaseMask):
            mask = mask.roi
        if show_mask:
            show_mask = not mask.is_empty_mask()

        self.axes = axes
        self.image_data = image.get_voxel_grid()

        self.mask_data = None
        if show_mask:
            self.mask_data = mask.get_voxel_grid()

        self.n_slices, _, _ = image.image_dimension
        self.slice_index = int(np.floor(self.n_slices / 2.0))

        # Set plotting options
        colour_map = image.get_colour_map()
        min_intensity = image.get_default_lowest_intensity() if image.get_default_lowest_intensity() is not None else (
            np.min(self.image_data))
        max_intensity = image.get_default_upper_intensity() if image.get_default_upper_intensity() is not None else (
            np.max(self.image_data))

        # Create plot.
        self.plot = self.axes.imshow(
            self.image_data[self.slice_index, :, :],
            vmin=min_intensity,
            vmax=max_intensity,
            cmap=colour_map
        )
        self.update()

    def onscroll(self, event):
        # Update, but limit to range of available slices.
        if event.button == "up":
            self.slice_index += 1
            if self.slice_index >= self.n_slices:
                self.slice_index -= 1
        else:
            self.slice_index -= 1
            if self.slice_index < 0:
                self.slice_index += 1
        self.update()

    def update(self):
        self.plot.set_data(self.image_data[self.slice_index, :, :])
        self.axes.set_title(f"slice {self.slice_index + 1}")
        self.plot.axes.figure.canvas.draw()
