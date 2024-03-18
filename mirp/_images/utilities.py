import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from mirp._images.generic_image import GenericImage
from mirp._images.mask_image import MaskImage
from mirp._masks.baseMask import BaseMask


class InteractivePlot:

    def __init__(
            self,
            axes: plt.Axes,
            image: GenericImage,
            mask: MaskImage | BaseMask | None = None
    ):

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
        self.image_layer = self.axes.imshow(
            self.image_data[self.slice_index, :, :],
            vmin=min_intensity,
            vmax=max_intensity,
            cmap=colour_map
        )

        # Create mask.
        if show_mask:
            # Define color map. The custom color map goes from transparent black to semi-transparent green and is
            # used as an overlay.

            # Create map and register
            plt.register_cmap(
                cmap=LinearSegmentedColormap(
                    "mask_cm",
                    {
                        'red': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                        'green': ((0.0, 0.0, 0.0), (1.0, 0.6, 0.6)),
                        'blue': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                        'alpha': ((0.0, 0.0, 0.0), (1.0, 0.4, 0.4))
                    }
                ),
                override_builtin=True
            )

            self.mask_layer = self.axes.imshow(
                self.mask_data[self.slice_index, :, :],
                vmin=0.0,
                vmax=1.0,
                cmap="mask_cm"
            )
        else:
            self.mask_layer = None

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
        self.axes.set_title(f"slice {self.slice_index + 1}")
        self.image_layer.set_data(self.image_data[self.slice_index, :, :])
        self.image_layer.axes.figure.canvas.draw()

        if self.mask_layer is not None:
            self.mask_layer.set_data(self.mask_data[self.slice_index, :, :])
            self.mask_layer.axes.figure.canvas.draw()
