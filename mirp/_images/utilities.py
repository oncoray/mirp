import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from mirp._images.generic_image import GenericImage
from mirp._images.mask_image import MaskImage
from mirp._masks.base_mask import BaseMask


class InteractivePlot:  # pragma: no cover

    def __init__(
            self,
            image: GenericImage,
            mask: MaskImage | BaseMask | None = None,
            slice_id: int | None = None
    ):

        # Determine if a mask should be shown.
        show_mask = mask is not None and not mask.is_empty()

        if show_mask and isinstance(mask, BaseMask):
            mask = mask.roi
        if show_mask:
            show_mask = not mask.is_empty_mask()

        # Generate figure
        figure, axes = plt.subplots()
        self.axes = axes
        self.figure = figure

        # Attach connections
        self.scroll_cid = self.figure.canvas.mpl_connect("scroll_event", self.onscroll)
        self.close_cid = self.figure.canvas.mpl_connect("close_event", self.disconnect)

        self.image_data = image.get_voxel_grid()

        self.mask_data = None
        if show_mask:
            self.mask_data = mask.get_voxel_grid()

        self.n_slices, _, _ = image.image_dimension
        if slice_id is None:
            self.slice_index = int(np.floor(self.n_slices / 2.0))
        elif not isinstance(slice_id, int):
            raise TypeError(f"slice_id should be an integer (int). Found: {type(slice_id)}")
        elif slice_id < 1:
            self.slice_index = 0
            warnings.warn(f"slice_id cannot be smaller than 1. Found: {slice_id}", UserWarning)
        elif slice_id > self.n_slices:
            self.slice_index = self.n_slices - 1
            warnings.warn(
                f"slice_id cannot be greater than the number of slices ({self.n_slices}). Found: {slice_id}",
                UserWarning
            )
        else:
            self.slice_index = slice_id - 1

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
            from importlib_metadata import version
            from packaging.version import Version

            # Define color map. The custom color map goes from transparent black to semi-transparent green and is
            # used as an overlay. Note that register_cmap is deprecated in version 3.9.0 of matplotlib
            if Version(version("matplotlib")) >= Version("3.9.0"):
                import matplotlib

                matplotlib.colormaps.register(
                    cmap=LinearSegmentedColormap(
                        "mask_cm",
                        {
                            'red': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                            'green': ((0.0, 0.0, 0.0), (1.0, 0.6, 0.6)),
                            'blue': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                            'alpha': ((0.0, 0.0, 0.0), (1.0, 0.4, 0.4))
                        }
                    ),
                    force=True
                )

            else:
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

    def disconnect(self):
        self.figure.canvas.mpl_disconnect(self.scroll_cid)
        self.figure.canvas.mpl_disconnect(self.close_cid)