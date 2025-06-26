import numpy as np
import copy

from mirp._images.generic_image import GenericImage
from mirp._images.transformed_image import LocalBinaryPatternImage
from mirp.settings.generic import SettingsClass
from mirp._imagefilters.generic import GenericFilter


class LocalBinaryPatternFilter(GenericFilter):

    def __init__(self, image: GenericImage, settings: SettingsClass, name: str):

        super().__init__(image=image, settings=settings, name=name)

        self.ibsi_compliant = False

        self.separate_slices = False
        self.lbp_method = ""
        self.d = 1.8

    def generate_object(self):
        # Generator for transformation objects.
        distance = copy.deepcopy(self.d)
        if not isinstance(distance, list):
            distance = [distance]

        # Iterate over options to yield filter objects with specific settings. A copy of the parent object is made to
        # avoid updating by reference.
        for current_distance in distance:
            filter_object = copy.deepcopy(self)
            filter_object.d = current_distance

            yield filter_object

    def transform(self, image: GenericImage) -> LocalBinaryPatternImage:
        # Create placeholder Laplacian-of-Gaussian response map.
        response_map = LocalBinaryPatternImage(
            image_data=None,
            separate_slices=self.separate_slices,
            distance=self.d,
            template=image
        )
        response_map.ibsi_compliant = self.ibsi_compliant and image.ibsi_compliant

        if image.is_empty():
            return response_map

        response_map.set_voxel_grid(
            voxel_grid=self.transform_grid(voxel_grid=image.get_voxel_grid())
        )

        return response_map

    def transform_grid(
            self,
            voxel_grid: np.ndarray
    ):

        if self.separate_slices:
            # Set the number of dimensions.
            d = 2.0

            # Create the grid coordinates, with [0, 0, 0] in the center.
            y, x = np.mgrid[:filter_size[1], :filter_size[2]]
            y -= (filter_size[1] - 1.0) / 2.0
            x -= (filter_size[2] - 1.0) / 2.0

            # Compute the square of the norm.
            norm_2 = np.power(y, 2.0) + np.power(x, 2.0)

        else:
            # Set the number of dimensions.
            d = 3.0

            # Create the grid coordinates, with [0, 0, 0] in the center.
            z, y, x = np.mgrid[:filter_size[0], :filter_size[1], :filter_size[2]]
            z -= (filter_size[0] - 1.0) / 2.0
            y -= (filter_size[1] - 1.0) / 2.0
            x -= (filter_size[2] - 1.0) / 2.0

            # Compute the square of the norm.
            norm_2 = np.power(z, 2.0) + np.power(y, 2.0) + np.power(x, 2.0)

        # Set a single sigma value.
        sigma = np.max(sigma)

        # Compute the scale factor
        scale_factor = - 1.0 / sigma ** 2.0 * np.power(1.0 / np.sqrt(2.0 * np.pi * sigma ** 2), d) * (d - norm_2 /
                                                                                                      sigma ** 2.0)

        # Compute the exponent which determines filter width.
        width_factor = - norm_2 / (2.0 * sigma ** 2.0)

        # Compute the weights of the filter.
        filter_weights = np.multiply(scale_factor, np.exp(width_factor))

        if self.separate_slices:
            # Set filter weights and create a filter.
            log_filter = FilterSet2D(
                filter_weights,
                riesz_order=self.riesz_order,
                riesz_steered=self.riesz_steered,
                riesz_sigma=self.riesz_sigma)

            # Convolve laplacian of gaussian filter with the image.
            response_map = log_filter.convolve(
                voxel_grid=voxel_grid,
                mode=self.mode,
                response="real")

        else:
            # Set filter weights and create a filter.
            log_filter = FilterSet3D(
                filter_weights,
                riesz_order=self.riesz_order,
                riesz_steered=self.riesz_steered,
                riesz_sigma=self.riesz_sigma)

            # Convolve laplacian of gaussian filter with the image.
            response_map = log_filter.convolve(
                voxel_grid=voxel_grid,
                mode=self.mode,
                response="real")

        # Compute the convolution
        return response_map
