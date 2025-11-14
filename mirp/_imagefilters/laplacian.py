import numpy as np
import copy

from warnings import warn

from mirp._images.generic_image import GenericImage
from mirp._images.transformed_image import LaplacianTransformedImage
from mirp._imagefilters.utilities import FilterSet2D, FilterSet3D
from mirp.settings.generic import SettingsClass
from mirp._imagefilters.generic import GenericFilter
from mirp._imagefilters.utilities import pool_voxel_grids


class LaplacianFilter(GenericFilter):

    def __init__(self, image: GenericImage, settings: SettingsClass, name: str):

        super().__init__(image=image, settings=settings, name=name)

        self.ibsi_compliant = False
        self.ibsi_id = None

        self.laplace_type = settings.img_transform.laplace_type
        self.mode = settings.img_transform.laplace_boundary_condition

    def generate_object(self, allow_pooling: bool = True):
        # Generator for transformation objects.
         yield copy.deepcopy(self)

    def transform(self, image: GenericImage) -> LaplacianTransformedImage:
        # Create placeholder Laplacian-of-Gaussian response map.
        response_map = LaplacianTransformedImage(
            image_data=None,
            laplace_type=self.laplace_type,
            boundary_condition=self.mode,
            template=image
        )
        response_map.ibsi_compliant = self.ibsi_compliant and image.ibsi_compliant

        if image.is_empty():
            return response_map

        # Initialise iterator ii to avoid IDE warnings.
        filter_object = [self.generate_object()][0]

        # Compute filtered image.
        response_voxel_grid = filter_object.transform_grid(voxel_grid=image.get_voxel_grid())

        # Set voxel grid.
        response_map.set_voxel_grid(voxel_grid=response_voxel_grid)

        return response_map

    def transform_grid(
            self,
            voxel_grid: np.ndarray
    ):
        if self.laplace_type == "convolution":
            if self.separate_slices:
                filter_weights = np.array([
                    [0.0, 1.0, 0.0],
                    [1.0, -4.0, 1.0],
                    [0.0, 1.0, 0.0]
                ])

            else:
                filter_weights = np.array([
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0]
                    ], [
                        [0.0, 1.0, 0.0],
                        [1.0, -6.0, 1.0],
                        [0.0, 1.0, 0.0]
                    ], [
                        [0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0]
                    ]
                ])

        elif self.laplace_type == "oono_puri":
            return "lapl_op"
        elif self.laplace_type == "lynch":
            return "lapl_l"
        elif self.laplace_type == "equidistant":
            if self.separate_slices:
                filter_weights = np.array([
                    [1.0, 1.0, 1.0],
                    [1.0, -8.0, 1.0],
                    [1.0, 1.0, 1.0]
                ])

            else:
                filter_weights = np.array([
                    [
                        [1.0, 1.0, 1.0],
                        [1.0, 3.0, 1.0],
                        [1.0, 1.0, 1.0]
                    ], [
                        [1.0, 3.0, 1.0],
                        [3.0, -18.0, 3.0],
                        [1.0, 3.0, 1.0]
                    ], [
                        [1.0, 1.0, 1.0],
                        [1.0, 3.0, 1.0],
                        [1.0, 1.0, 1.0]
                    ]
                ])

        else:
            raise ValueError(f"An unexpected value for laplace_type was encountered: {self.laplace_type}")

        if self.separate_slices:


        # Update sigma to voxel units.
        sigma = np.divide(np.full(shape=3, fill_value=self.sigma), spacing)
        if self.separate_slices:
            sigma = sigma[[1, 2]]

        if max(sigma) < 1.0:
            warn(f"Sigma is smaller than the image spacing: this may lead to poor sampling of the "
                 f"Laplacian-of-Gaussian function. ", UserWarning)

        # Determine the size of the filter
        filter_size = 1 + 2 * np.floor(self.sigma_cutoff * sigma + 0.5)

        if self.separate_slices:
            # Set the number of dimensions.
            d = 2.0

            # Create the grid coordinates, with [0, 0, 0] in the center.
            y, x = np.mgrid[:filter_size[0], :filter_size[1]]
            y -= (filter_size[0] - 1.0) / 2.0
            x -= (filter_size[1] - 1.0) / 2.0

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
        scale_factor = self._get_scale_factor(sigma=sigma, d=d, norm_2=norm_2)

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

    @staticmethod
    def _get_scale_factor(sigma, d, norm_2):
        return (
            - 1.0 / sigma ** 2.0 *
            np.power(1.0 / np.sqrt(2.0 * np.pi * sigma ** 2), d) *
            (d - norm_2 / sigma ** 2.0)
        )