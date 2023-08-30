import numpy as np
import copy

from typing import Union, List
from mirp.imageClass import ImageClass
from mirp.images.genericImage import GenericImage
from mirp.images.transformedImage import GaborTransformedImage
from mirp.imageFilters.genericFilter import GenericFilter
from mirp.imageFilters.utilities import pool_voxel_grids, FilterSet2D
from mirp.settings.settingsClass import SettingsClass


class GaborFilter(GenericFilter):

    def __init__(self, settings: SettingsClass, name: str):

        super().__init__(
            settings=settings,
            name=name
        )

        # Sigma parameter that determines filter width.
        self.sigma: Union[None, float, List[float]] = settings.img_transform.gabor_sigma

        # Eccentricity parameter
        self.gamma: Union[None, float, List[float]] = settings.img_transform.gabor_gamma

        # Wavelength parameter
        self.lambda_parameter: Union[None, float, List[float]] = settings.img_transform.gabor_lambda

        # Initial angle.
        self.theta: Union[None, float, List[float], int, List[int]] = settings.img_transform.gabor_theta

        # Set whether theta is considered separate, or pooled.
        self.pool_theta: bool = settings.img_transform.gabor_pool_theta

        # Update ype of response
        self.response_type = settings.img_transform.gabor_response

        # Rotational invariance.
        self.rotation_invariance = settings.img_transform.gabor_rotation_invariance

        # Which pooling method is used.
        self.pooling_method = settings.img_transform.gabor_pooling_method

        # Boundary conditions.
        self.mode = settings.img_transform.gabor_boundary_condition

        # Riesz transformation settings.
        self.riesz_order: Union[None, List[int], List[List[int]]] = None
        self.riesz_steered: bool = False
        self.riesz_sigma: Union[None, float, List[float]] = None
        if settings.img_transform.has_riesz_filter(x=name):
            self.riesz_order = settings.img_transform.riesz_order

            if settings.img_transform.has_steered_riesz_filter(x=name):
                self.riesz_steered = True
                self.riesz_sigma = settings.img_transform.riesz_filter_tensor_sigma

        # Set the axis orthogonal to the plane in which the Gabor kernel is applied.
        if self.by_slice or not self.rotation_invariance:
            self.stack_axis: Union[int, List[int]] = [0]
        else:
            self.stack_axis: Union[int, List[int]] = [0, 1, 2]

    def generate_object(self, allow_pooling: bool = True):
        # Generator for transformation objects.
        sigma = copy.deepcopy(self.sigma)
        if not isinstance(sigma, list):
            sigma = [sigma]

        gamma = copy.deepcopy(self.gamma)
        if not isinstance(gamma, list):
            gamma = [gamma]

        lambda_p = copy.deepcopy(self.lambda_parameter)
        if not isinstance(lambda_p, list):
            lambda_p = [lambda_p]

        theta = copy.deepcopy(self.theta)
        if not isinstance(theta, list):
            theta = [theta]

        # Nest theta for internal iterations.
        if self.pool_theta and allow_pooling:
            theta = [theta]

        axis = copy.deepcopy(self.stack_axis)
        if not isinstance(axis, list):
            axis = [axis]

        # Nest axis for internal iterations.
        if self.pool_theta and allow_pooling:
            axis = [axis]

        riesz_order = copy.deepcopy(self.riesz_order)
        if riesz_order is None:
            riesz_order = [None]
        elif not all(isinstance(riesz_order_set, list) for riesz_order_set in riesz_order):
            riesz_order = [riesz_order]

        riesz_sigma = copy.deepcopy(self.riesz_sigma)
        if not isinstance(riesz_sigma, list):
            riesz_sigma = [riesz_sigma]

        # Iterate over options to yield filter objects with specific settings. A copy of the parent object is made to
        # avoid updating by reference.
        for current_sigma in sigma:
            for current_gamma in gamma:
                for current_lambda in lambda_p:
                    for current_riesz_order in riesz_order:
                        for current_riesz_sigma in riesz_sigma:
                            for current_theta in theta:
                                for current_axis in axis:

                                    filter_object = copy.deepcopy(self)
                                    filter_object.sigma = current_sigma
                                    filter_object.gamma = current_gamma
                                    filter_object.lambda_parameter = current_lambda
                                    filter_object.riesz_order = current_riesz_order
                                    filter_object.riesz_sigma = current_riesz_sigma
                                    filter_object.theta = current_theta
                                    filter_object.stack_axis = current_axis

                                    yield filter_object

    def transform(self, image: GenericImage) -> GaborTransformedImage:
        # Create placeholder Gabor response map.
        response_map = GaborTransformedImage(
            image_data=None,
            sigma_parameter=self.sigma,
            gamma_parameter=self.gamma,
            lambda_parameter=self.lambda_parameter,
            theta_parameter=self.theta,
            pool_theta=self.pool_theta,
            response_type=self.response_type,
            rotation_invariance=self.rotation_invariance,
            pooling_method=self.pooling_method,
            boundary_condition=self.mode,
            riesz_order=self.riesz_order,
            riesz_steering=self.riesz_steered,
            riesz_sigma_parameter=self.riesz_sigma,
            template=image
        )

        if image.is_empty():
            return response_map

        # Set response voxel grid.
        response_voxel_grid = None

        # Initialise iterator ii to avoid IDE warnings.
        ii = 0
        for ii, pooled_filter_object in enumerate(self.generate_object(allow_pooling=False)):
            # Generate transformed voxel grid.
            pooled_voxel_grid = pooled_filter_object.transform_grid(
                voxel_grid=image.get_voxel_grid(),
                spacing=image.image_spacing)

            # Pool voxel grids.
            response_voxel_grid = pool_voxel_grids(
                x1=response_voxel_grid,
                x2=pooled_voxel_grid,
                pooling_method=self.pooling_method)

        if self.pooling_method == "mean":
            # Perform final pooling step for mean pooling.
            response_voxel_grid = np.divide(response_voxel_grid, ii + 1)

        # Set voxel grid.
        response_map.set_voxel_grid(voxel_grid=response_voxel_grid)

        return response_map

    def transform_deprecated(self, img_obj: ImageClass):
        """
        Transform image by calculating the laplacian of the gaussian second derivatives
        :param img_obj: image object
        :return:
        """

        # Copy base image
        response_map = img_obj.copy(drop_image=True)

        # Prepare the string for the spatial transformation.
        spatial_transform_string = [
            "gabor",
            "s", str(self.sigma),
            "g", str(self.gamma),
            "l", str(self.lambda_parameter)]

        if not self.pool_theta:
            spatial_transform_string += ["t", str(self.theta)]

        spatial_transform_string += ["2D" if self.by_slice else "3D"]

        if self.rotation_invariance and not self.by_slice:
            spatial_transform_string += ["invar"]

        # Set the name of the transformation.
        response_map.set_spatial_transform("_".join(spatial_transform_string))

        if img_obj.is_missing:
            return response_map

        # Set response voxel grid.
        response_voxel_grid = None

        # Initialise iterator ii to avoid IDE warnings.
        ii = 0
        for ii, pooled_filter_object in enumerate(self.generate_object(allow_pooling=False)):
            # Generate transformed voxel grid.
            pooled_voxel_grid = pooled_filter_object.transform_grid(
                voxel_grid=img_obj.get_voxel_grid(),
                spacing=img_obj.spacing)

            # Pool voxel grids.
            response_voxel_grid = pool_voxel_grids(
                x1=response_voxel_grid,
                x2=pooled_voxel_grid,
                pooling_method=self.pooling_method)

        if self.pooling_method == "mean":
            # Perform final pooling step for mean pooling.
            response_voxel_grid = np.divide(response_voxel_grid, ii + 1)

        # Set voxel grid.
        response_map.set_voxel_grid(voxel_grid=response_voxel_grid)

        return response_map

    def transform_grid(
            self,
            voxel_grid: np.ndarray,
            spacing: np.array):

        # Get in-plane spacing, i.e. not stack_axis.
        spacing: float = max([
            current_spacing for ii, current_spacing in enumerate(spacing.tolist())
            if not ii == self.stack_axis
        ])

        # Convert sigma from physical units to voxel units.
        sigma: float = self.sigma / spacing
        lambda_p: float = self.lambda_parameter / spacing

        # Convert theta to radians.
        theta = np.deg2rad(self.theta)

        # Get size of the voxelgrid as filter size.
        x_size = y_size = max([
            current_shape for ii, current_shape in enumerate(voxel_grid.shape)
            if not ii == self.stack_axis
        ])

        # Ensure that size is uneven.
        x_size = int(1 + 2 * np.floor(x_size / 2.0))
        y_size = int(1 + 2 * np.floor(y_size / 2.0))

        # Create grid coordinates with [0, 0] in the center.
        y, x = np.mgrid[:y_size, :x_size].astype(float)
        y -= (y_size - 1.0) / 2.0
        x -= (x_size - 1.0) / 2.0

        # Compute rotation matrix: Since we are computing clock-wise rotations, use negative angles.
        rotation_matrix = np.array([[-np.cos(theta), np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

        # Compute rotated grid coordinates around the center.
        rotated_scan_coordinates = np.dot(rotation_matrix, np.array((y.flatten(), x.flatten())))
        y = rotated_scan_coordinates[0, :].reshape((y_size, x_size))
        x = rotated_scan_coordinates[1, :].reshape((y_size, x_size))

        # Create filter weights.
        gabor_filter = np.exp(-(np.power(x, 2.0) + self.gamma ** 2.0 * np.power(y, 2.0)) / (2.0 * sigma ** 2.0) + 1.0j
                              * (2.0 * np.pi * x) / lambda_p)

        # Create filter
        gabor_filter = FilterSet2D(
            gabor_filter,
            riesz_order=self.riesz_order,
            riesz_steered=self.riesz_steered,
            riesz_sigma=self.riesz_sigma)

        # Convolve gabor filter with the image.
        response_map = gabor_filter.convolve(
            voxel_grid=voxel_grid,
            mode=self.mode,
            response=self.response_type,
            axis=self.stack_axis)

        # Compute the convolution
        return response_map
