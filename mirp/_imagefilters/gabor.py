from typing import Any

import numpy as np
import copy

import pandas as pd

from mirp._images.generic_image import GenericImage
from mirp._imagefilters.generic import GenericFilter
from mirp._imagefilters.utilities import pool_voxel_grids, FilterSet2D
from mirp._images.transformed_image import TransformedImage
from mirp.settings.generic import SettingsClass


class GaborTransformedImage(TransformedImage):
    def __init__(
            self,
            sigma_parameter: None | float = None,
            gamma_parameter: None | float = None,
            lambda_parameter: None | float = None,
            theta_parameter: None | float = None,
            pool_theta: None | bool = None,
            response_type: None | str = None,
            rotation_invariance: None | bool = None,
            pooling_method: None | str = None,
            boundary_condition: None | str = None,
            riesz_order: None | int | list[int] = None,
            riesz_steering: None | bool = None,
            riesz_sigma_parameter: None | float = None,
            template: None | GenericImage = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Filter parameters
        self.sigma_parameter = sigma_parameter
        self.gamma_parameter = gamma_parameter
        self.lambda_parameter = lambda_parameter
        self.theta_parameter = theta_parameter
        self.pool_theta = pool_theta
        self.response_type = response_type
        self.rotation_invariance = rotation_invariance
        self.pooling_method = pooling_method
        self.boundary_condition = boundary_condition
        self.riesz_transformed = riesz_order is not None
        self.riesz_order = copy.deepcopy(riesz_order)
        self.riesz_steering = riesz_steering
        self.riesz_sigma_parameter = riesz_sigma_parameter

        # Update image parameters using the template.
        if isinstance(template, GenericImage):
            self.update_from_template(template=template)

    def get_file_name_descriptor(self) -> list[str]:
        descriptors = super().get_file_name_descriptor()

        descriptors += [
            "gabor",
            "s", str(self.sigma_parameter),
            "g", str(self.gamma_parameter),
            "l", str(self.lambda_parameter)
        ]

        if not self.pool_theta:
            descriptors += ["t", str(self.theta_parameter)]

        descriptors += ["2D" if self.separate_slices else "3D"]

        if self.rotation_invariance and not self.separate_slices:
            descriptors += ["invar"]

        return descriptors

    def get_export_attributes(self) -> dict[str, Any]:
        parent_attributes = super().get_export_attributes()

        attributes = [
            ("filter_type", "gabor"),
            ("sigma_parameter", self.sigma_parameter),
            ("gamma_parameter", self.gamma_parameter),
            ("lambda_parameter", self.lambda_parameter),
            ("theta_parameter", self.theta_parameter),
            ("pool_theta", self.pool_theta),
            ("response_type", self.response_type),
            ("rotation_invariance", self.rotation_invariance),
            ("boundary_condition", self.boundary_condition)
        ]

        if self.pooling_method is not None:
            attributes += [("pooling_method", self.pooling_method)]

        if self.riesz_transformed:
            attributes += [("riesz_order", self.riesz_order)]

            if self.riesz_steering:
                attributes += [("riesz_sigma_parameter", self.riesz_sigma_parameter)]

        parent_attributes.update(dict(attributes))

        return parent_attributes

    def parse_feature_names(self, x: None | pd.DataFrame) -> pd.DataFrame:
        x = super().parse_feature_names(x=x)

        feature_name_prefix = [
            "gabor",
            "s", str(self.sigma_parameter),
            "g", str(self.gamma_parameter),
            "l", str(self.lambda_parameter)
        ]

        if not self.pool_theta:
            feature_name_prefix += ["t", str(self.theta_parameter)]

        feature_name_prefix += ["2D" if self.separate_slices else "3D"]

        if self.rotation_invariance and not self.separate_slices:
            feature_name_prefix += ["invar"]

        if len(feature_name_prefix) > 0:
            feature_name_prefix = "_".join(feature_name_prefix)
            feature_name_prefix += "_"
            x.columns = feature_name_prefix + x.columns

        return x


class GaborFilter(GenericFilter):

    def __init__(self, image: GenericImage, settings: SettingsClass, name: str):

        super().__init__(image=image, settings=settings, name=name)
        self.ibsi_compliant = True
        self.ibsi_id = "Q88H"

        # Sigma parameter that determines filter width.
        self.sigma: None | float | list[float] = settings.img_transform.gabor_sigma

        # Eccentricity parameter
        self.gamma: None | float | list[float] = settings.img_transform.gabor_gamma

        # Wavelength parameter
        self.lambda_parameter: None | float | list[float] = settings.img_transform.gabor_lambda

        # Initial angle.
        self.theta: None | float | list[float] | int | list[int] = settings.img_transform.gabor_theta

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
        self.riesz_order: None | list[int] | list[list[int]] = None
        self.riesz_steered: bool = False
        self.riesz_sigma: None | float | list[float] = None
        if settings.img_transform.has_riesz_filter(x=name):
            self.riesz_order = settings.img_transform.riesz_order

            if settings.img_transform.has_steered_riesz_filter(x=name):
                self.riesz_steered = True
                self.riesz_sigma = settings.img_transform.riesz_filter_tensor_sigma

            # Riesz transformed filters are not IBSI-compliant
            self.ibsi_compliant = False

        # Set the axis orthogonal to the plane in which the Gabor kernel is applied.
        if self.separate_slices or not self.rotation_invariance:
            self.stack_axis: int | list[int] = [0]
        else:
            self.stack_axis: int | list[int] = [0, 1, 2]

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
        response_map.ibsi_compliant = self.ibsi_compliant and image.ibsi_compliant

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
                spacing=np.array(image.image_spacing))

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
            spacing: np.ndarray):

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
