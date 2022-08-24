import numpy as np
import copy

from typing import Union, List
from mirp.imageProcess import calculate_features
from mirp.imageClass import ImageClass
from mirp.imageFilters.utilities import pool_voxel_grids, FilterSet2D
from mirp.importSettings import SettingsClass
from mirp.roiClass import RoiClass


class GaborFilter:

    def __init__(self, settings: SettingsClass, name: str):

        # Sigma parameter that determines filter width.
        self.sigma: Union[None, float, List[float]] = settings.img_transform.gabor_sigma

        # Cut-off for filter size.
        self.sigma_cutoff = settings.img_transform.gabor_sigma_truncate

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
        self.pooling_method = settings.img_transform.gabor_pooling_method and self.rotation_invariance

        # Boundary conditions.
        self.mode = settings.img_transform.gabor_boundary_condition

        # In-slice (2D) or 3D filtering.
        self.by_slice = settings.img_transform.by_slice

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

    def _generate_object(self, allow_pooling: bool = False):
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

    def apply_transformation(self,
                             img_obj: ImageClass,
                             roi_list: List[RoiClass],
                             settings: SettingsClass,
                             compute_features: bool = False,
                             extract_images: bool = False,
                             file_path: str = None):
        """Run feature extraction for transformed data"""

        feature_list = []

        # Iterate over generated filter objects with unique settings.
        for filter_object in self._generate_object(allow_pooling=True):

            # Create a response map.
            response_map = filter_object.transform(img_obj=img_obj)

            # Export the image.
            if extract_images:
                response_map.export(file_path=file_path)

            # Compute features.
            if compute_features:
                feature_list += [calculate_features(img_obj=response_map,
                                                    roi_list=[roi_obj.copy() for roi_obj in roi_list],
                                                    settings=settings.img_transform.feature_settings,
                                                    append_str=response_map.spat_transform + "_")]

            del response_map

        return feature_list

    def transform(self, img_obj: ImageClass):
        """
        Transform image by calculating the laplacian of the gaussian second derivatives
        :param img_obj: image object
        :return:
        """

        # Copy base image
        response_map = img_obj.copy(drop_image=True)

        # Prepare the string for the spatial transformation.
        spatial_transform_string = ["gabor",
                                    "s", str(self.sigma),
                                    "g", str(self.gamma),
                                    "l", str(self.lambda_parameter)]

        if not self.pool_theta:
            spatial_transform_string += ["t", self.theta]

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
        for ii, pooled_filter_object in enumerate(self._generate_object(allow_pooling=False)):
            # Generate transformed voxel grid.
            pooled_voxel_grid = pooled_filter_object.transform_grid(voxel_grid=img_obj.get_voxel_grid(),
                                                                    spacing=img_obj.spacing)

            # Pool voxel grids.
            response_voxel_grid = pool_voxel_grids(x1=response_voxel_grid,
                                                   x2=pooled_voxel_grid,
                                                   pooling_method=self.pooling_method)

        if self.pooling_method == "mean":
            # Perform final pooling step for mean pooling.
            response_voxel_grid = np.divide(response_voxel_grid, ii + 1)

        # Set voxel grid.
        response_map.set_voxel_grid(voxel_grid=response_voxel_grid)

        return response_map

    def transform_grid(self,
                       voxel_grid: np.ndarray,
                       spacing: np.array):

        # Get in-plane spacing, i.e. not stack_axis.
        spacing: float = max([current_spacing for ii, current_spacing in enumerate(spacing.tolist())
                              if not ii == self.stack_axis])

        # Convert sigma from physical units to voxel units.
        sigma: float = self.sigma / spacing
        lambda_p: float = self.lambda_parameter / spacing

        # Convert theta to radians.
        theta = np.deg2rad(self.theta)

        # Determine size for x (alpha) and y (beta), prior to rotation.
        if self.sigma_cutoff is not None:
            alpha = self.sigma_cutoff * sigma
            beta = self.sigma_cutoff * sigma * self.gamma

            # Determine filter size.
            x_size = max(np.abs(alpha * np.cos(theta) + beta * np.sin(theta)),
                         np.abs(-alpha * np.cos(theta) + beta * np.sin(theta)),
                         1)
            y_size = max(np.abs(alpha * np.sin(theta) - beta * np.cos(theta)),
                         np.abs(-alpha * np.sin(theta) - beta * np.cos(theta)),
                         1)

            x_size = int(1 + 2 * np.floor(x_size + 0.5))
            y_size = int(1 + 2 * np.floor(y_size + 0.5))

        else:
            x_size = voxel_grid.shape[2]
            y_size = voxel_grid.shape[1]

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
        gabor_filter = FilterSet2D(gabor_filter,
                                   riesz_order=self.riesz_order,
                                   riesz_steered=self.riesz_steered,
                                   riesz_sigma=self.riesz_sigma)

        # Convolve gabor filter with the image.
        response_map = gabor_filter.convolve(voxel_grid=voxel_grid,
                                             mode=self.mode,
                                             response=self.response_type,
                                             axis=self.stack_axis)

        # Compute the convolution
        return response_map
