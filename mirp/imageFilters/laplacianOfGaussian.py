import numpy as np
import copy

from typing import List, Union
from mirp.imageProcess import calculate_features
from mirp.imageClass import ImageClass
from mirp.imageFilters.utilities import FilterSet2D, FilterSet3D
from mirp.importSettings import SettingsClass
from mirp.roiClass import RoiClass
from mirp.imageFilters.utilities import pool_voxel_grids


class LaplacianOfGaussianFilter:

    def __init__(self, settings: SettingsClass, name: str):
        self.sigma: Union[float, List[float]] = settings.img_transform.log_sigma
        self.sigma_cutoff = settings.img_transform.log_sigma_truncate
        self.pooling_method = settings.img_transform.log_pooling_method
        self.mode = settings.img_transform.log_boundary_condition

        # Riesz transformation settings.
        self.riesz_order: Union[None, List[int], List[List[int]]] = None
        self.riesz_steered: bool = False
        self.riesz_sigma: Union[None, float, List[float]] = None
        if settings.img_transform.has_riesz_filter(x=name):
            self.riesz_order = settings.img_transform.riesz_order

            if settings.img_transform.has_steered_riesz_filter(x=name):
                self.riesz_steered = True
                self.riesz_sigma = settings.img_transform.riesz_filter_tensor_sigma

        # In-slice (2D) or 3D filtering
        self.by_slice = settings.img_transform.by_slice

    def _generate_object(self, allow_pooling: bool = False):
        # Generator for transformation objects.
        sigma = copy.deepcopy(self.sigma)
        if not isinstance(sigma, list):
            sigma = [sigma]

        # Nest sigma.
        if not self.pooling_method == "none" and allow_pooling:
            sigma = [sigma]

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
        for current_riesz_order in riesz_order:
            for current_riesz_sigma in riesz_sigma:
                for current_sigma in sigma:
                    filter_object = copy.deepcopy(self)
                    filter_object.sigma = current_sigma
                    filter_object.riesz_order = current_riesz_order
                    filter_object.riesz_sigma = current_riesz_sigma

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
        :sigma_cut_off: number of standard deviations for cut-off of the gaussian filter
        """

        # Copy base image
        response_map = img_obj.copy(drop_image=True)

        # Prepare the string for the spatial transformation.
        spatial_transform_string = ["log"]

        if self.pooling_method == "none":
            spatial_transform_string += ["s", str(self.sigma)]

        else:
            spatial_transform_string += [self.pooling_method]

        # Set spatial transformation name.
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
            response_voxel_grid = np.divide(response_voxel_grid, ii+1)

        # Set voxel grid.
        response_map.set_voxel_grid(voxel_grid=response_voxel_grid)

        return response_map

    def transform_grid(self,
                       voxel_grid: np.ndarray,
                       spacing: np.ndarray):

        # Update sigma to voxel units.
        sigma = np.divide(np.full(shape=3, fill_value=self.sigma), spacing)

        # Determine the size of the filter
        filter_size = 1 + 2 * np.floor(self.sigma_cutoff * sigma + 0.5)

        if self.by_slice:
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

        if self.by_slice:
            # Set filter weights and create a filter.
            log_filter = FilterSet2D(filter_weights,
                                     riesz_order=self.riesz_order,
                                     riesz_steered=self.riesz_steered,
                                     riesz_sigma=self.riesz_sigma)

            # Convolve laplacian of gaussian filter with the image.
            response_map = log_filter.convolve(voxel_grid=voxel_grid,
                                               mode=self.mode,
                                               response="real")

        else:
            # Set filter weights and create a filter.
            log_filter = FilterSet3D(filter_weights,
                                     riesz_order=self.riesz_order,
                                     riesz_steered=self.riesz_steered,
                                     riesz_sigma=self.riesz_sigma)

            # Convolve laplacian of gaussian filter with the image.
            response_map = log_filter.convolve(voxel_grid=voxel_grid,
                                               mode=self.mode,
                                               response="real")

        # Compute the convolution
        return response_map
