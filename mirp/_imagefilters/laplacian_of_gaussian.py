from typing import Any

import numpy as np
import pandas as pd
import copy

from warnings import warn

from mirp._images.generic_image import GenericImage
from mirp._imagefilters.utilities import FilterSet2D, FilterSet3D
from mirp._images.transformed_image import TransformedImage
from mirp.settings.generic import SettingsClass
from mirp._imagefilters.generic import GenericFilter
from mirp._imagefilters.utilities import pool_voxel_grids


class LaplacianOfGaussianTransformedImage(TransformedImage):
    def __init__(
            self,
            is_normalised: None | bool = None,
            sigma_parameter: None | float = None,
            sigma_cutoff_parameter: None | float = None,
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
        self.is_normalised = False if is_normalised is None else is_normalised
        self.sigma_parameter = sigma_parameter
        self.sigma_cutoff_parameter = sigma_cutoff_parameter
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

        if self.is_normalised:
            descriptors += ["norm"]

        descriptors += [
            "log",
            "s", str(self.sigma_parameter)
        ]

        return descriptors

    def get_export_attributes(self) -> dict[str, Any]:
        parent_attributes = super().get_export_attributes()

        filter_type = "normalised_laplacian_of_gaussian" if self.is_normalised else "laplacian_of_gaussian"

        attributes = [
            ("filter_type", filter_type),
            ("sigma_parameter", self.sigma_parameter),
            ("sigma_cutoff_parameter", self.sigma_cutoff_parameter),
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

        feature_name_prefix = []
        if self.is_normalised:
            feature_name_prefix += ["norm"]

        feature_name_prefix += [
            "log",
            "s", str(self.sigma_parameter)
        ]

        if len(feature_name_prefix) > 0:
            feature_name_prefix = "_".join(feature_name_prefix)
            feature_name_prefix += "_"
            x.columns = feature_name_prefix + x.columns

        return x



class LaplacianOfGaussianFilter(GenericFilter):

    def __init__(self, image: GenericImage, settings: SettingsClass, name: str):

        super().__init__(image=image, settings=settings, name=name)

        self.ibsi_compliant = True
        self.ibsi_id = "L6PA"

        self.is_normalised = False
        self.sigma: float | list[float] = settings.img_transform.log_sigma
        self.sigma_cutoff = settings.img_transform.log_sigma_truncate
        self.pooling_method = settings.img_transform.log_pooling_method
        self.mode = settings.img_transform.log_boundary_condition

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

    def generate_object(self, allow_pooling: bool = True):
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

    def transform(self, image: GenericImage) -> LaplacianOfGaussianTransformedImage:
        # Create placeholder Laplacian-of-Gaussian response map.
        response_map = LaplacianOfGaussianTransformedImage(
            is_normalised=self.is_normalised,
            image_data=None,
            sigma_parameter=self.sigma,
            sigma_cutoff_parameter=self.sigma_cutoff,
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
                spacing=np.array(image.image_spacing)
            )

            # Pool voxel grids.
            response_voxel_grid = pool_voxel_grids(
                x1=response_voxel_grid,
                x2=pooled_voxel_grid,
                pooling_method=self.pooling_method
            )

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


class NormalisedLaplacianOfGaussianFilter(LaplacianOfGaussianFilter):

    def __init__(self, image: GenericImage, settings: SettingsClass, name: str):

        super().__init__(image=image, settings=settings, name=name)

        self.is_normalised = True
        self.ibsi_compliant = False
        self.ibsi_id = None

    @staticmethod
    def _get_scale_factor(sigma, d, norm_2):
        # Compared to the conventional version, the normalised version lacks the initial 1.0 / sigma^2 factor.
        return (
            - 1.0 *
            np.power(1.0 / np.sqrt(2.0 * np.pi * sigma ** 2), d) *
            (d - norm_2 / sigma ** 2.0)
        )
