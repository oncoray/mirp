import copy

import numpy as np
import scipy.fft as fft

from typing import List, Union
from mirp.imageClass import ImageClass
from mirp.images.genericImage import GenericImage
from mirp.images.transformedImage import NonSeparableWaveletTransformedImage
from mirp.settings.importSettings import SettingsClass
from mirp.imageFilters.genericFilter import GenericFilter


class NonseparableWaveletFilter(GenericFilter):

    def __init__(self, settings: SettingsClass, name: str):

        super().__init__(
            settings=settings,
            name=name
        )

        # Set wavelet family
        self.wavelet_family: Union[str, List[str]] = settings.img_transform.nonseparable_wavelet_families

        # Wavelet decomposition level
        self.decomposition_level: Union[int, List[int]] = \
            settings.img_transform.nonseparable_wavelet_decomposition_level

        # Riesz transformation settings.
        self.riesz_order: Union[None, List[int], List[List[int]]] = None
        self.riesz_steered: bool = False
        self.riesz_sigma: Union[None, float, List[float]] = None
        if settings.img_transform.has_riesz_filter(x=name):
            self.riesz_order = settings.img_transform.riesz_order

            if settings.img_transform.has_steered_riesz_filter(x=name):
                self.riesz_steered = True
                self.riesz_sigma = settings.img_transform.riesz_filter_tensor_sigma

        # Set boundary condition
        self.mode = settings.img_transform.nonseparable_wavelet_boundary_condition

        # Set response.
        self.response = settings.img_transform.nonseparable_wavelet_response

    def generate_object(self):
        # Generator for transformation objects.
        wavelet_family = copy.deepcopy(self.wavelet_family)
        if not isinstance(wavelet_family, list):
            wavelet_family = [wavelet_family]

        decomposition_level = copy.deepcopy(self.decomposition_level)
        if not isinstance(decomposition_level, list):
            decomposition_level = [decomposition_level]

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
        for current_family in wavelet_family:
            for current_riesz_order in riesz_order:
                for current_riesz_sigma in riesz_sigma:
                    for current_decomposition_level in decomposition_level:
                        filter_object = copy.deepcopy(self)
                        filter_object.wavelet_family = current_family
                        filter_object.decomposition_level = current_decomposition_level
                        filter_object.riesz_order = current_riesz_order
                        filter_object.riesz_sigma = current_riesz_sigma

                        yield filter_object

    def transform(self, image: GenericImage) -> NonSeparableWaveletTransformedImage:
        # Create placeholder non-separable wavelet response map.
        response_map = NonSeparableWaveletTransformedImage(
            image_data=None,
            wavelet_family=self.wavelet_family,
            decomposition_level=self.decomposition_level,
            response_type=self.response,
            boundary_condition=self.mode,
            riesz_order=self.riesz_order,
            riesz_steering=self.riesz_steered,
            riesz_sigma_parameter=self.riesz_sigma,
            template=image
        )

        if image.is_empty():
            return response_map

        # Create voxel grid
        response_voxel_grid = self.convolve(voxel_grid=image.get_voxel_grid())

        # Store the voxel grid in the ImageObject.
        response_map.set_voxel_grid(voxel_grid=response_voxel_grid)

        return response_map

    def transform_deprecated(
            self,
            img_obj: ImageClass):

        # Copy base image
        response_map = img_obj.copy(drop_image=True)

        # Prepare the string for the spatial transformation.
        spatial_transform_string = ["wavelet", self.wavelet_family]
        spatial_transform_string += ["level", str(self.decomposition_level)]

        # Set the name of the transformation.
        response_map.set_spatial_transform("_".join(spatial_transform_string))

        # Skip transformation in case the input image is missing
        if img_obj.is_missing:
            return response_map

        # Create voxel grid
        img_wavelet_grid = self.convolve(voxel_grid=img_obj.get_voxel_grid())

        # Store the voxel grid in the ImageObject.
        response_map.set_voxel_grid(voxel_grid=img_wavelet_grid)

        return response_map

    def shannon_filter(self, filter_size):
        """
        Set up the shannon filter in the Fourier domain.
        @param filter_size: Size of the filter. By default equal to the size of the image.
        """

        # Get the distance grid.
        distance_grid, max_frequency = self.get_distance_grid(filter_size=filter_size)

        # Set up a wavelet filter for the decomposition specifically.
        wavelet_filter = np.zeros(distance_grid.shape, dtype=float)

        # Set the mask for the filter.
        mask = np.logical_and(distance_grid >= max_frequency / 2.0, distance_grid <= max_frequency)

        # Update the filter.
        wavelet_filter[mask] += 1.0

        return wavelet_filter

    def simoncelli_filter(self, filter_size):
        """
        Set up the simoncelli filter in the Fourier domain.
        @param filter_size: Size of the filter. By default equal to the size of the image.
        """

        # Get the distance grid.
        distance_grid, max_frequency = self.get_distance_grid(filter_size=filter_size)

        # Set up a wavelet filter for the decomposition specifically.
        wavelet_filter = np.zeros(distance_grid.shape, dtype=float)

        # Set the mask for the filter.
        mask = np.logical_and(distance_grid >= max_frequency / 4.0,
                              distance_grid <= max_frequency)

        # Update the filter.
        wavelet_filter[mask] += np.cos(np.pi / 2.0 * np.log2(2.0 * distance_grid[mask] / max_frequency))

        return wavelet_filter

    def get_distance_grid(self, filter_size):
        """
        Create the distance grid.
        @param filter_size: Size of the filter. By default equal to the size of the image.
        """
        # Set up filter shape
        if filter_size is not None:
            filter_size = np.array(filter_size)
            if self.by_slice:
                filter_shape = (filter_size[1], filter_size[2])
            else:
                filter_shape = (filter_size[0], filter_size[1], filter_size[2])
        else:
            if self.by_slice:
                filter_shape = (filter_size, filter_size)

            else:
                filter_shape = (filter_size, filter_size, filter_size)

        # Determine the grid center.
        grid_center = (np.array(filter_shape, dtype=float) - 1.0) / 2.0

        # Determine distance from center
        distance_grid = list(np.indices(filter_shape, sparse=True))
        distance_grid = [(distance_grid[ii] - center_pos) / center_pos for ii, center_pos in enumerate(grid_center)]

        # Compute the distances in the grid.
        distance_grid = np.linalg.norm(distance_grid)

        # Set the Nyquist frequency
        decomposed_max_frequency = 1.0 / 2.0 ** (self.decomposition_level - 1.0)

        return distance_grid, decomposed_max_frequency

    def convolve(self, voxel_grid):

        from mirp.imageFilters.utilities import FilterSet2D, FilterSet3D

        # Create the kernel.
        if self.wavelet_family == "simoncelli":
            wavelet_kernel_f = self.simoncelli_filter(filter_size=voxel_grid.shape)

        elif self.wavelet_family == "shannon":

            wavelet_kernel_f = self.shannon_filter(filter_size=voxel_grid.shape)

        else:
            raise ValueError(f"The specified wavelet family is not implemented: {self.wavelet_family}")

        if self.by_slice:
            # Create filter set, and assign wavelet filter. Note the ifftshift that is present to go from a centric
            # to quadrant FFT representation.
            filter_set = FilterSet2D(filter_set=fft.ifftshift(wavelet_kernel_f),
                                     transformed=True,
                                     pad_image=False,
                                     riesz_order=self.riesz_order,
                                     riesz_steered=self.riesz_steered,
                                     riesz_sigma=self.riesz_sigma)

            # Create the response map.
            response_map = filter_set.convolve(voxel_grid=voxel_grid,
                                               mode=self.mode,
                                               response=self.response,
                                               axis=0)
        else:
            # Create filter set, and assign wavelet filter. Note the ifftshift that is present to go from a centric
            # to quadrant FFT representation.
            filter_set = FilterSet3D(filter_set=fft.ifftshift(wavelet_kernel_f),
                                     transformed=True,
                                     pad_image=False,
                                     riesz_order=self.riesz_order,
                                     riesz_steered=self.riesz_steered,
                                     riesz_sigma=self.riesz_sigma)

            # Create the response map.
            response_map = filter_set.convolve(voxel_grid=voxel_grid,
                                               mode=self.mode,
                                               response=self.response)

        return response_map
