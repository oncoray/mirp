import numpy as np
import copy

from mirp._images.generic_image import GenericImage
from mirp._images.transformed_image import SeparableWaveletTransformedImage
from mirp.settings.generic import SettingsClass
from mirp._imagefilters.generic import GenericFilter
from mirp._imagefilters.utilities import pool_voxel_grids, SeparableFilterSet


class SeparableWaveletFilter(GenericFilter):

    def __init__(self, settings: SettingsClass, name: str):

        super().__init__(
            settings=settings,
            name=name
        )

        self.ibsi_compliant = True
        self.ibsi_id = "25BO"

        # Set wavelet family
        self.wavelet_family: str | list[str] = settings.img_transform.separable_wavelet_families

        # Wavelet decomposition level
        self.decomposition_level: int | list[int] = settings.img_transform.separable_wavelet_decomposition_level

        # Set the filter set for separable wavelets.
        self.filter_configuration: str | list[str] = settings.img_transform.separable_wavelet_filter_set

        # Set rotational invariance
        self.rotational_invariance = settings.img_transform.separable_wavelet_rotation_invariance

        # Which pooling method is used.
        self.pooling_method = settings.img_transform.separable_wavelet_pooling_method

        # Wavelet cascade type
        self.stationary_wavelet = settings.img_transform.separable_wavelet_stationary

        # Set boundary condition
        self.mode = settings.img_transform.separable_wavelet_boundary_condition

    def generate_object(self):
        # Generator for transformation objects.
        wavelet_family = copy.deepcopy(self.wavelet_family)
        if not isinstance(wavelet_family, list):
            wavelet_family = [wavelet_family]

        filter_configuration = copy.deepcopy(self.filter_configuration)
        if not isinstance(filter_configuration, list):
            filter_configuration = [filter_configuration]

        decomposition_level = copy.deepcopy(self.decomposition_level)
        if not isinstance(decomposition_level, list):
            decomposition_level = [decomposition_level]

        # Iterate over options to yield filter objects with specific settings. A copy of the parent object is made to
        # avoid updating by reference.
        for current_wavelet_family in wavelet_family:
            for current_filter_configuration in filter_configuration:
                for current_decomposition_level in decomposition_level:
                    filter_object = copy.deepcopy(self)
                    filter_object.wavelet_family = current_wavelet_family
                    filter_object.filter_configuration = current_filter_configuration
                    filter_object.decomposition_level = current_decomposition_level

                    yield filter_object

    def transform(self, image: GenericImage) -> SeparableWaveletTransformedImage:
        # Create placeholder separable wavelet response map.
        response_map = SeparableWaveletTransformedImage(
            image_data=None,
            wavelet_family=self.wavelet_family,
            decomposition_level=self.decomposition_level,
            filter_kernel_set=self.filter_configuration,
            stationary_wavelet=self.stationary_wavelet,
            rotation_invariance=self.rotational_invariance,
            pooling_method=self.pooling_method,
            boundary_condition=self.mode,
            riesz_order=None,
            riesz_steering=None,
            riesz_sigma_parameter=None,
            template=image
        )
        response_map.ibsi_compliant = self.ibsi_compliant and image.ibsi_compliant

        if image.is_empty():
            return response_map

        # Initialise voxel grid.
        response_voxel_grid = None

        # Get filter list.
        filter_set_list: list[SeparableFilterSet] = self.get_filter_set().permute_filters(
            rotational_invariance=self.rotational_invariance,
            require_pre_filter=True
        )

        for ii, filter_set in enumerate(filter_set_list):

            # Extract the voxel grid as starting point.
            pooled_voxel_grid = image.get_voxel_grid()

            for decomposition_level in np.arange(1, self.decomposition_level + 1):

                # Determine whether the pre-filter should be applied. This is the case for decomposition levels
                # smaller than self.decomposition_level.
                use_pre_filter = decomposition_level < self.decomposition_level

                # Convolve and compute the response map.
                pooled_voxel_grid = filter_set.convolve(
                    voxel_grid=pooled_voxel_grid,
                    mode=self.mode,
                    use_pre_filter=use_pre_filter
                )

                if use_pre_filter:
                    # Decompose the filter set for the next level.
                    filter_set.decompose_filter()

            # Pool grids.
            response_voxel_grid = pool_voxel_grids(
                x1=response_voxel_grid,
                x2=pooled_voxel_grid,
                pooling_method=self.pooling_method
            )

            # Remove pooled_voxel_grid to explicitly release memory when collecting garbage.
            del pooled_voxel_grid

        if self.pooling_method == "mean":
            # Perform final pooling step for mean pooling.
            response_voxel_grid = np.divide(response_voxel_grid, len(filter_set_list))

        # Store the voxel grid in the ImageObject.
        response_map.set_voxel_grid(voxel_grid=response_voxel_grid)

        return response_map

    def get_filter_set(self):
        import pywt
        from copy import deepcopy

        # Deparse convolution kernels to a list
        kernel_list = [self.filter_configuration[ii:ii + 1] for ii in range(0, len(self.filter_configuration), 1)]

        # Declare filter kernels
        filter_x, filter_y, filter_z = None, None, None
        pre_filter_x, pre_filter_y, pre_filter_z = None, None, None

        # Define the pre-filter kernel for decomposition.
        pre_filter_kernel = np.array(pywt.Wavelet(self.wavelet_family).dec_lo)

        for ii, kernel in enumerate(kernel_list):
            if kernel.lower() == "l":
                wavelet_kernel = np.array(pywt.Wavelet(self.wavelet_family).dec_lo)
            elif kernel.lower() == "h":
                wavelet_kernel = np.array(pywt.Wavelet(self.wavelet_family).dec_hi)
            else:
                raise ValueError(f"{kernel} was not recognised as the component of a separable wavelet filter. It "
                                 f"should be L or H.")

            # Assign filter to variable.
            if ii == 0:
                filter_x = wavelet_kernel
                pre_filter_x = deepcopy(pre_filter_kernel)
            elif ii == 1:
                filter_y = wavelet_kernel
                pre_filter_y = deepcopy(pre_filter_kernel)
            elif ii == 2:
                filter_z = wavelet_kernel
                pre_filter_z = deepcopy(pre_filter_kernel)

        # Create FilterSet object
        return SeparableFilterSet(filter_x=filter_x,
                                  filter_y=filter_y,
                                  filter_z=filter_z,
                                  pre_filter_x=pre_filter_x,
                                  pre_filter_y=pre_filter_y,
                                  pre_filter_z=pre_filter_z)
