import numpy as np
import copy

from mirp._images.genericImage import GenericImage
from mirp._images.transformedImage import LawsTransformedImage
from mirp._imagefilters.generic import GenericFilter
from mirp._imagefilters.utilities import SeparableFilterSet, pool_voxel_grids
from mirp.settings.settingsGeneric import SettingsClass


class LawsFilter(GenericFilter):
    def __init__(self, settings: SettingsClass, name: str):

        super().__init__(
            settings=settings,
            name=name
        )

        # Normalise kernel and energy filters? This is true by default (see IBSI).
        self.kernel_normalise = True
        self.energy_normalise = True

        # Set the filter name
        self.laws_kernel: None | str | list[str] = settings.img_transform.laws_kernel

        # Size of neighbourhood in chebyshev distance from center voxel
        self.delta: None | int | list[int] = settings.img_transform.laws_delta

        # Whether Laws texture energy should be calculated
        self.calculate_energy = settings.img_transform.laws_calculate_energy

        # Whether response maps or texture energy _images should be made rotationally invariant
        self.rotation_invariance = settings.img_transform.laws_rotation_invariance

        # Which pooling method is used.
        self.pooling_method = settings.img_transform.laws_pooling_method

        # Set boundary condition
        self.mode = settings.img_transform.laws_boundary_condition

    def generate_object(self):
        # Generator for transformation objects.
        laws_kernel = copy.deepcopy(self.laws_kernel)
        if not isinstance(laws_kernel, list):
            laws_kernel = [laws_kernel]

        delta = copy.deepcopy(self.delta)
        if not isinstance(delta, list):
            delta = [delta]

        if not self.calculate_energy:
            delta = [None]

        # Iterate over options to yield filter objects with specific settings. A copy of the parent object is made to
        # avoid updating by reference.
        for current_laws_kernel in laws_kernel:
            for current_delta in delta:
                filter_object = copy.deepcopy(self)
                filter_object.laws_kernel = current_laws_kernel
                filter_object.delta = current_delta

                yield filter_object

    def transform(self, image: GenericImage) -> LawsTransformedImage:
        # Create placeholder Laws kernel response map.
        response_map = LawsTransformedImage(
            image_data=None,
            laws_kernel=self.laws_kernel,
            delta_parameter=self.delta,
            energy_map=self.calculate_energy,
            rotation_invariance=self.rotation_invariance,
            pooling_method=self.pooling_method,
            boundary_condition=self.mode,
            riesz_order=None,
            riesz_steering=None,
            riesz_sigma_parameter=None,
            template=image
        )

        if image.is_empty():
            return response_map

        # Initialise voxel grid.
        response_voxel_grid = None

        # Get filter list.
        filter_set_list: list[SeparableFilterSet] = self.get_filter_set().permute_filters(
            rotational_invariance=self.rotation_invariance)

        for ii, filter_set in enumerate(filter_set_list):
            # Convolve and compute response map.
            pooled_voxel_grid = filter_set.convolve(
                voxel_grid=image.get_voxel_grid(),
                mode=self.mode
            )

            # Pool grids.
            response_voxel_grid = pool_voxel_grids(
                x1=response_voxel_grid,
                x2=pooled_voxel_grid,
                pooling_method=self.pooling_method
            )

            # Remove img_laws_grid to explicitly release memory when collecting garbage.
            del pooled_voxel_grid

        if self.pooling_method == "mean":
            # Perform final pooling step for mean pooling.
            response_voxel_grid = np.divide(response_voxel_grid, len(filter_set_list))

        # Compute energy map from the response map.
        if self.calculate_energy:
            response_voxel_grid = self.response_to_energy(voxel_grid=response_voxel_grid)

        # Set voxel grid.
        response_map.set_voxel_grid(voxel_grid=response_voxel_grid)

        return response_map

    def response_to_energy(self, voxel_grid):

        # Take absolute value of the voxel grid.
        voxel_grid = np.abs(voxel_grid)

        # Set the filter size.
        filter_size = 2 * self.delta + 1

        # Set up the filter kernel.
        if self.energy_normalise:
            filter_kernel = np.ones(filter_size, dtype=float) / filter_size
        else:
            filter_kernel = np.ones(filter_size, dtype=float)

        # Create a filter set.
        if self.by_slice:
            filter_set = SeparableFilterSet(
                filter_x=filter_kernel,
                filter_y=filter_kernel)
        else:
            filter_set = SeparableFilterSet(
                filter_x=filter_kernel,
                filter_y=filter_kernel,
                filter_z=filter_kernel)

        # Apply the filter.
        voxel_grid = filter_set.convolve(voxel_grid=voxel_grid, mode=self.mode)

        return voxel_grid

    def get_filter_set(self):

        # Get kernels
        kernels: str = copy.deepcopy(self.laws_kernel)

        # Deparse kernels to a list
        kernel_list = [kernels[ii:ii + 2] for ii in range(0, len(kernels), 2)]

        filter_x = None
        filter_y = None
        filter_z = None

        for ii, kernel in enumerate(kernel_list):
            if kernel.lower() == "l5":
                laws_kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
            elif kernel.lower() == "e5":
                laws_kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])
            elif kernel.lower() == "s5":
                laws_kernel = np.array([-1.0, 0.0, 2.0, 0.0, -1.0])
            elif kernel.lower() == "w5":
                laws_kernel = np.array([-1.0, 2.0, 0.0, -2.0, 1.0])
            elif kernel.lower() == "r5":
                laws_kernel = np.array([1.0, -4.0, 6.0, -4.0, 1.0])
            elif kernel.lower() == "l3":
                laws_kernel = np.array([1.0, 2.0, 1.0])
            elif kernel.lower() == "e3":
                laws_kernel = np.array([-1.0, 0.0, 1.0])
            elif kernel.lower() == "s3":
                laws_kernel = np.array([-1.0, 2.0, -1.0])
            else:
                raise ValueError(f"{kernel} is not an implemented Laws kernel")

            # Normalise kernel
            if self.kernel_normalise:
                laws_kernel /= np.sqrt(np.sum(np.power(laws_kernel, 2.0)))

            # Assign filter to variable.
            if ii == 0:
                filter_x = laws_kernel
            elif ii == 1:
                filter_y = laws_kernel
            elif ii == 2:
                filter_z = laws_kernel

        # Create FilterSet object
        return SeparableFilterSet(filter_x=filter_x,
                                  filter_y=filter_y,
                                  filter_z=filter_z)
