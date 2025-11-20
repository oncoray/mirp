from ctypes import c_int16

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

        self.separate_slices = None
        self.stencil_size = settings.img_transform.laplace_stencil_size
        self.mode = settings.img_transform.laplace_boundary_condition

    def generate_object(self):
        # Generator for transformation objects.
        stencil_size = copy.deepcopy(self.stencil_size)
        if not isinstance(stencil_size, list):
            stencil_size = [stencil_size]

        for current_stencil_size in stencil_size:
            filter_object = copy.deepcopy(self)
            filter_object.stencil_size = current_stencil_size

            # 5 and 9-stencil filters are 2D, whereas 7, 15, 19, 21, and 27-stencils are 3D.
            filter_object.separate_slices = current_stencil_size in [5, 9]

            yield filter_object

    def transform(self, image: GenericImage) -> LaplacianTransformedImage:
        # Create placeholder Laplacian-of-Gaussian response map.
        response_map = LaplacianTransformedImage(
            image_data=None,
            stencil_size=self.stencil_size,
            boundary_condition=self.mode,
            template=image
        )
        response_map.ibsi_compliant = self.ibsi_compliant and image.ibsi_compliant

        if image.is_empty():
            return response_map

        # Set response voxel grid.
        response_voxel_grid = None

        # Initialise iterator ii to avoid IDE warnings.
        for ii, pooled_filter_object in enumerate(self.generate_object()):
            # Generate transformed voxel grid.
            response_voxel_grid = pooled_filter_object.transform_grid(
                voxel_grid=image.get_voxel_grid()
            )

            if ii > 1:
                raise ValueError(f"Laplace response maps cannot be stacked.")

        # Set voxel grid.
        response_map.set_voxel_grid(voxel_grid=response_voxel_grid)

        return response_map

    def transform_grid(
            self,
            voxel_grid: np.ndarray
    ):
        # See Patra and Karttunen (10.1002/num.20129) for constants.
        if self.separate_slices:
            # two-dimensional filters.
            if self.stencil_size == 5:
                # Anisotropic filter.
                c1 = 0.0
            elif self.stencil_size == 9:
                # Isotropic filter.
                c1 = 1.0 / 6.0
            else:
                raise ValueError(f"stencil size is not valid. Found: {self.stencil_size}. Expected: 5 or 7")

            c2 = 1.0 - 2.0 * c1
            c3 = 4.0 * c1 - 4.0

            filter_weights = np.array([
                [c1, c2, c1],
                [c2, c3, c2],
                [c1, c2, c1]
            ])

            # Set filter weights and create a filter.
            laplace_filter = FilterSet2D(filter_weights)

        else:
            # three-dimensional filters.
            if self.stencil_size == 7:
                # Anisotropic filter.
                c1 = 0.0

            elif self.stencil_size == 15:
                # Isotropic filter.
                c1 = 1.0 / 12.0

            elif self.stencil_size == 19:
                c1 = 0.0

            elif self.stencil_size == 21:
                c1 = -1.0 / 12.0

            elif self.stencil_size == 27:
                c1 = 1.0 / 30.0

            else:
                raise ValueError(f"stencil size is not valid. Found: {self.stencil_size}. Expected: 7, 15, 19, 21, "
                                 f"or 27.")

            c2 = 1.0 / 6.0 - 2.0 * c1
            c3 = 1.0 / 3.0 + 4.0 * c1
            c4 = -4.0 - 8.0 * c1
            if self.stencil_size == 7:
                c2 = 0.0
                c3 = 1.0
                c4 = -6.0

            filter_weights = np.array([
                [
                    [c1, c2, c1],
                    [c2, c3, c2],
                    [c1, c2, c1]
                ], [
                    [c2, c3, c2],
                    [c3, c4, c3],
                    [c2, c3, c2]
                ], [
                    [c1, c2, c1],
                    [c2, c3, c2],
                    [c1, c2, c1]
                ]
            ])

            # Set filter weights and create a filter.
            laplace_filter = FilterSet3D(filter_weights)

        # Convolve laplace filter with the image.
        response_map = laplace_filter.convolve(
            voxel_grid=voxel_grid,
            mode=self.mode,
            response="real"
        )

        # Compute the convolution
        return response_map