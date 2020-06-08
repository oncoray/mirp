import numpy as np

from mirp.imageClass import ImageClass
from mirp.imageProcess import calculate_features
from mirp.imageFilters.utilities import FilterSet, pool_voxel_grids


class LawsFilter:
    def __init__(self, settings):

        # Normalise kernel and energy filters? This is true by default (see IBSI).
        self.kernel_normalise = True
        self.energy_normalise = True

        # Set the filter name
        self.name = settings.img_transform.laws_kernel

        # Whether response maps or texture energy images should be made rotationally invariant
        self.rot_invariance = settings.img_transform.laws_rot_invar

        # Which pooling method is used.
        self.pooling_method = settings.img_transform.laws_pooling_method

        # Size of neighbourhood in chebyshev distance from center voxel
        self.delta = settings.img_transform.laws_delta

        # 2D or 3D filter
        self.by_slice = settings.general.by_slice

        # Whether Laws texture energy should be calculated
        self.calculate_energy = settings.img_transform.laws_calculate_energy

        # Get the laws filter set and permute, if required.
        filter_set = self.get_filter_set(kernels=settings.img_transform.laws_kernel)
        self.filter_list = filter_set.permute_filters(rotational_invariance=self.rot_invariance)

        # Set boundary condition
        self.mode = settings.img_transform.boundary_condition

    def apply_transformation(self,
                             img_obj: ImageClass,
                             roi_list,
                             settings,
                             compute_features=False,
                             extract_images=False,
                             file_path=None):

        feat_list = []

        # Copy roi list
        roi_trans_list = [roi_obj.copy() for roi_obj in roi_list]

        # Add spatially transformed image object.
        img_trans_obj = self.transform(img_obj=img_obj)

        # Export image
        if extract_images:
            img_trans_obj.export(file_path=file_path)

        # Compute features
        if compute_features:
            feat_list += [calculate_features(img_obj=img_trans_obj,
                                             roi_list=roi_trans_list,
                                             settings=settings,
                                             append_str=img_trans_obj.spat_transform + "_")]
        # Clean up
        del img_trans_obj

        return feat_list

    def transform(self, img_obj):

        # Copy base image
        img_laws_obj = img_obj.copy(drop_image=True)

        # Set spatial transformation filter string
        spat_transform = ["laws", self.name]
        if self.calculate_energy:
            spat_transform += ["energy", "delta", str(self.delta)]
        if self.rot_invariance:
            spat_transform += ["invar"]

        # Set the name of the transform.
        img_laws_obj.set_spatial_transform("_".join(spat_transform))

        # Skip transformation in case the input image is missing
        if img_obj.is_missing:
            return img_laws_obj

        # Create empty voxel grid
        img_voxel_grid = np.zeros(img_obj.size, dtype=np.float32)

        for ii, filter_set in enumerate(self.filter_list):

            # Convolve and compute response map.
            img_laws_grid = filter_set.convolve(voxel_grid=img_obj.get_voxel_grid(),
                                                mode=self.mode)

            # Compute energy map from the response map.
            if self.calculate_energy:
                img_laws_grid = self.response_to_energy(voxel_grid=img_laws_grid)

            # Perform pooling
            if ii == 0:
                # Initially, update img_voxel_grid.
                img_voxel_grid = img_laws_grid
            else:
                # Pool grids.
                img_voxel_grid = pool_voxel_grids(x1=img_voxel_grid, x2=img_laws_grid,
                                                  pooling_method=self.pooling_method)

            # Remove img_laws_grid to explicitly release memory when collecting garbage.
            del img_laws_grid

        if self.pooling_method == "mean":
            # Perform final pooling step for mean pooling.
            img_voxel_grid = np.divide(img_voxel_grid, len(self.filter_list))

        # Store the voxel grid in the ImageObject.
        img_laws_obj.set_voxel_grid(voxel_grid=img_voxel_grid)

        return img_laws_obj

    def response_to_energy(self, voxel_grid):

        # Take absolute value of the voxel grid.
        voxel_grid = np.abs(voxel_grid)

        # Set the filter size.
        filter_size = 2 * self.delta + 1

        # Set up the filter kernel.
        if self.energy_normalise:
            filter_kernel = np.ones(filter_size, dtype=np.float) / filter_size
        else:
            filter_kernel = np.ones(filter_size, dtype=np.float)

        # Create a filter set.
        if self.by_slice:
            filter_set = FilterSet(filter_x=filter_kernel, filter_y=filter_kernel)
        else:
            filter_set = FilterSet(filter_x=filter_kernel, filter_y=filter_kernel, filter_z=filter_kernel)

        # Apply the filter.
        voxel_grid = filter_set.convolve(voxel_grid=voxel_grid, mode=self.mode)

        return voxel_grid

    def get_filter_set(self, kernels=None):

        # Deparse kernels to a list
        kernel_list = [kernels[ii:ii + 2] for ii in range(0, len(kernels), 2)]

        filter_x = None
        filter_y = None
        filter_z = None

        for ii, kernel in enumerate(kernel_list):
            if kernel == "L5":
                laws_kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
            elif kernel == "E5":
                laws_kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])
            elif kernel == "S5":
                laws_kernel = np.array([-1.0, 0.0, 2.0, 0.0, -1.0])
            elif kernel == "W5":
                laws_kernel = np.array([-1.0, 2.0, 0.0, -2.0, 1.0])
            elif kernel == "R5":
                laws_kernel = np.array([1.0, -4.0, 6.0, -4.0, 1.0])
            elif kernel == "L3":
                laws_kernel = np.array([1.0, 2.0, 1.0])
            elif kernel == "E3":
                laws_kernel = np.array([-1.0, 0.0, 1.0])
            elif kernel == "S3":
                laws_kernel = np.array([-1.0, 2.0, -1.0])
            else:
                raise ValueError("%s is not an implemented Laws kernel")

            # Normalise kernel
            if self.kernel_normalise:
                laws_kernel /= np.sum(np.abs(laws_kernel))

            # Assign filter to variable.
            if ii == 0:
                filter_x = laws_kernel
            elif ii == 1:
                filter_y = laws_kernel
            elif ii == 2:
                filter_z = laws_kernel

        # Create FilterSet object
        return FilterSet(filter_x=filter_x,
                         filter_y=filter_y,
                         filter_z=filter_z)
