import numpy as np
import scipy.fft as fft

from mirp.imageClass import ImageClass
from mirp.imageProcess import calculate_features
from mirp.importSettings import SettingsClass
from mirp.imageFilters.utilities import pool_voxel_grids, SeparableFilterSet


class WaveletFilter:

    def __init__(self, settings: SettingsClass):
        import pywt

        # In-slice (2D) or 3D wavelet filters
        self.by_slice = settings.general.by_slice

        # Set wavelet family
        self.wavelet_family = settings.img_transform.wavelet_fam

        # Set separability of the wavelet.
        self.is_separable = self.wavelet_family in pywt.wavelist(kind="discrete")

        # Set the filter set for separable wavelets.
        self.filter_config = settings.img_transform.wavelet_filter_set

        # Set filter size for non-separable wavelets
        self.filter_size = None  # Deprecated as external input.

        if self.filter_config is None:
            self.filter_config = ["all"]

        if "all" in self.filter_config:
            if self.is_separable:
                if self.by_slice:
                    self.filter_config = ["HH", "HL", "LH", "LL"]
                else:
                    self.filter_config = ["HHH", "HHL", "HLH", "LHH", "LLH", "LHL", "HLL", "LLL"]

            else:
                self.filter_config = ["B"]

        # Set rotational invariance
        self.rot_invariance = settings.img_transform.wavelet_rot_invar

        # Which pooling method is used.
        self.pooling_method = settings.img_transform.wavelet_pooling_method

        # Wavelet cascade type
        self.stationary_wavelet = settings.img_transform.wavelet_stationary

        # Wavelet decomposition level
        self.decomposition_level = settings.img_transform.wavelet_decomposition_level

        # Set boundary condition
        self.mode = settings.img_transform.boundary_condition

    def apply_transformation(self,
                             img_obj: ImageClass,
                             roi_list,
                             settings: SettingsClass,
                             compute_features=False,
                             extract_images=False,
                             file_path=None):
        """Run feature computation and/or image extraction for transformed data"""
        feat_list = []

        # Iterate over wavelet filters
        for filter_configuration in self.filter_config:
            for decomposition_level in self.decomposition_level:

                # Make a copy of the rois.
                roi_trans_list = [roi_obj.copy() for roi_obj in roi_list]

                # Transform the image.
                img_trans_obj = self.transform(img_obj=img_obj,
                                               filter_configuration=filter_configuration,
                                               decomposition_level=decomposition_level)

                # Decimate the rois in case the wavelets are not stationary
                if not self.stationary_wavelet:
                    for ii in np.arange(decomposition_level):
                        [roi_obj.decimate(by_slice=self.by_slice) for roi_obj in roi_trans_list]

                # Export image
                if extract_images:
                    img_trans_obj.export(file_path=file_path)

                # Compute features
                if compute_features:
                    feat_list += [calculate_features(img_obj=img_trans_obj, roi_list=roi_trans_list, settings=settings,
                                                     append_str=img_trans_obj.spat_transform + "_")]
                # Clean up
                del img_trans_obj, roi_trans_list

        return feat_list

    def transform(self, img_obj: ImageClass, filter_configuration, decomposition_level):

        # Treat separable and non-separable wavelets differently.
        if self.is_separable:
            img_wav_obj = self.transform_separable(img_obj=img_obj,
                                                   filter_configuration=filter_configuration,
                                                   decomposition_level=decomposition_level)

        else:
            img_wav_obj = self.transform_non_separable(img_obj=img_obj,
                                                       filter_configuration=filter_configuration,
                                                       decomposition_level=decomposition_level)

        return img_wav_obj

    def transform_separable(self, img_obj: ImageClass, filter_configuration, decomposition_level):
        # Copy base image
        img_wav_obj = img_obj.copy(drop_image=True)

        # Prepare the string for the spatial transformation.
        spat_transform = ["wavelet", self.wavelet_family, filter_configuration]
        if not self.stationary_wavelet:
            spat_transform += ["decimated"]
        if self.rot_invariance:
            spat_transform += ["invar"]
        spat_transform += ["level", str(decomposition_level)]

        # Set the name of the transformation.
        img_wav_obj.set_spatial_transform("_".join(spat_transform))

        # Skip transformation in case the input image is missing
        if img_obj.is_missing:
            return img_wav_obj

        # Create empty voxel grid
        img_voxel_grid = np.zeros(img_obj.size, dtype=np.float32)

        # Create the list of filters from the configuration.
        main_filter_set = self.get_filter_set(filter_configuration=filter_configuration)
        filter_list = main_filter_set.permute_filters(rotational_invariance=self.rot_invariance,
                                                      require_pre_filter=decomposition_level > 1)

        # Iterate over the filters.
        for ii, filter_set in enumerate(filter_list):

            # Extract the voxel grid.
            img_wavelet_grid = img_obj.get_voxel_grid()

            for decomp_level in np.arange(decomposition_level):

                # Determine whether the pre-filter should be applied. Note that the baseline decomposition level is 1,
                # not 0. We therefore subtract 1 because Python counts from 0.
                use_pre_filter = decomp_level < decomposition_level - 1

                # Convolve and compute the response map.
                img_wavelet_grid = filter_set.convolve(voxel_grid=img_wavelet_grid,
                                                       mode=self.mode,
                                                       use_pre_filter=use_pre_filter)

                if use_pre_filter:
                    # Decompose the filter set for the next level.
                    filter_set.decompose_filter()

            # Perform pooling
            if ii == 0:
                # Initially, set img_voxel_grid.
                img_voxel_grid = img_wavelet_grid
            else:
                # Pool grids.
                img_voxel_grid = pool_voxel_grids(x1=img_voxel_grid, x2=img_wavelet_grid,
                                                  pooling_method=self.pooling_method)

                # Remove img_wavelet_grid to explicitly release memory when collecting garbage.
                del img_wavelet_grid

        if self.pooling_method == "mean":
            # Perform final pooling step for mean pooling.
            img_voxel_grid = np.divide(img_voxel_grid, len(filter_list))

        # Store the voxel grid in the ImageObject.
        img_wav_obj.set_voxel_grid(voxel_grid=img_voxel_grid)

        return img_wav_obj

    def transform_non_separable(self, img_obj: ImageClass, filter_configuration, decomposition_level):
        # Copy base image
        img_wav_obj = img_obj.copy(drop_image=True)

        # Prepare the string for the spatial transformation.
        spat_transform = ["wavelet", self.wavelet_family, filter_configuration]
        if not self.stationary_wavelet:
            spat_transform += ["decimated"]
        spat_transform += ["level", str(decomposition_level)]

        # Set the name of the transformation.
        img_wav_obj.set_spatial_transform("_".join(spat_transform))

        # Skip transformation in case the input image is missing
        if img_obj.is_missing:
            return img_wav_obj

        if self.wavelet_family in ["simoncelli", "shannon"]:
            filter_set = NonSeparableWavelet(by_slice=self.by_slice,
                                             mode=self.mode,
                                             wavelet_family=self.wavelet_family,
                                             filter_size=self.filter_size)

        else:
            raise ValueError(f"{self.wavelet_family} is not a known separable wavelet.")

        # Create voxel grid
        img_wavelet_grid = filter_set.convolve(voxel_grid=img_obj.get_voxel_grid(),
                                               decomposition_level=decomposition_level)

        # Store the voxel grid in the ImageObject.
        img_wav_obj.set_voxel_grid(voxel_grid=img_wavelet_grid)

        return img_wav_obj

    def get_filter_set(self, filter_configuration):
        import pywt
        from copy import deepcopy

        # Deparse convolution kernels to a list
        kernel_list = [filter_configuration[ii:ii + 1] for ii in range(0, len(filter_configuration), 1)]

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


class NonSeparableWavelet:
    def __init__(self, by_slice, mode, wavelet_family, filter_size, response="real"):
        self.by_slice = by_slice
        self.wavelet_family = wavelet_family
        self.filter_size = filter_size
        self.max_frequency = 1.0

        # # Modes in scipy and numpy are defined differently.
        # if mode == "reflect":
        #     mode = "symmetric"
        # elif mode == "symmetric":
        #     mode = "reflect"
        # else:
        #     mode = mode

        self.mode = mode
        self.response = response

    def shannon_filter(self, decomposition_level=1, filter_size=None):
        """
        Set up the shannon filter in the Fourier domain.
        @param decomposition_level: Decomposition level for the filter.
        @param filter_size: Size of the filter. By default equal to the size of the image.
        """

        # Get the distance grid.
        distance_grid, max_frequency = self.get_distance_grid(decomposition_level=decomposition_level,
                                                              filter_size=filter_size)

        # Set up a wavelet filter for the decomposition specifically.
        wavelet_filter = np.zeros(distance_grid.shape, dtype=np.float)

        # Set the mask for the filter.
        mask = np.logical_and(distance_grid >= max_frequency / 2.0, distance_grid <= max_frequency)

        # Update the filter.
        wavelet_filter[mask] += 1.0

        return wavelet_filter

    def simoncelli_filter(self, decomposition_level=1, filter_size=None):
        """
        Set up the simoncelli filter in the Fourier domain.
        @param decomposition_level: Decomposition level for the filter.
        @param filter_size: Size of the filter. By default equal to the size of the image.
        """

        # Get the distance grid.
        distance_grid, max_frequency = self.get_distance_grid(decomposition_level=decomposition_level,
                                                              filter_size=filter_size)

        # Set up a wavelet filter for the decomposition specifically.
        wavelet_filter = np.zeros(distance_grid.shape, dtype=np.float)

        # Set the mask for the filter.
        mask = np.logical_and(distance_grid >= max_frequency / 4.0,
                              distance_grid <= max_frequency)

        # Update the filter.
        wavelet_filter[mask] += np.cos(np.pi / 2.0 * np.log2(2.0 * distance_grid[mask] / max_frequency))

        return wavelet_filter

    def get_distance_grid(self, decomposition_level=1, filter_size=None):
        """
        Create the distance grid.
        @param decomposition_level: Decomposition level for the filter.
        @param filter_size: Size of the filter. By default equal to the size of the image.
        """
        # Set up filter shape
        if filter_size is not None:
            self.filter_size = np.array(filter_size)
            if self.by_slice:
                filter_shape = (self.filter_size[1], self.filter_size[2])
            else:
                filter_shape = (self.filter_size[0], self.filter_size[1], self.filter_size[2])
        else :
            if self.by_slice:
                filter_shape = (self.filter_size, self.filter_size)

            else:
                filter_shape = (self.filter_size, self.filter_size, self.filter_size)

        # Determine the grid center.
        grid_center = (np.array(filter_shape, dtype=np.float) - 1.0) / 2.0

        # Determine distance from center
        distance_grid = list(np.indices(filter_shape, sparse=True))
        distance_grid = [(distance_grid[ii] - center_pos) / center_pos for ii, center_pos in enumerate(grid_center)]

        # Compute the distances in the grid.
        distance_grid = np.linalg.norm(distance_grid)

        # Set the Nyquist frequency
        decomposed_max_frequency = self.max_frequency / 2.0 ** (decomposition_level - 1.0)

        return distance_grid, decomposed_max_frequency

    def convolve(self, voxel_grid, decomposition_level=1):

        from mirp.imageFilters.utilities import FilterSet2D, FilterSet3D

        # Create the kernel.
        if self.wavelet_family == "simoncelli":
            wavelet_kernel_f = self.simoncelli_filter(decomposition_level=decomposition_level,
                                                      filter_size=voxel_grid.shape)
        elif self.wavelet_family == "shannon":
            wavelet_kernel_f = self.shannon_filter(decomposition_level=decomposition_level,
                                                   filter_size=voxel_grid.shape)
        else:
            raise ValueError(f"The specified wavelet family is not implemented: {self.wavelet_family}")

        if self.by_slice:
            # Create filter set, and assign wavelet filter. Note the ifftshift that is present to go from a centric
            # to quadrant FFT representation.
            filter_set = FilterSet2D(filter_set=fft.ifftshift(wavelet_kernel_f),
                                     transformed=True,
                                     pad_image=False)

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
                                     pad_image=False)

            # Create the response map.
            response_map = filter_set.convolve(voxel_grid=voxel_grid,
                                               mode=self.mode,
                                               response=self.response)

        return response_map
