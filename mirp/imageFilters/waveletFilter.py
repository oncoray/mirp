import numpy as np

from mirp.imageClass import ImageClass
from mirp.imageProcess import calculate_features
from mirp.importSettings import SettingsClass
from mirp.imageFilters.utilities import pool_voxel_grids, FilterSet


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
        if "all" in self.filter_config:
            if self.by_slice:
                self.filter_config = ["HH", "HL", "LH", "LL"]
            else:
                self.filter_config = ["HHH", "HHL", "HLH", "LHH", "LLH", "LHL", "HLL", "LLL"]

        if not self.is_separable:
            self.filter_config = ["default"]

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
            img_wav_obj = self.transform_non_separable(img_obj=img_obj, decomposition_level=decomposition_level)

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

        # Create the low-pass image for the current configuration.
        input_img_obj = self.create_initial_separable_response_map(img_obj=img_obj,
                                                                   decomposition_level=decomposition_level)

        # Create empty voxel grid
        img_voxel_grid = np.zeros(input_img_obj.size, dtype=np.float32)

        # Create the list of filters from the configuration.
        main_filter_set = self.get_filter_set(filter_configuration=filter_configuration,
                                              decomposition_level=decomposition_level)
        filter_list = main_filter_set.permute_filters(rotational_invariance=self.rot_invariance)

        # Iterate over the filters.
        for ii, filter_set in enumerate(filter_list):

            # Convolve and compute response map.
            img_wavelet_grid = filter_set.convolve(voxel_grid=input_img_obj.get_voxel_grid(),
                                                   mode=self.mode)

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

    def transform_non_separable(self, img_obj: ImageClass, decomposition_level):
        # Copy base image
        img_wav_obj = img_obj.copy(drop_image=True)

        # Prepare the string for the spatial transformation.
        spat_transform = ["wavelet", self.wavelet_family]
        if not self.stationary_wavelet:
            spat_transform += ["decimated"]
        spat_transform += ["level", str(decomposition_level)]

        # Set the name of the transformation.
        img_wav_obj.set_spatial_transform("_".join(spat_transform))

        # Skip transformation in case the input image is missing
        if img_obj.is_missing:
            return img_wav_obj

        # Create the low-pass image for the current configuration.
        input_img_obj = img_obj
        # TODO: implement

        # TODO: missing code - Stefan

        return img_wav_obj

    def get_filter_set(self, filter_configuration, decomposition_level=1):
        import pywt

        # Deparse convolution kernels to a list
        kernel_list = [filter_configuration[ii:ii + 1] for ii in range(0, len(filter_configuration), 1)]

        filter_x = None
        filter_y = None
        filter_z = None

        for ii, kernel in enumerate(kernel_list):
            if kernel.lower() == "l":
                wavelet_kernel = np.array(pywt.Wavelet(self.wavelet_family).dec_lo)
            elif kernel.lower() == "h":
                wavelet_kernel = np.array(pywt.Wavelet(self.wavelet_family).dec_hi)
            else:
                raise ValueError(f"{kernel} was not recognised as the component of a separable wavelet filter. It "
                                 f"should be L or H.")

            # Add in 0s for the à trous algorithm
            if decomposition_level > 1:
                for jj in np.arange(start=1, stop=decomposition_level):
                    new_wavelet_kernel = np.zeros(len(wavelet_kernel) * 2 - 1, dtype=np.float)
                    new_wavelet_kernel[::2] = wavelet_kernel
                    wavelet_kernel = new_wavelet_kernel

            # Assign filter to variable.
            if ii == 0:
                filter_x = wavelet_kernel
            elif ii == 1:
                filter_y = wavelet_kernel
            elif ii == 2:
                filter_z = wavelet_kernel

        # Create FilterSet object
        return FilterSet(filter_x=filter_x,
                         filter_y=filter_y,
                         filter_z=filter_z)

    def create_initial_separable_response_map(self, img_obj: ImageClass, decomposition_level=1):

        # If the decomposition level equals 1, use the initial image:
        if decomposition_level == 1:
            return img_obj.copy()

        # Create empty voxel grid
        img_voxel_grid = np.zeros(img_obj.size, dtype=np.float32)

        # Set the filter configuration. These are low-pass filters.
        if self.by_slice:
            filter_configuration = "ll"
        else:
            filter_configuration = "lll"

        # Get the voxel grid from the original dataset.
        main_voxel_grid = img_obj.get_voxel_grid()

        for current_level in np.arange(start=1, stop=decomposition_level):

            # Create the list of filters from the configuration.
            main_filter_set = self.get_filter_set(filter_configuration=filter_configuration,
                                                  decomposition_level=current_level)
            filter_list = main_filter_set.permute_filters(rotational_invariance=self.rot_invariance)

            for ii, filter_set in enumerate(filter_list):

                # Convolve and compute response map.
                img_wavelet_grid = filter_set.convolve(voxel_grid=main_voxel_grid,
                                                       mode=self.mode)

                # Perform pooling
                if ii == 0:
                    # Initially, set img_voxel_grid.
                    img_voxel_grid = img_wavelet_grid
                else:
                    # Pool grids.
                    img_voxel_grid = pool_voxel_grids(x1=img_voxel_grid, x2=img_wavelet_grid,
                                                      pooling_method="max")

                    # Remove img_wavelet_grid to explicitly release memory when collecting garbage.
                    del img_wavelet_grid

            # Set main voxel grid.
            main_voxel_grid = img_voxel_grid

        # Copy the image object
        img_decomp_obj = img_obj.copy(drop_image=True)

        # Store the voxel grid in the ImageObject.
        img_decomp_obj.set_voxel_grid(voxel_grid=main_voxel_grid)

        return img_decomp_obj
