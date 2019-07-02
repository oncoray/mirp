import numpy as np

from mirp.imageProcess import calculate_features


class WaveletFilter:

    def __init__(self, settings):
        self.filter_list = []

        # Set wavelet family
        self.wavelet_fam = settings.img_transform.wavelet_fam

        # Set rotational invariance
        self.rot_invariance = settings.img_transform.wavelet_rot_invar

        # Update filter_list based on input settings
        self.get_filter_order(settings=settings)

        # Wavelet cascade type
        self.stationary_wavelet = settings.img_transform.wavelet_stationary

        # Wavelet decomposition level
        self.max_decomp_level = 1

        # In-slice (2D) or 3D wavelet filters
        self.by_slice = settings.general.by_slice

    def apply_transformation(self, img_obj, roi_list, settings, compute_features=False, extract_images=False, file_path=None):
        """Run feature computation and/or image extraction for transformed data"""
        feat_list = []

        # Iterate over wavelet filters
        for current_filter_set in self.filter_list:

            # Copy roi list
            roi_trans_list = [roi_obj.copy() for roi_obj in roi_list]

            # Add spatially transformed image object. In case of rotational invariance, this is averaged.
            img_trans_obj = self.transform(img_obj=img_obj, filter_set=current_filter_set, mode=settings.img_transform.boundary_condition)

            # Decimate in case the wavelets are not stationary
            if not self.stationary_wavelet:
                img_trans_obj.decimate(by_slice=self.by_slice)
                [roi_obj.decimate(by_slice=self.by_slice) for roi_obj in roi_trans_list]

            # Export image
            if extract_images:
                img_trans_obj.export(file_path=file_path)

            # Compute features
            if compute_features:
                feat_list += [calculate_features(img_obj=img_trans_obj, roi_list=roi_trans_list, settings=settings,
                                                 append_str=img_trans_obj.spat_transform + "_")]
            # Clean up
            del img_trans_obj

        return feat_list

    def get_filter_order(self, settings):
        """
        Loads ordered list of wavelet filter orders
        :param settings:
        :return:
        """

        if settings.general.by_slice:
            self.filter_list += [["hh"], ["ll"]]

            # Rotational invariance
            if self.rot_invariance:
                self.filter_list += [["lh", "hl"]]
            else:
                self.filter_list += [["lh"], ["hl"]]

        else:
            self.filter_list += [["hhh"], ["lll"]]

            # Rotational invariance
            if self.rot_invariance:
                self.filter_list += [["llh", "lhl", "hll"]]
                self.filter_list += [["hhl", "hlh", "lhh"]]
            else:
                self.filter_list += [["llh"], ["lhl"], ["hll"]]
                self.filter_list += [["hhl"], ["hlh"], ["lhh"]]

    def transform(self, img_obj, filter_set, mode):
        """
        Applies a multidimensional stationary wavelet
            filter_order: string of H (hi-pass) and L (lo-pass), e.g. LLH
        :param img_obj:
        :param filter_set:
        :param mode:
        :return:
        """

        import pywt

        # Get filter constants for the selected wavelet
        hi_filt = np.array(pywt.Wavelet(self.wavelet_fam).dec_hi)
        lo_filt = np.array(pywt.Wavelet(self.wavelet_fam).dec_lo)

        # Copy base image
        img_wav_obj = img_obj.copy(drop_image=True)

        # Set spatial transformation string for transformed object
        if self.rot_invariance:
            img_wav_obj.spat_transform = "wav_" + self.wavelet_fam + "_" + filter_set[0] + "_invar"
        else:
            img_wav_obj.spat_transform = "wav_" + self.wavelet_fam + "_" + filter_set[0]

        # Skip transformations in case the image is missing
        if img_obj.is_missing:
            return img_wav_obj

        # Create an grid with zeros
        img_voxel_grid = np.zeros((img_wav_obj.size), dtype=np.float32)

        # Apply filters
        for current_filter in filter_set:
            img_voxel_grid += self.transform_grid(voxel_grid=img_obj.get_voxel_grid(), filter_order=current_filter, hi_filt=hi_filt, lo_filt=lo_filt, mode=mode) / (len(filter_set) * 1.0)

        # Update voxel grid
        img_wav_obj.set_voxel_grid(voxel_grid=img_voxel_grid)

        return img_wav_obj

    def transform_grid(self, voxel_grid, filter_order, hi_filt, lo_filt, mode):
        import scipy.ndimage as ndi

        # Set filters based on filter_order - note that order is interpreted as (xyz) to maintain consistency with
        # publications where (xyz) is the usual order
        if filter_order[0] == "h":
            x_filt = hi_filt
        else:
            x_filt = lo_filt
        if filter_order[1] == "h":
            y_filt = hi_filt
        else:
            y_filt = lo_filt

        if self.by_slice:
            if filter_order[2] == "h":
                z_filt = hi_filt
            else:
                z_filt = lo_filt

        # Apply filters
        if self.by_slice:
            voxel_grid = ndi.convolve1d(voxel_grid, weights=z_filt, axis=0, mode=mode)

        voxel_grid = ndi.convolve1d(voxel_grid, weights=y_filt, axis=1, mode=mode)
        voxel_grid = ndi.convolve1d(voxel_grid, weights=x_filt, axis=2, mode=mode)

        return voxel_grid
