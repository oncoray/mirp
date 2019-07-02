import numpy as np

from mirp.imageProcess import calculate_features


class LaplacianOfGaussianFilter:

    def __init__(self, settings):
        self.sigma = settings.img_transform.log_sigma
        self.img_average = settings.img_transform.log_average
        self.sigma_cutoff = settings.img_transform.log_sigma_truncate
        self.mode = settings.img_transform.boundary_condition

        # In-slice (2D) or 3D filtering
        self.by_slice = settings.general.by_slice

    def apply_transformation(self, img_obj, roi_list, settings, compute_features=False, extract_images=False, file_path=None):
        """Run feature extraction for transformed data"""

        feat_list = []

        # Transform sigma to voxel distance
        if self.img_average:
            # Generate average image
            img_trans_obj = self.average_image(img_obj=img_obj)

            # Export image
            if extract_images:
                img_trans_obj.export(file_path=file_path)

            # Compute features
            if compute_features:
                feat_list += [calculate_features(img_obj=img_trans_obj, roi_list=roi_list, settings=settings,
                                                 append_str=img_trans_obj.spat_transform + "_")]

            # Clean up
            del img_trans_obj

        else:
            for curr_sigma in self.sigma:
                # Generate transformed image
                img_trans_obj = self.transform(img_obj=img_obj, sigma=curr_sigma)

                # Export image
                if extract_images:
                    img_trans_obj.export(file_path=file_path)

                # Compute features
                if compute_features:
                    feat_list += [calculate_features(img_obj=img_trans_obj, roi_list=roi_list, settings=settings,
                                                     append_str=img_trans_obj.spat_transform + "_")]

                # Clean up
                del img_trans_obj

        return feat_list

    def average_image(self, img_obj):
        """
        Creates an average image of multiple Laplacian-of-Gaussian images.
        :param img_obj: image object
        :return:
        """

        # Copy original
        img_log_obj = img_obj.copy(drop_image=True)

        # Set spatial transformation string for average laplacian of gaussian image
        img_log_obj.spat_transform = "log"

        # Skip transformation if input image is missing
        if img_obj.is_missing:
            return img_log_obj

        # Copy base image and empty all voxel data
        img_voxel_grid = np.zeros((img_obj.size), dtype=np.float32)

        # Add voxel intensities from laplacian of gaussian transforms
        for curr_sigma in self.sigma:
            img_voxel_grid += self.transform(img_obj=img_obj, sigma=curr_sigma).get_voxel_grid() / (len(self.sigma) * 1.0)

        # Set voxel grid
        img_log_obj.set_voxel_grid(voxel_grid=img_voxel_grid)

        return img_log_obj

    def transform(self, img_obj, sigma):
        """
        Transform image by calculating the laplacian of the gaussian second derivatives
        :param img_obj: image object
        :param sigma: sigma (in image dimensions, e.g. mm)
        :sigma_cut_off: number of standard deviations for cut-off of the gaussian filter
        :return:
        """

        import scipy.ndimage as ndi

        # Copy base image
        img_log_obj = img_obj.copy(drop_image=True)

        if sigma == 0.0:
            # Set spatial transformation string for transformed object
            img_log_obj.spat_transform = "lapl"

            # Skip transform in case the input image is missing
            if img_obj.is_missing:
                return img_log_obj

            # If sigma equals 0.0, perform only a laplacian transformation
            img_log_obj.set_voxel_grid(voxel_grid=ndi.laplace(img_obj.get_voxel_grid(), mode=self.mode))

        elif sigma > 0.0:
            # Set spatial transformation string for transformed object
            img_log_obj.spat_transform = "log_s" + str(np.max(sigma))

            # Skip transform in case the input image is missing
            if img_obj.is_missing:
                return img_log_obj

            # Calculate sigma for current image
            vox_sigma = np.divide(np.full(shape=(3), fill_value=sigma), img_obj.spacing)

            # Apply filters
            img_log_obj.set_voxel_grid(voxel_grid=self.transform_grid(voxel_grid=img_obj.get_voxel_grid(), sigma=vox_sigma, mode=self.mode, truncate=self.sigma_cutoff))

        else:
            raise ValueError("Laplacian of Gaussian transformation with negative sigma values are not allowed.")

        return img_log_obj

    def transform_grid(self, voxel_grid, sigma, mode, truncate, transform_method="scipy"):

        import scipy.ndimage as ndi

        if transform_method == "scipy":

            if self.by_slice:
                sigma[0] = 0.0

            return ndi.gaussian_laplace(voxel_grid, sigma=sigma, mode=mode, truncate=truncate)

        else:
            # Determine the size of the filter
            filter_size = np.round(2.0 * np.multiply(sigma, truncate))

            z, y, x = np.mgrid[:filter_size[0], :filter_size[1], :filter_size[2]]
            z -= (filter_size[0] - 1.0) / 2.0
            y -= (filter_size[1] - 1.0) / 2.0
            x -= (filter_size[2] - 1.0) / 2.0

            g_2 = np.power(z, 2.0) + np.power(y, 2.0) + np.power(x, 2.0)

            sigma = np.max(sigma)
            log_pre_factor = (g_2 - 3.0 * sigma**2.0) / (sigma**4.0 * np.sqrt(2.0 * np.pi * sigma**2.0)**3.0)
            log_gaussian_factor = np.exp(-g_2 / (2.0 * sigma**2.0))

            filter_weights = np.multiply(log_pre_factor, log_gaussian_factor)

            return ndi.convolve(voxel_grid, weights=filter_weights, mode=mode)
