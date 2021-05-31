import numpy as np

from mirp.imageProcess import calculate_features
from mirp.imageClass import ImageClass
from mirp.imageFilters.utilities import pool_voxel_grids, FilterSet2D


class GaborFilter:

    def __init__(self, settings):

        # Sigma parameter that determines filter width.
        self.sigma = settings.img_transform.gabor_sigma

        # Cut-off for filter size.
        self.sigma_cutoff = settings.img_transform.gabor_sigma_truncate

        # Eccentricity parameter
        self.gamma = settings.img_transform.gabor_gamma

        # Wavelength parameter
        self.lambda_parameter = settings.img_transform.gabor_lambda

        # Initial angle.
        self.theta = settings.img_transform.gabor_theta

        # Update angle for rotational invariance.
        self.theta_step = settings.img_transform.gabor_theta_step

        # Update ype of response
        self.response_type = settings.img_transform.gabor_response

        # Boundary conditions.
        self.mode = settings.img_transform.boundary_condition

        # Rotational invariance.
        self.rot_invariance = settings.img_transform.gabor_rot_invar

        # Which pooling method is used.
        self.pooling_method = settings.img_transform.gabor_pooling_method

        # In-slice (2D) or 3D filtering
        self.by_slice = settings.general.by_slice

    def apply_transformation(self, img_obj: ImageClass, roi_list, settings, compute_features=False, extract_images=False, file_path=None):
        """Run feature extraction for transformed data"""

        feat_list = []

        # Generate transformed image
        img_trans_obj = self.transform(img_obj=img_obj)

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

    def transform(self, img_obj):
        """
        Transform image by calculating the laplacian of the gaussian second derivatives
        :param img_obj: image object
        :return:
        """

        # Copy base image
        img_gabor_obj = img_obj.copy(drop_image=True)

        # Prepare the string for the spatial transformation.
        spat_transform = ["gabor",
                          "s", str(self.sigma),
                          "g", str(self.gamma),
                          "l", str(self.lambda_parameter),
                          "t", str(self.theta)]

        if self.theta_step != 0.0:
            spat_transform += ["tstep", str(self.theta_step)]

        spat_transform += ["2D" if self.by_slice else "3D"]

        if self.rot_invariance and not self.by_slice:
            spat_transform += ["invar"]

        # Set the name of the transformation.
        img_gabor_obj.set_spatial_transform("_".join(spat_transform))

        # Convert sigma and lambda to voxel coordinates.
        if self.by_slice or not self.rot_invariance:
            sigma = np.max(np.divide(np.full(shape=(2), fill_value=self.sigma), img_obj.spacing[[1, 2]]))
            lamda = np.max(np.divide(np.full(shape=(2), fill_value=self.lambda_parameter), img_obj.spacing[[1, 2]]))
        else:
            sigma = np.max(np.divide(np.full(shape=(3), fill_value=self.sigma), img_obj.spacing))
            lamda = np.max(np.divide(np.full(shape=(3), fill_value=self.lambda_parameter), img_obj.spacing))
        # gamma_trunc = self.gamma if self.gamma > 1.0 else 1.0

        # Determine filter size.
        # filter_size = int(1 + 2 * np.floor(self.sigma_cutoff * sigma * gamma_trunc + 0.5))

        # Determine theta including steps.
        if self.theta_step > 0.0:
            theta = self.theta + np.arange(start=0.0, stop=2.0, step=self.theta_step)
            theta = theta.tolist()
        else:
            theta = [self.theta]

        # Determine the stacking axis. For 2D, axial planes are stacked, whereas in 3D, all orthogonal planes are used.
        if self.by_slice or not self.rot_invariance:
            stack_axis = [0]
        else:
            stack_axis = [0, 1, 2]

        # Create empty voxel grid
        img_voxel_grid = np.zeros(img_obj.size, dtype=np.float32)

        for jj, current_axis in enumerate(stack_axis):
            for ii, current_theta in enumerate(theta):

                # Create filter and compute response map.
                img_gabor_grid = self.transform_grid(voxel_grid=img_obj.get_voxel_grid(),
                                                     sigma=sigma,
                                                     gamma=self.gamma,
                                                     lamda=lamda,
                                                     theta=current_theta * np.pi,
                                                     # filter_size=filter_size,
                                                     stack_axis=current_axis)

                # Perform pooling
                if ii == jj == 0:
                    # Initially, update img_voxel_grid.
                    img_voxel_grid = img_gabor_grid
                else:
                    # Pool grids.
                    img_voxel_grid = pool_voxel_grids(x1=img_voxel_grid, x2=img_gabor_grid,
                                                      pooling_method=self.pooling_method)

                # Remove img_laws_grid to explicitly release memory when collecting garbage.
                del img_gabor_grid

        if self.pooling_method == "mean":
            # Perform final pooling step for mean pooling.
            img_voxel_grid = np.divide(img_voxel_grid, len(stack_axis) * len(theta))

        # Store the voxel grid in the ImageObject.
        img_gabor_obj.set_voxel_grid(voxel_grid=img_voxel_grid)

        return img_gabor_obj

    def transform_grid(self,
                       voxel_grid: np.ndarray,
                       sigma: np.float,
                       gamma: np.float,
                       lamda: np.float,
                       theta: np.float,
                       # filter_size,
                       stack_axis):

        # Determine size for x (alpha) and y (beta), prior to rotation.
        alpha = self.sigma_cutoff * sigma
        beta = self.sigma_cutoff * sigma * gamma

        # Determine filter size.
        x_size = max(np.abs(alpha * np.cos(theta) + beta * np.sin(theta)),
                     np.abs(-alpha * np.cos(theta) + beta * np.sin(theta)),
                     1)
        y_size = max(np.abs(alpha * np.sin(theta) - beta * np.cos(theta)),
                     np.abs(-alpha * np.sin(theta) - beta * np.cos(theta)),
                     1)

        x_size = int(1 + 2 * np.floor(x_size + 0.5))
        y_size = int(1 + 2 * np.floor(y_size + 0.5))

        # Create grid coordinates with [0, 0] in the center.
        y, x = np.mgrid[:y_size, :x_size].astype(np.float)
        y -= (y_size - 1.0) / 2.0
        x -= (x_size - 1.0) / 2.0

        # Compute rotation matrix: Since we are computing clock-wise rotations, use negative angles.
        rotation_matrix = np.array([[-np.cos(theta), np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

        # Compute rotated grid coordinates around the center.
        rotated_scan_coordinates = np.dot(rotation_matrix, np.array((y.flatten(), x.flatten())))
        y = rotated_scan_coordinates[0, :].reshape((y_size, x_size))
        x = rotated_scan_coordinates[1, :].reshape((y_size, x_size))

        # Create filter weights.
        gabor_filter = np.exp(-(np.power(x, 2.0) + gamma ** 2.0 * np.power(y, 2.0)) / (2.0 * sigma ** 2.0) + 1.0j * (
            2.0 * np.pi * y) / lamda)

        # Create filter
        gabor_filter = FilterSet2D(gabor_filter)

        # Convolve gabor filter with the image.
        response_map = gabor_filter.convolve(voxel_grid=voxel_grid,
                                             mode=self.mode,
                                             response=self.response_type,
                                             axis=stack_axis)

        # Compute the convolution
        return response_map
