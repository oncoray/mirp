import numpy as np

from mirp.imageProcess import calculate_features


class LawsFilter:
    def __init__(self, settings):

        # Whether response maps or texture energy images should be made rotationally invariant
        self.rot_invariance = settings.img_transform.laws_rot_invar

        # Size of neighbourhood in chebyshev distance from center voxel
        self.delta = settings.img_transform.laws_delta

        # 2D or 3D filter
        self.by_slice = settings.general.by_slice

        # Whether Laws texture energy should be calculated
        self.calculate_energy = settings.img_transform.laws_calculate_energy

        # Generate filter list
        self.filter_list = self.filter_order(user_combination=settings.img_transform.laws_kernel)

        # Normalise filters?
        self.kernel_normalise = True

    def apply_transformation(self, img_obj, roi_list, settings, compute_features=False, extract_images=False, file_path=None):

        feat_list = []

        # Iterate over wavelet filters
        for current_filter_set in self.filter_list:

            # Copy roi list
            roi_trans_list = [roi_obj.copy() for roi_obj in roi_list]

            # Add spatially transformed image object. In case of rotational invariance, this is averaged.
            img_trans_obj = self.transform(img_obj=img_obj, filter_set=current_filter_set, mode=settings.img_transform.boundary_condition)

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

    def transform(self, img_obj, filter_set, mode):

        # Copy base image
        img_laws_obj = img_obj.copy(drop_image=True)

        # Set spatial transformation filter string
        img_laws_obj.spat_transform = "laws_" + "".join(filter_set[0])
        if self.calculate_energy:
            img_laws_obj.spat_transform += "_energy"
        if self.rot_invariance:
            img_laws_obj.spat_transform += "_invar"

        # Skip transformation in case the input image is missing
        if img_obj.is_missing:
            return img_laws_obj

        # Create empty voxel grid
        img_voxel_grid = np.zeros(img_obj.size, dtype=np.float32)

        for current_filter in filter_set:
            # Calculate response map for the given combination of kernels
            img_laws_grid = self.transform_grid(img_voxel_grid=img_obj.get_voxel_grid(), filter_order=current_filter, mode=mode)

            if self.calculate_energy:
                img_laws_grid = self.response_to_energy(voxel_grid=img_laws_grid, mode=mode)

            img_voxel_grid += img_laws_grid / (len(filter_set) * 1.0)

            del img_laws_grid

        # Update voxel grid
        img_laws_obj.set_voxel_grid(voxel_grid=img_voxel_grid)

        return img_laws_obj

    def transform_grid(self, img_voxel_grid, filter_order, mode):
        import scipy.ndimage as ndi

        for ii in np.arange(len(filter_order)):

            # Define kernel based on input filter names. Note that these are ordered as "XYZ" whereas numpy assumes "ZYX".
            # Hence we read the filter names in the reverse order.
            laws_kernel_name = filter_order[-ii-1]
            if laws_kernel_name == "L5":
                laws_kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
            elif laws_kernel_name == "E5":
                laws_kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])
            elif laws_kernel_name == "S5":
                laws_kernel = np.array([-1.0, 0.0, 2.0, 0.0, -1.0])
            elif laws_kernel_name == "W5":
                laws_kernel = np.array([-1.0, 2.0, 0.0, -2.0, 1.0])
            elif laws_kernel_name == "R5":
                laws_kernel = np.array([1.0, -4.0, 6.0, -4.0, 1.0])
            else:
                raise ValueError("%s is not an implemented Laws kernel")

            # Normalise kernel
            if self.kernel_normalise:
                laws_kernel /= np.sum(np.abs(laws_kernel))

            # Apply filters
            if self.by_slice:
                img_voxel_grid = ndi.convolve1d(img_voxel_grid, weights=laws_kernel, axis=ii + 1, mode=mode)
            else:
                img_voxel_grid = ndi.convolve1d(img_voxel_grid, weights=laws_kernel, axis=ii, mode=mode)

        return img_voxel_grid

    def response_to_energy(self, voxel_grid, mode):
        import scipy.ndimage as ndi

        # Define the local neighbourhood for summation
        sum_kernel = np.ones(2*self.delta+1, dtype=np.float32)

        # Take absolute value of the voxel grid
        voxel_grid = np.abs(voxel_grid)

        if self.kernel_normalise:
            sum_kernel /= np.sum(sum_kernel)

        # Perform summation across slices
        if not self.by_slice:
            voxel_grid = ndi.convolve1d(voxel_grid, weights=sum_kernel, axis=0, mode=mode)
        # TODO: Update results so that energy outside volume is not taken into account: i.e. better

        # Sum in slice
        voxel_grid = ndi.convolve1d(voxel_grid, weights=sum_kernel, axis=1, mode=mode)
        voxel_grid = ndi.convolve1d(voxel_grid, weights=sum_kernel, axis=2, mode=mode)

        return voxel_grid

    def filter_order(self, user_combination=None):
        import itertools

        if self.by_slice:
            q_rep = 2
        else:
            q_rep = 3

        # Check if there is user-provided combination
        if user_combination is None or user_combination == ["all"]:
            # Generate combinations of kernels. If rotational invariance is required, only the base combinations are generated.
            if self.rot_invariance:
                comb_list = list(itertools.combinations_with_replacement(["L5", "E5", "R5", "S5", "W5"], q_rep))
            else:
                comb_list = list(itertools.product(["L5", "E5", "R5", "S5", "W5"], repeat=q_rep))

            # Convert tuples from itertools to list
            comb_list = [[list(curr_comb)] for curr_comb in comb_list]
        else:
            # Split list of input strings and parse to combinations list
            # comb_list = [[[user_combination[ii:ii+2] for ii in range(0, len(user_combination), 2)]]]
            comb_list = [[[curr_combination[ii:ii + 2] for ii in range(0, len(curr_combination), 2)]] for curr_combination in user_combination]

        # If rotational invariance is required, all unique permutations for each base combinations are added to the list
        if self.rot_invariance:
            comb_list = [[list(curr_comb_1) for curr_comb_1 in list(set(list(itertools.permutations(curr_comb[0], q_rep))))] for curr_comb in comb_list]

        # Set filter_list
        return comb_list
