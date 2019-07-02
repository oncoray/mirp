import numpy as np

from mirp.imageProcess import calculate_features


class MeanFilter:

    def __init__(self, settings):
        self.filter_size = settings.img_transform.mean_filter_size
        self.mode = settings.img_transform.boundary_condition

        # In-slice (2D) or 3D filtering
        self.by_slice = settings.general.by_slice

    def apply_transformation(self, img_obj, roi_list, settings, compute_features=False, extract_images=False, file_path=None):
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
        Transform image by calculating the mean
        :param img_obj: image object
        :return:
        """

        import scipy.ndimage as ndi

        # Copy base image
        img_trans_obj = img_obj.copy(drop_image=True)

        # Set spatial transformation string for transformed object
        img_trans_obj.spat_transform = "mean"

        # Skip transform in case the input image is missing
        if img_obj.is_missing:
            return img_trans_obj

        # If sigma equals 0.0, perform only a laplacian transformation
        img_trans_obj.set_voxel_grid(voxel_grid=ndi.uniform_filter(input=img_obj.get_voxel_grid, size=self.filter_size, mode=self.mode))

        return img_trans_obj
