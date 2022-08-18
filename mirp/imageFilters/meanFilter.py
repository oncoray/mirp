import copy
import numpy as np

from typing import List, Union
from mirp.imageClass import ImageClass
from mirp.imageProcess import calculate_features
from mirp.imageFilters.utilities import SeparableFilterSet
from mirp.importSettings import SettingsClass
from mirp.roiClass import RoiClass


class MeanFilter:

    def __init__(self, settings: SettingsClass, name: str):
        # Set the filter size
        self.filter_size = settings.img_transform.mean_filter_size

        # Set the filter mode
        self.mode = settings.img_transform.mean_filter_boundary_condition

        # In-slice (2D) or 3D filtering
        self.by_slice = settings.img_transform.by_slice

    def _generate_object(self):
        # Generator for transformation objects.
        filter_size = copy.deepcopy(self.filter_size)
        if not isinstance(filter_size, list):
            filter_size = [filter_size]

        # Iterate over options to yield filter objects with specific settings. A copy of the parent object is made to
        # avoid updating by reference.
        for current_filter_size in filter_size:
            filter_object = copy.deepcopy(self)
            filter_object.filter_size = current_filter_size

            yield filter_object

    def apply_transformation(self,
                             img_obj: ImageClass,
                             roi_list: List[RoiClass],
                             settings: SettingsClass,
                             compute_features: bool = False,
                             extract_images: bool = False,
                             file_path: Union[None, str] = None):
        """Run feature extraction for transformed data"""

        feature_list = []

        # Iterate over generated filter objects with unique settings.
        for filter_object in self._generate_object():

            # Create a response map.
            response_map = filter_object.transform(img_obj=img_obj)

            # Export the image.
            if extract_images:
                response_map.export(file_path=file_path)

            # Compute features.
            if compute_features:
                feature_list += [calculate_features(img_obj=response_map,
                                                    roi_list=[roi_obj.copy() for roi_obj in roi_list],
                                                    settings=settings.img_transform.feature_settings,
                                                    append_str=response_map.spat_transform + "_")]

            del response_map

        return feature_list

    def transform(self, img_obj: ImageClass):
        """
        Transform image by calculating the mean
        :param img_obj: image object
        :return:
        """
        # Copy base image
        response_map = img_obj.copy(drop_image=True)

        # Prepare the string for the spatial transformation.
        spatial_transform_string = ["mean"]
        spatial_transform_string += ["d", str(self.filter_size)]

        # Set the name of the transformation.
        response_map.set_spatial_transform("_".join(spatial_transform_string))

        # Skip transform in case the input image is missing
        if img_obj.is_missing:
            return response_map

        # Set up the filter kernel.
        filter_kernel = np.ones(self.filter_size, dtype=np.float) / self.filter_size

        # Create a filter set.
        if self.by_slice:
            filter_set = SeparableFilterSet(filter_x=filter_kernel,
                                            filter_y=filter_kernel)
        else:
            filter_set = SeparableFilterSet(filter_x=filter_kernel,
                                            filter_y=filter_kernel,
                                            filter_z=filter_kernel)

        # Apply the filter.
        response_map.set_voxel_grid(voxel_grid=filter_set.convolve(
            voxel_grid=img_obj.get_voxel_grid(),
            mode=self.mode))

        return response_map
