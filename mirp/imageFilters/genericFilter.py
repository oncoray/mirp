from typing import List, Union

from mirp.imageProcess import calculate_features
from mirp.settings.settingsClass import SettingsClass
from mirp.imageClass import ImageClass
from mirp.roiClass import RoiClass
from mirp.images.genericImage import GenericImage


class GenericFilter:

    def __init__(self, settings: SettingsClass, name: str):
        # In-slice (2D) or 3D filtering
        self.by_slice = settings.img_transform.by_slice

    def generate_object(self):
        raise NotImplementedError("_generate_object method should be defined in the subclasses")

    def apply_transformation(
            self,
            img_obj: ImageClass,
            roi_list: List[RoiClass],
            settings: SettingsClass,
            compute_features: bool = False,
            extract_images: bool = False,
            file_path: Union[None, str] = None
    ):

        feature_list = []
        response_map_list = []

        # Iterate over generated filter objects with unique settings.
        for filter_object in self.generate_object():

            # Create a response map.
            response_map = filter_object.transform_deprecated(img_obj=img_obj)

            # Compute features.
            if compute_features:
                feature_list += [
                    calculate_features(
                        img_obj=response_map,
                        roi_list=[roi_obj.copy() for roi_obj in roi_list],
                        settings=settings.img_transform.feature_settings,
                        append_str=response_map.spat_transform + "_")
                ]

            # Export the image.
            if extract_images:
                if file_path is None:
                    response_map_list += [response_map]
                else:
                    response_map.export(file_path=file_path)
                    del response_map

        return feature_list, response_map_list

    def transform(self, image: GenericImage):
        raise NotImplementedError("transform method should be defined in the subclasses.")

    def transform_deprecated(self, img_obj: ImageClass):
        raise NotImplementedError("transform method should be defined in the subclasses")
