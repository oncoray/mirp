from typing import Optional

from mirp.settings.settingsFeatureExtraction import FeatureExtractionSettingsClass
from mirp.settings.settingsGeneral import GeneralSettingsClass
from mirp.settings.settingsImageProcessing import ImagePostProcessingClass
from mirp.settings.settingsImageTransformation import ImageTransformationSettingsClass
from mirp.settings.settingsInterpolation import ImageInterpolationSettingsClass, MaskInterpolationSettingsClass
from mirp.settings.settingsMaskResegmentation import ResegmentationSettingsClass
from mirp.settings.settingsPerturbation import ImagePerturbationSettingsClass


class SettingsClass:

    def __init__(
            self,
            general_settings: Optional[GeneralSettingsClass] = None,
            img_interpolate_settings: Optional[ImageInterpolationSettingsClass] = None,
            roi_interpolate_settings: Optional[MaskInterpolationSettingsClass] = None,
            post_process_settings: Optional[ImagePostProcessingClass] = None,
            perturbation_settings: Optional[ImagePerturbationSettingsClass] = None,
            roi_resegment_settings: Optional[ResegmentationSettingsClass] = None,
            feature_extr_settings: Optional[FeatureExtractionSettingsClass] = None,
            img_transform_settings: Optional[ImageTransformationSettingsClass] = None,
            **kwargs
    ):
        # General settings.
        if general_settings is None:
            general_settings = GeneralSettingsClass(**kwargs)
        self.general = general_settings

        # Image interpolation settings.
        if img_interpolate_settings is None:
            img_interpolate_settings = ImageInterpolationSettingsClass(
                by_slice=general_settings.by_slice,
                **kwargs
            )
        self.img_interpolate = img_interpolate_settings

        # Mask interpolation settings.
        if roi_interpolate_settings is None:
            roi_interpolate_settings = MaskInterpolationSettingsClass(**kwargs)
        self.roi_interpolate = roi_interpolate_settings

        # Image (post-)processing settings.
        if post_process_settings is None:
            post_process_settings = ImagePostProcessingClass(**kwargs)
        self.post_process = post_process_settings

        # Image perturbation settings.
        if perturbation_settings is None:
            perturbation_settings = ImagePerturbationSettingsClass(**kwargs)
        self.perturbation = perturbation_settings

        # Mask resegmentation settings.
        if roi_resegment_settings is None:
            roi_resegment_settings = ResegmentationSettingsClass(**kwargs)
        self.roi_resegment = roi_resegment_settings

        # Feature extraction settings.
        if feature_extr_settings is None:
            feature_extr_settings = FeatureExtractionSettingsClass(
                by_slice=general_settings.by_slice,
                no_approximation=general_settings.no_approximation,
                **kwargs
            )
        self.feature_extr = feature_extr_settings

        # Image transformation settings
        if img_transform_settings is None:
            img_transform_settings = ImageTransformationSettingsClass(
                by_slice=general_settings.by_slice,
                response_map_feature_settings=FeatureExtractionSettingsClass(
                    by_slice=general_settings.by_slice,
                    no_approximation=general_settings.no_approximation,
                    **kwargs
                )
            )
        self.img_transform = img_transform_settings
