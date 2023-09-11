from typing import Union, List, Dict, Optional, Generator, Iterable
import copy

from mirp.importData.imageGenericFile import ImageFile
from mirp.settings.settingsGeneric import SettingsClass
from mirp.workflows.standardWorkflow import StandardWorkflow


def extract_features(
        write_features: bool = True,
        export_features: bool = False,
        **kwargs
):
    return extract_features_and_images(
        write_features=write_features,
        export_features=export_features,
        write_images=False,
        export_images=False,
        **kwargs
    )


def extract_features_generator(
        write_features: bool = True,
        export_features: bool = False,
        **kwargs
):
    yield from extract_features_and_images(
        write_features=write_features,
        export_features=export_features,
        write_images=False,
        export_images=False,
        **kwargs
    )


def extract_images(
        write_images: bool = True,
        export_images: bool = False,
        **kwargs
):
    return extract_features_and_images(
        write_features=False,
        export_features=False,
        write_images=write_images,
        export_images=export_images,
        **kwargs
    )


def extract_images_generator(
        write_images: bool = True,
        export_images: bool = False,
        **kwargs
):
    yield from extract_features_and_images_generator(
        write_features=False,
        export_features=False,
        write_images=write_images,
        export_images=export_images,
        **kwargs
    )


def extract_features_and_images(
        image_export_format: str = "dict",
        **kwargs
):
    workflows = list(_base_extract_features_and_images(**kwargs))
    return [workflow.standard_extraction(image_export_format=image_export_format) for workflow in workflows]


def extract_features_and_images_generator(
        image_export_format: str = "dict",
        **kwargs
):
    workflows = list(_base_extract_features_and_images(**kwargs))
    for workflow in workflows:
        yield workflow.standard_extraction(image_export_format=image_export_format)


def _base_extract_features_and_images(
        image,
        mask=None,
        sample_name: Union[None, str, List[str]] = None,
        image_name: Union[None, str, List[str]] = None,
        image_file_type: Union[None, str] = None,
        image_modality: Union[None, str, List[str]] = None,
        image_sub_folder: Union[None, str] = None,
        mask_name: Union[None, str, List[str]] = None,
        mask_file_type: Union[None, str] = None,
        mask_modality: Union[None, str, List[str]] = None,
        mask_sub_folder: Union[None, str] = None,
        roi_name: Union[None, str, List[str], Dict[str, str]] = None,
        association_strategy: Union[None, str, List[str]] = None,
        settings: Union[None, str, SettingsClass, List[SettingsClass]] = None,
        stack_masks: str = "auto",
        stack_images: str = "auto",
        write_features: bool = False,
        export_features: bool = False,
        write_images: bool = False,
        export_images: bool = False,
        write_dir: Optional[str] = None,
        **kwargs
):
    from mirp.importData.importImageAndMask import import_image_and_mask
    from mirp.settings.importConfigurationSettings import import_configuration_settings

    # Import settings (to provide immediate feedback if something is amiss).
    if isinstance(settings, str):
        settings = import_configuration_settings(
            compute_features=write_features or export_features,
            path=settings
        )
    elif isinstance(settings, SettingsClass):
        settings = [settings]
    elif isinstance(settings, Iterable) and all(isinstance(x, SettingsClass) for x in settings):
        settings = list(settings)
    elif settings is None:
        settings = import_configuration_settings(
            compute_features=write_features or export_features,
            **kwargs
        )
    else:
        raise TypeError(f"The 'settings' argument is expected to be a path to a configuration xml file, "
                        f"a SettingsClass object, or a list thereof. Found: {type(settings)}.")

    if not write_images and not write_features:
        write_dir = None

    if write_images and write_dir is None:
        raise ValueError("write_dir argument should be provided for writing images and masks to.")
    if write_features and write_dir is None:
        raise ValueError("write_dir argument should be provided for writing feature tables to.")

    image_list = import_image_and_mask(
        image=image,
        mask=mask,
        sample_name=sample_name,
        image_name=image_name,
        image_file_type=image_file_type,
        image_modality=image_modality,
        image_sub_folder=image_sub_folder,
        mask_name=mask_name,
        mask_file_type=mask_file_type,
        mask_modality=mask_modality,
        mask_sub_folder=mask_sub_folder,
        roi_name=roi_name,
        association_strategy=association_strategy,
        stack_images=stack_images,
        stack_masks=stack_masks
    )

    yield from _generate_feature_and_image_extraction_workflows(
        image_list=image_list,
        settings=settings,
        write_dir=write_dir,
        write_features=write_features,
        export_features=export_features,
        write_images=write_images,
        export_images=export_images
    )


def _generate_feature_and_image_extraction_workflows(
        image_list: List[ImageFile],
        settings: List[SettingsClass],
        write_dir: Optional[str],
        write_features: bool,
        export_features: bool,
        write_images: bool,
        export_images: bool
) -> Generator[StandardWorkflow, None, None]:

    for image_file in image_list:
        for current_settings in settings:

            if not current_settings.feature_extr.has_any_feature_family() and (
                    current_settings.img_transform.spatial_filters is not None and not
                    current_settings.img_transform.feature_settings.has_any_feature_family()
            ) and (export_features or write_features):
                raise ValueError(
                    "No feature families were specified. Please set 'base_feature_families' or"
                    " 'response_map_feature_families'."
                )

            if current_settings.perturbation.noise_repetitions is None or \
                    current_settings.perturbation.noise_repetitions == 0:
                noise_repetition_ids = [None]
            else:
                noise_repetition_ids = list(range(current_settings.perturbation.noise_repetitions))

            if current_settings.perturbation.rotation_angles is None or len(
                    current_settings.perturbation.rotation_angles) == 0 or all(
                x == 0.0 for x in current_settings.perturbation.rotation_angles
            ):
                rotation_angles = [None]
            else:
                rotation_angles = copy.deepcopy(current_settings.perturbation.rotation_angles)

            if current_settings.perturbation.translation_fraction is None or len(
                current_settings.perturbation.translation_fraction) == 0 or all(
                x == 0.0 for x in current_settings.perturbation.translation_fraction
            ):
                translations = [None]
            else:
                config_translation = copy.deepcopy(current_settings.perturbation.translation_fraction)
                translations = []
                for translation_x in config_translation:
                    for translation_y in config_translation:
                        if not current_settings.general.by_slice:
                            for translation_z in config_translation:
                                translations += [(translation_z, translation_y, translation_x)]
                        else:
                            translations += [(0.0, translation_y, translation_x)]

            if current_settings.img_interpolate.new_spacing is None or len(
                    current_settings.img_interpolate.new_spacing) == 0 or all(
                x == 0.0 for x in current_settings.img_interpolate.new_spacing
            ):
                spacings = [None]
            else:
                spacings = copy.deepcopy(current_settings.img_interpolate.new_spacing)

            for noise_repetition_id in noise_repetition_ids:
                for rotation_angle in rotation_angles:
                    for translation in translations:
                        for spacing in spacings:
                            yield StandardWorkflow(
                                image_file=copy.deepcopy(image_file),
                                write_dir=write_dir,
                                settings=current_settings,
                                settings_name=current_settings.general.config_str,
                                write_features=write_features,
                                export_features=export_features,
                                write_images=write_images,
                                export_images=export_images,
                                noise_iteration_id=noise_repetition_id,
                                rotation=rotation_angle,
                                translation=translation,
                                new_image_spacing=spacing
                            )
