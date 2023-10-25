from typing import Generator, Iterable, Any
import copy
import ray

from mirp.importData.imageGenericFile import ImageFile
from mirp.settings.settingsGeneric import SettingsClass
from mirp.workflows.standardWorkflow import StandardWorkflow


def deep_learning_preprocessing(
        output_slices: bool = False,
        crop_size: None | list[float] | list[int] = None,
        image_export_format: str = "numpy",
        write_file_format: str = "numpy",
        export_images: None | bool = None,
        write_images: None | bool = None,
        write_dir: None | str = None,
        num_cpus: None | int = None,
        **kwargs
) -> None | list[Any]:
    """
    Pre-processes images for deep learning.

    Parameters
    ----------
    output_slices: bool, optional, default: False
        Determines whether separate slices should be extracted.

    crop_size: list of float or list of int, optional, default: None
        Size to which the images and masks should be cropped. Images and masks are cropped around the center of the
        mask(s).

        .. note::
            MIRP follows the numpy convention for indexing (*z*, *y*, *x*). The final element always corresponds to the
            *x* dimension.

    image_export_format: {"dict", "native", "numpy"}, default: "numpy"
        Return format for processed images and masks. ``"dict"`` returns dictionaries of images and masks as numpy
        arrays and associated characteristics. ``"native"`` returns images and masks in their internal format.
        ``"numpy"`` returns images and masks in numpy format. This argument is only used if ``export_images=True``.

    write_file_format: {"nifti", "numpy"}, default: "numpy"
        File format for processed images and masks. ``"nifti"`` writes images and masks in the NIfTI file format,
        and ``"numpy"`` writes images and masks as numpy files. This argument is only used if ``write_images=True``.

    export_images: bool, optional
        Determines whether processed images and masks should be returned by the function.

    write_images: bool, optional
        Determines whether processed images and masks should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    write_dir: str, optional
        Path to directory where processed images and masks should be written. If not set, processed images and masks
        are returned by this function. Required if ``write_images=True``.

    num_cpus: int, optional, default: None
        Number of CPU nodes that should be used for parallel processing. Image and mask processing can be
        parallelized using the ``ray`` package. If a ray cluster is defined by the user, this cluster will be used
        instead. By default, image and mask processing are processed sequentially.

    **kwargs:
        Keyword arguments passed for importing images and masks (
        :func:`~mirp.importData.importImageAndMask.import_image_and_mask`) and configuring settings (notably
        :class:`~mirp.settings.settingsImageProcessing.ImagePostProcessingClass`,
        :class:`~mirp.settings.settingsPerturbation.ImagePerturbationSettingsClass`), among others.

    Returns
    -------
    None | list[Any]
        List of images and masks in the format indicated by ``image_export_format``, if ``export_images=True``.

    See Also
    --------
    Keyword arguments can be provided to configure the following:

    * image and mask import (:func:`~mirp.importData.importImageAndMask.import_image_and_mask`)
    * image post-processing (:class:`~mirp.settings.settingsImageProcessing.ImagePostProcessingClass`)
    * image perturbation / augmentation (:class:`~mirp.settings.settingPerturbation.ImagePerturbationSettingsClass`)
    * image interpolation / resampling (:class:`~mirp.settings.settingsInterpolation.ImageInterpolationSettingsClass` and
      :class:`~mirp.settings.settingsInterpolation.MaskInterpolationSettingsClass`)
    * mask resegmentation (:class:`~mirp.settings.settingsMaskResegmentation.ResegmentationSettingsClass`)

    """

    # Conditionally start a ray cluster.
    external_ray = ray.is_initialized()
    if not external_ray and num_cpus is not None and num_cpus > 1:
        ray.init(num_cpus=num_cpus)

    if ray.is_initialized():
        # Parallel processing.
        results = [
            _ray_extractor.remote(
                workflow=workflow,
                output_slices=output_slices,
                crop_size=crop_size,
                image_export_format=image_export_format,
                write_file_format=write_file_format
            )
            for workflow in _base_deep_learning_preprocessing(
                export_images=export_images,
                write_images=write_images,
                write_dir=write_dir,
                **kwargs
            )
        ]

        results = ray.get(results)
        if not external_ray:
            ray.shutdown()
    else:
        workflows = list(_base_deep_learning_preprocessing(
            export_images=export_images,
            write_images=write_images,
            write_dir=write_dir,
            **kwargs)
        )

        results = [
            workflow.deep_learning_conversion(
                output_slices=output_slices,
                crop_size=crop_size,
                image_export_format=image_export_format,
                write_file_format=write_file_format
            )
            for workflow in workflows
        ]

    return results


@ray.remote
def _ray_extractor(
        workflow: StandardWorkflow,
        output_slices: bool = False,
        crop_size: None | list[float] | list[int] = None,
        image_export_format: str = "numpy",
        write_file_format: str = "numpy"
):
    return workflow.deep_learning_conversion(
        output_slices=output_slices,
        crop_size=crop_size,
        image_export_format=image_export_format,
        write_file_format=write_file_format
    )


def deep_learning_preprocessing_generator(
        output_slices: bool = False,
        crop_size: None | list[float] | list[int] = None,
        image_export_format: str = "numpy",
        write_file_format: str = "numpy",
        export_images: None | bool = None,
        write_images: None | bool = None,
        write_dir: None | str = None,
        **kwargs
) -> Generator[Any, None, None]:
    """
    Generator for pre-processing images for deep learning.

    Parameters
    ----------
    output_slices: bool, optional, default: False
        Determines whether separate slices should be extracted.

    crop_size: list of float or list of int, optional, default: None
        Size to which the images and masks should be cropped. Images and masks are cropped around the center of the
        mask(s).

        .. note::
            MIRP follows the numpy convention for indexing (*z*, *y*, *x*). The final element always corresponds to the
            *x* dimension.

    image_export_format: {"dict", "native", "numpy"}, default: "numpy"
        Return format for processed images and masks. ``"dict"`` returns dictionaries of images and masks as numpy
        arrays and associated characteristics. ``"native"`` returns images and masks in their internal format.
        ``"numpy"`` returns images and masks in numpy format. This argument is only used if ``export_images=True``.

    write_file_format: {"nifti", "numpy"}, default: "numpy"
        File format for processed images and masks. ``"nifti"`` writes images and masks in the NIfTI file format,
        and ``"numpy"`` writes images and masks as numpy files. This argument is only used if ``write_images=True``.

    export_images: bool, optional
        Determines whether processed images and masks should be returned by the function.

    write_images: bool, optional
        Determines whether processed images and masks should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    write_dir: str, optional
        Path to directory where processed images and masks should be written. If not set, processed images and masks
        are returned by this function. Required if ``write_images=True``.

    **kwargs:
        Keyword arguments passed for importing images and masks (
        :func:`~mirp.importData.importImageAndMask.import_image_and_mask`) and configuring settings (notably
        :class:`~mirp.settings.settingsImageProcessing.ImagePostProcessingClass`,
        :class:`~mirp.settings.settingsPerturbation.ImagePerturbationSettingsClass`), among others.

    Yields
    -------
    None | list[Any]
        List of images and masks in the format indicated by ``image_export_format``, if ``export_images=True``.

    See Also
    --------
    Keyword arguments can be provided to configure the following:

    * image and mask import (:func:`~mirp.importData.importImageAndMask.import_image_and_mask`)
    * image post-processing (:class:`~mirp.settings.settingsImageProcessing.ImagePostProcessingClass`)
    * image perturbation / augmentation (:class:`~mirp.settings.settingPerturbation.ImagePerturbationSettingsClass`)
    * image interpolation / resampling (:class:`~mirp.settings.settingsInterpolation.ImageInterpolationSettingsClass` and
      :class:`~mirp.settings.settingsInterpolation.MaskInterpolationSettingsClass`)
    * mask resegmentation (:class:`~mirp.settings.settingsMaskResegmentation.ResegmentationSettingsClass`)

    """
    workflows = list(_base_deep_learning_preprocessing(
        export_images=export_images,
        write_images=write_images,
        write_dir=write_dir,
        **kwargs))

    for workflow in workflows:
        yield workflow.deep_learning_conversion(
            output_slices=output_slices,
            crop_size=crop_size,
            image_export_format=image_export_format,
            write_file_format=write_file_format
        )


def _base_deep_learning_preprocessing(
        image,
        mask=None,
        sample_name: None | str | list[str] = None,
        image_name: None | str | list[str] = None,
        image_file_type: None | str = None,
        image_modality: None | str | list[str] = None,
        image_sub_folder: None | str = None,
        mask_name: None | str | list[str] = None,
        mask_file_type: None | str = None,
        mask_modality: None | str | list[str] = None,
        mask_sub_folder: None | str = None,
        roi_name: None | str | list[str] | dict[str, str] = None,
        association_strategy: None | str | list[str] = None,
        settings: None | str | SettingsClass | list[SettingsClass] = None,
        stack_masks: str = "auto",
        stack_images: str = "auto",
        write_images: None | bool = None,
        export_images: None | bool = None,
        write_dir: None | str = None,
        **kwargs
):
    from mirp.importData.importImageAndMask import import_image_and_mask
    from mirp.settings.importConfigurationSettings import import_configuration_settings

    # Infer write_images, export_images based on write_dir.
    if write_images is None:
        write_images = write_dir is not None
    if export_images is None:
        export_images = write_dir is None

    if not write_images:
        write_dir = None

    if write_images and write_dir is None:
        raise ValueError("write_dir argument should be provided for writing images and masks to.")

    if not write_images and not export_images:
        raise ValueError(f"write_images and export_images arguments cannot both be False.")

    # Import settings (to provide immediate feedback if something is amiss).
    if isinstance(settings, str):
        settings = import_configuration_settings(
            compute_features=False,
            path=settings
        )
    elif isinstance(settings, SettingsClass):
        settings = [settings]
    elif isinstance(settings, Iterable) and all(isinstance(x, SettingsClass) for x in settings):
        settings = list(settings)
    elif settings is None:
        settings = import_configuration_settings(
            compute_features=False,
            **kwargs
        )
    else:
        raise TypeError(
            f"The 'settings' argument is expected to be a path to a configuration xml file, "
            f"a SettingsClass object, or a list thereof. Found: {type(settings)}."
        )

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

    yield from _generate_dl_preprocessing_workflows(
        image_list=image_list,
        settings=settings,
        write_dir=write_dir,
        write_images=write_images,
        export_images=export_images
    )


def _generate_dl_preprocessing_workflows(
        image_list: list[ImageFile],
        settings: list[SettingsClass],
        write_dir: None | str,
        write_images: bool,
        export_images: bool
) -> Generator[StandardWorkflow, None, None]:

    for image_file in image_list:
        for current_settings in settings:

            # Update settings to remove settings that may cause problems.
            current_settings.feature_extr.families = "none"
            current_settings.img_transform.feature_settings.families = "none"
            current_settings.perturbation.crop_around_roi = False
            current_settings.roi_resegment.resegmentation_method = "none"

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
                                write_features=False,
                                export_features=False,
                                write_images=write_images,
                                export_images=export_images,
                                noise_iteration_id=noise_repetition_id,
                                rotation=rotation_angle,
                                translation=translation,
                                new_image_spacing=spacing
                            )
