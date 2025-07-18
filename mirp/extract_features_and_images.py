from typing import Generator, Iterable, Any
import logging, copy, sys

from mirp._data_import.generic_file import ImageFile
from mirp.settings.generic import SettingsClass
from mirp.utilities.parallel import (
    start_parallel_cluster, parse_parallel_backend, message_parallel_process, cluster_exists, shutdown_cluster)
from mirp.utilities.parallel_ray import ray_remote, ray_init, ray_is_initialized, ray_get, ray_shutdown
from mirp._workflows.standardWorkflow import StandardWorkflow


def extract_features(
        write_features: None | bool = None,
        export_features: None | bool = None,
        write_dir: None | str = None,
        num_cpus: None | int = None,
        **kwargs
) -> None | list[Any]:
    """
    Compute features from regions of interest in images. This function is a wrapper around
    :func:`~mirp.extractFeaturesAndImages.extract_features_and_images`.

    Parameters
    ----------
    write_features: bool, optional
        Determines whether features computed from images should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    export_features: bool, optional
        Determines whether features computed from images should be returned by the function.

    write_dir: str, optional
        Path to directory where feature tables should be written. If not set, feature tables are returned by this
        function. Required if ``write_features=True``.

    num_cpus: int, optional, default: None
        Number of CPU nodes that should be used for parallel processing. Image processing and feature computation can be
        parallelized using the ``ray`` package. If a ray cluster is defined by the user, this cluster will be used
        instead. By default, images are processed sequentially.

    **kwargs:
        Keyword arguments passed for importing images and masks (
        :func:`mirp.data_import.import_image_and_mask.import_image_and_mask`) and configuring settings:

        * general settings (:class:`~mirp.settings.general_parameters.GeneralSettingsClass`)
        * image post-processing (:class:`~mirp.settings.image_processing_parameters.ImagePostProcessingClass`)
        * image perturbation / augmentation (:class:`~mirp.settings.perturbation_parameters.ImagePerturbationSettingsClass`)
        * image interpolation / resampling (
          :class:`~mirp.settings.interpolation_parameters.ImageInterpolationSettingsClass` and
          :class:`~mirp.settings.interpolation_parameters.MaskInterpolationSettingsClass`)
        * mask resegmentation (:class:`~mirp.settings.resegmentation_parameters.ResegmentationSettingsClass`)
        * image transformation (:class:`~mirp.settings.transformation_parameters.ImageTransformationSettingsClass`)
        * feature computation / extraction (
          :class:`~mirp.settings.feature_parameters.FeatureExtractionSettingsClass`)

    Returns
    -------
    None | list[Any]
        List of feature tables, if ``export_features=True``.

    See Also
    --------
    :func:`~mirp.extractFeaturesAndImages.extract_features_and_images`

    """
    return extract_features_and_images(
        write_features=write_features,
        export_features=export_features,
        write_images=False,
        export_images=False,
        write_dir=write_dir,
        num_cpus=num_cpus,
        **kwargs
    )


def extract_features_generator(
        write_features: bool = False,
        export_features: bool = True,
        **kwargs
):
    """
    Compute features from regions of interest in images. This generator is a wrapper around
    :func:`~mirp.extractFeaturesAndImages.extract_features_and_images_generator`.

    Parameters
    ----------
    write_features: bool, default: False
        Determines whether features computed from images should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    export_features: bool, default: True
        Determines whether features computed from images should be returned by the function.

    **kwargs:
        Keyword arguments passed for importing images and masks (
        :func:`mirp.data_import.import_image_and_mask.import_image_and_mask`) and configuring settings:

        * general settings (:class:`~mirp.settings.general_parameters.GeneralSettingsClass`)
        * image post-processing (:class:`~mirp.settings.image_processing_parameters.ImagePostProcessingClass`)
        * image perturbation / augmentation (:class:`~mirp.settings.perturbation_parameters.ImagePerturbationSettingsClass`)
        * image interpolation / resampling (
          :class:`~mirp.settings.interpolation_parameters.ImageInterpolationSettingsClass` and
          :class:`~mirp.settings.interpolation_parameters.MaskInterpolationSettingsClass`)
        * mask resegmentation (:class:`~mirp.settings.resegmentation_parameters.ResegmentationSettingsClass`)
        * image transformation (:class:`~mirp.settings.transformation_parameters.ImageTransformationSettingsClass`)
        * feature computation / extraction (
          :class:`~mirp.settings.feature_parameters.FeatureExtractionSettingsClass`)

    Returns
    -------
    None | list[Any]
        List of feature tables, if ``export_features=True``.

    See Also
    --------
    :func:`~mirp.extractFeaturesAndImages.extract_features_and_images_generator`

    """
    yield from extract_features_and_images_generator(
        write_features=write_features,
        export_features=export_features,
        write_images=False,
        export_images=False,
        **kwargs
    )


def extract_images(
        write_images: None | bool = None,
        export_images: None | bool = None,
        write_dir: None | str = None,
        image_export_format: str = "dict",
        num_cpus: None | int = None,
        **kwargs
):
    """
    Process images and masks. This function is a wrapper around
    :func:`~mirp.extractFeaturesAndImages.extract_features_and_images`.

    Parameters
    ----------
    write_images: bool, optional
        Determines whether processed images and masks should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    export_images: bool, optional
        Determines whether processed images and masks should be returned by the function.

    write_dir: str, optional
        Path to directory where processed images and masks should be written. If not set, processed images and masks
        are returned by this function. Required if ``write_images=True``.

    image_export_format: {"dict", "native", "numpy"}, default: "dict"
        Return format for processed images and masks. ``"dict"`` returns dictionaries of images and masks as numpy
        arrays and associated characteristics. ``"native"`` returns images and masks in their internal format.
        ``"numpy"`` returns images and masks in numpy format. This argument is only used if ``export_images=True``.

    num_cpus: int, optional, default: None
        Number of CPU nodes that should be used for parallel processing. Image processing and feature computation can be
        parallelized using the ``ray`` package. If a ray cluster is defined by the user, this cluster will be used
        instead. By default, images are processed sequentially.

    **kwargs:
        Keyword arguments passed for importing images and masks (
        :func:`mirp.data_import.import_image_and_mask.import_image_and_mask`) and configuring settings:

        * general settings (:class:`~mirp.settings.general_parameters.GeneralSettingsClass`)
        * image post-processing (:class:`~mirp.settings.image_processing_parameters.ImagePostProcessingClass`)
        * image perturbation / augmentation (:class:`~mirp.settings.perturbation_parameters.ImagePerturbationSettingsClass`)
        * image interpolation / resampling (
          :class:`~mirp.settings.interpolation_parameters.ImageInterpolationSettingsClass` and
          :class:`~mirp.settings.interpolation_parameters.MaskInterpolationSettingsClass`)
        * mask resegmentation (:class:`~mirp.settings.resegmentation_parameters.ResegmentationSettingsClass`)
        * image transformation (:class:`~mirp.settings.transformation_parameters.ImageTransformationSettingsClass`)

    Returns
    -------
    None | list[Any]
        List of feature tables, if ``export_images=True``.

    See Also
    --------
    :func:`~mirp.extractFeaturesAndImages.extract_features_and_images`

    """
    return extract_features_and_images(
        write_features=False,
        export_features=False,
        write_images=write_images,
        export_images=export_images,
        write_dir=write_dir,
        image_export_format=image_export_format,
        num_cpus=num_cpus,
        **kwargs
    )


def extract_images_generator(
        write_images: bool = False,
        export_images: bool = True,
        write_dir: None | str = None,
        image_export_format: str = "dict",
        **kwargs
):
    """
    Process images and masks. This generator is a wrapper around
    :func:`~mirp.extractFeaturesAndImages.extract_features_and_images_generator`.

    Parameters
    ----------
    write_images: bool, default: True
       Determines whether processed images and masks should be written to the directory indicated by the
       ``write_dir`` keyword argument.

    export_images: bool, default: False
       Determines whether processed images and masks should be returned by the function.

    write_dir: str, optional
        Path to directory where processed images and masks should be written. If not set, processed images and masks
        are returned by this function. Required if ``write_images=True``.

    image_export_format: {"dict", "native", "numpy"}, default: "dict"
        Return format for processed images and masks. ``"dict"`` returns dictionaries of images and masks as numpy
        arrays and associated characteristics. ``"native"`` returns images and masks in their internal format.
        ``"numpy"`` returns images and masks in numpy format. This argument is only used if ``export_images=True``.

    **kwargs:
        Keyword arguments passed for importing images and masks (
        :func:`mirp.data_import.import_image_and_mask.import_image_and_mask`) and configuring settings:

        * general settings (:class:`~mirp.settings.general_parameters.GeneralSettingsClass`)
        * image post-processing (:class:`~mirp.settings.image_processing_parameters.ImagePostProcessingClass`)
        * image perturbation / augmentation (:class:`~mirp.settings.perturbation_parameters.ImagePerturbationSettingsClass`)
        * image interpolation / resampling (
          :class:`~mirp.settings.interpolation_parameters.ImageInterpolationSettingsClass` and
          :class:`~mirp.settings.interpolation_parameters.MaskInterpolationSettingsClass`)
        * mask resegmentation (:class:`~mirp.settings.resegmentation_parameters.ResegmentationSettingsClass`)
        * image transformation (:class:`~mirp.settings.transformation_parameters.ImageTransformationSettingsClass`)

    Yields
    ------
    None | list[Any]
       List of feature tables, if ``export_images=True``.

    See Also
    --------
    :func:`~mirp.extractFeaturesAndImages.extract_features_and_images_generator`

    """
    yield from extract_features_and_images_generator(
        write_features=False,
        export_features=False,
        write_images=write_images,
        export_images=export_images,
        write_dir=write_dir,
        image_export_format=image_export_format,
        **kwargs
    )


def extract_features_and_images(
        write_features: None | bool = None,
        export_features: None | bool = None,
        write_images: None | bool = None,
        export_images: None | bool = None,
        write_dir: None | str = None,
        image_export_format: str = "dict",
        num_cpus: None | int = None,
        parallel_backend: None | str = None,
        **kwargs
):
    """
    Processes images and computes features from regions of interest.

    Parameters
    ----------
    write_features: bool, optional
        Determines whether features computed from images should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    export_features: bool, optional
        Determines whether features computed from images should be returned by the function.

    write_images: bool, optional
        Determines whether processed images and masks should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    export_images: bool, optional
        Determines whether processed images and masks should be returned by the function.

    write_dir: str, optional
        Path to directory where features, processed images and masks should be written. If not set, features, processed images and masks
        are returned by this function. Required if ``write_features=True`` or ``write_images=True``.

    image_export_format: {"dict", "native", "numpy"}, default: "dict"
        Return format for processed images and masks. ``"dict"`` returns dictionaries of images and masks as numpy
        arrays and associated characteristics. ``"native"`` returns images and masks in their internal format.
        ``"numpy"`` returns images and masks in numpy format. This argument is only used if ``export_images=True``.

    num_cpus: int, optional, default: None
        Number of CPU nodes that should be used for parallel processing. Image and mask processing can be
        parallelized using the ``ray`` or ``joblib`` packages. If a ray cluster is defined by the user, this cluster
        will be used instead. By default, image and mask processing are processed sequentially.

    parallel_backend: {"none", "ray", "joblib"}, optional, default: "none"
        Type of backend to use. Default is the sequential backend (``"none"``). Alternative backends are ``"ray"`` and
        ``"joblib"``, which rely on the ray and joblib libraries respectively.

    **kwargs:
        Keyword arguments passed for importing images and masks (
        :func:`mirp.data_import.import_image_and_mask.import_image_and_mask`) and configuring settings:

        * general settings (:class:`~mirp.settings.general_parameters.GeneralSettingsClass`)
        * image post-processing (:class:`~mirp.settings.image_processing_parameters.ImagePostProcessingClass`)
        * image perturbation / augmentation (:class:`~mirp.settings.perturbation_parameters.ImagePerturbationSettingsClass`)
        * image interpolation / resampling (
          :class:`~mirp.settings.interpolation_parameters.ImageInterpolationSettingsClass` and
          :class:`~mirp.settings.interpolation_parameters.MaskInterpolationSettingsClass`)
        * mask resegmentation (:class:`~mirp.settings.resegmentation_parameters.ResegmentationSettingsClass`)
        * image transformation (:class:`~mirp.settings.transformation_parameters.ImageTransformationSettingsClass`)
        * feature computation / extraction (
          :class:`~mirp.settings.feature_parameters.FeatureExtractionSettingsClass`)

    Returns
    -------
    None | list[Any]
        List of features, images and masks, depending on ``export_features`` and ``export_images``.

    See Also
    --------
    Keyword arguments can be provided to configure the following:

    * image and mask import (:func:`~mirp.data_import.import_image_and_mask.import_image_and_mask`)
    * general settings (:class:`~mirp.settings.general_parameters.GeneralSettingsClass`)
    * image post-processing (:class:`~mirp.settings.image_processing_parameters.ImagePostProcessingClass`)
    * image perturbation / augmentation (:class:`~mirp.settings.perturbation_parameters.ImagePerturbationSettingsClass`)
    * image interpolation / resampling (:class:`~mirp.settings.interpolation_parameters.ImageInterpolationSettingsClass` and
      :class:`~mirp.settings.interpolation_parameters.MaskInterpolationSettingsClass`)
    * mask resegmentation (:class:`~mirp.settings.resegmentation_parameters.ResegmentationSettingsClass`)
    * image transformation (:class:`~mirp.settings.transformation_parameters.ImageTransformationSettingsClass`)
    * feature computation / extraction (:class:`~mirp.settings.feature_parameters.FeatureExtractionSettingsClass`)

    """

    # Configure logger
    logging.basicConfig(
        format="%(levelname)s\t: %(processName)s \t %(asctime)s \t %(message)s",
        level=logging.INFO,
        stream=sys.stdout
    )

    backend = parse_parallel_backend(backend=parallel_backend, num_cpus=num_cpus)
    external_cluster = cluster_exists(backend=backend)
    start_parallel_cluster(backend=backend, num_cpus=num_cpus)

    # Switch to sequential backend if ray cluster is not formed somehow.
    if backend == "ray" and not cluster_exists(backend=backend):
        backend = "none"

    logging.info(message_parallel_process(backend=backend, num_cpus=num_cpus))

    if backend == "none":
        results = [
            workflow.standard_extraction(image_export_format=image_export_format)
            for workflow in _base_extract_features_and_images(
                write_features=write_features,
                export_features=export_features,
                write_images=write_images,
                export_images=export_images,
                write_dir=write_dir,
                **kwargs
            )
        ]

    elif backend == "ray":
        results = [
            _ray_extractor.remote(
                workflow=workflow,
                image_export_format=image_export_format
            )
            for workflow in _base_extract_features_and_images(
                write_features=write_features,
                export_features=export_features,
                write_images=write_images,
                export_images=export_images,
                write_dir=write_dir,
                **kwargs
            )
        ]
        results = ray_get(results)

    elif backend == "joblib":
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=num_cpus)(
            delayed(workflow.standard_extraction)(
                image_export_format=image_export_format
            ) for workflow in _base_extract_features_and_images(
                write_features=write_features,
                export_features=export_features,
                write_images=write_images,
                export_images=export_images,
                write_dir=write_dir,
                **kwargs
            )
        )

    else:
        raise ValueError(f"parallel_backend is expected to be one of 'none', 'ray' or 'joblib'. Found: {backend}")

    if not external_cluster:
        shutdown_cluster(backend=backend)

    return results


def extract_features_and_images_generator(
        write_features: None | bool = None,
        export_features: None | bool = None,
        write_images: None | bool = None,
        export_images: None | bool = None,
        write_dir: None | str = None,
        image_export_format: str = "dict",
        num_cpus: None | int = None,
        parallel_backend: None | str = None,
        **kwargs
):
    """
    Processes images and computes features from regions of interest as a generator.

    Parameters
    ----------
    write_features: bool, optional
        Determines whether features computed from images should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    export_features: bool, optional
        Determines whether features computed from images should be returned by the function.

    write_images: bool, optional
        Determines whether processed images and masks should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    export_images: bool, optional
        Determines whether processed images and masks should be returned by the function.

    write_dir: str, optional
        Path to directory where features, processed images and masks should be written. If not set, features, processed images and masks
        are returned by this function. Required if ``write_features=True`` or ``write_images=True``.

    image_export_format: {"dict", "native", "numpy"}, default: "dict"
        Return format for processed images and masks. ``"dict"`` returns dictionaries of images and masks as numpy
        arrays and associated characteristics. ``"native"`` returns images and masks in their internal format.
        ``"numpy"`` returns images and masks in numpy format. This argument is only used if ``export_images=True``.

    num_cpus: int, optional, default: None
        Number of CPU nodes that should be used for parallel processing. Image and mask processing can be
        parallelized using the ``joblib`` package. By default, image and mask processing are processed sequentially.

    parallel_backend: {"none", "joblib"}, optional, default: "none"
        Type of backend to use. Default is the sequential backend (``"none"``). ``"joblib"`` can be used as
        an alternative backend. ``"ray"`` cannot be used in a generator context, because only a single worker will be
        used.

    **kwargs:
        Keyword arguments passed for importing images and masks (
        :func:`mirp.data_import.import_image_and_mask.import_image_and_mask`) and configuring settings:

        * general settings (:class:`~mirp.settings.general_parameters.GeneralSettingsClass`)
        * image post-processing (:class:`~mirp.settings.image_processing_parameters.ImagePostProcessingClass`)
        * image perturbation / augmentation (:class:`~mirp.settings.perturbation_parameters.ImagePerturbationSettingsClass`)
        * image interpolation / resampling (
          :class:`~mirp.settings.interpolation_parameters.ImageInterpolationSettingsClass` and
          :class:`~mirp.settings.interpolation_parameters.MaskInterpolationSettingsClass`)
        * mask resegmentation (:class:`~mirp.settings.resegmentation_parameters.ResegmentationSettingsClass`)
        * image transformation (:class:`~mirp.settings.transformation_parameters.ImageTransformationSettingsClass`)
        * feature computation / extraction (
          :class:`~mirp.settings.feature_parameters.FeatureExtractionSettingsClass`)

    Yields
    ------
    None | list[Any]
        List of features, images and masks, depending on ``export_features`` and ``export_images``.

    See Also
    --------
    Keyword arguments can be provided to configure the following:

    * image and mask import (:func:`~mirp.data_import.import_image_and_mask.import_image_and_mask`)
    * general settings (:class:`~mirp.settings.general_parameters.GeneralSettingsClass`)
    * image post-processing (:class:`~mirp.settings.image_processing_parameters.ImagePostProcessingClass`)
    * image perturbation / augmentation (:class:`~mirp.settings.perturbation_parameters.ImagePerturbationSettingsClass`)
    * image interpolation / resampling (:class:`~mirp.settings.interpolation_parameters.ImageInterpolationSettingsClass`
      and :class:`~mirp.settings.interpolation_parameters.MaskInterpolationSettingsClass`)
    * mask resegmentation (:class:`~mirp.settings.resegmentation_parameters.ResegmentationSettingsClass`)
    * image transformation (:class:`~mirp.settings.transformation_parameters.ImageTransformationSettingsClass`)
    * feature computation / extraction (:class:`~mirp.settings.feature_parameters.FeatureExtractionSettingsClass`)

    """

    # Configure logger
    logging.basicConfig(
        format="%(levelname)s\t: %(processName)s \t %(asctime)s \t %(message)s",
        level=logging.INFO,
        stream=sys.stdout
    )

    # Do not allow ray as a backend.
    backend = parse_parallel_backend(backend=parallel_backend, num_cpus=num_cpus, ray_allowed=False)
    external_cluster = cluster_exists(backend=backend)
    start_parallel_cluster(backend=backend, num_cpus=num_cpus)

    # Switch to sequential backend if ray cluster is not formed somehow.
    if backend == "ray" and not cluster_exists(backend=backend):
        backend = "none"

    logging.info(message_parallel_process(backend=backend, num_cpus=num_cpus))

    if backend == "none":
        for workflow in _base_extract_features_and_images(
            write_features=write_features,
            export_features=export_features,
            write_images=write_images,
            export_images=export_images,
            write_dir=write_dir,
            **kwargs
        ):
            yield workflow.standard_extraction(image_export_format=image_export_format)

    elif backend == "joblib":
        from joblib import Parallel, delayed
        parallel_gen = Parallel(n_jobs=num_cpus, return_as="generator")(
            delayed(workflow.standard_extraction)(
                image_export_format=image_export_format
            ) for workflow in _base_extract_features_and_images(
                write_features=write_features,
                export_features=export_features,
                write_images=write_images,
                export_images=export_images,
                write_dir=write_dir,
                **kwargs
            )
        )

        yield from parallel_gen

    else:
        raise ValueError(f"parallel_backend is expected to be one of 'none' or 'joblib'. Found: {backend}")

    if not external_cluster:
        shutdown_cluster(backend=backend)


@ray_remote
def _ray_extractor(workflow: StandardWorkflow, image_export_format="dict"):
    # Limit internal threading by third-party libraries.
    from mirp.utilities.parallel_ray import limit_inner_threads
    limit_inner_threads()

    return workflow.standard_extraction(image_export_format=image_export_format)


def _base_extract_features_and_images(
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
        write_features: None | bool = None,
        export_features: None | bool = None,
        write_images: None | bool = None,
        export_images: None | bool = None,
        write_dir: None | str = None,
        **kwargs
):
    from mirp.data_import.import_image_and_mask import import_image_and_mask
    from mirp.settings.import_config_parameters import import_configuration_settings

    # Infer write_images, export_images, write_features, export_features based on write_dir.
    if write_images is None:
        write_images = write_dir is not None
    if export_images is None:
        export_images = write_dir is None
    if write_features is None:
        write_features = write_dir is not None
    if export_features is None:
        export_features = write_dir is None

    if not write_images and not write_features:
        write_dir = None

    if write_images and write_dir is None:
        raise ValueError("write_dir argument is required for writing images and masks, but not provided.")
    if write_features and write_dir is None:
        raise ValueError("write_dir argument is required for writing feature tables, but not provided.")

    if not write_features and not write_images and not export_features and not export_images:
        raise ValueError(
            f"At least one of write_features, write_images, export_features and export_images should be True."
        )

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
        raise TypeError(
            f"The 'settings' argument is expected to be a path to a configuration xml file, a SettingsClass object, or "
            f"a list thereof. Found: {type(settings)}."
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
        image_list: list[ImageFile],
        settings: list[SettingsClass],
        write_dir: None | str,
        write_features: bool,
        export_features: bool,
        write_images: bool,
        export_images: bool
) -> Generator[StandardWorkflow, None, None]:

    for image_file in image_list:

        # Check for potentially problematic stuff.
        image_file.check_associated_masks()

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
