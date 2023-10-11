from typing import Union, List, Dict, Optional, Generator, Iterable, Any
import copy

import ray

from mirp.importData.imageGenericFile import ImageFile
from mirp.settings.settingsGeneric import SettingsClass
from mirp.workflows.standardWorkflow import StandardWorkflow


def extract_features(
        write_features: bool = True,
        export_features: bool = False,
        **kwargs
) -> None | list[Any]:
    """
    Compute features from regions of interest in images. This function is a wrapper around
    :func:`mirp.extractFeaturesAndImages.extract_features_and_images`.

    Parameters
    ----------
    write_features: bool, default: True
        Determines whether features computed from images should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    export_features: bool, default: False
        Determines whether features computed from images should be returned by the function.

    **kwargs:
        Keyword arguments passed to :func:`mirp.extractFeaturesAndImages.extract_features_and_images`.

    Returns
    -------
    None | list[Any]
        List of feature tables, if ``export_features=True``.

    See Also
    --------
    :func:`mirp.extractFeaturesAndImages.extract_features_and_images`

    """
    return extract_features_and_images(
        write_features=write_features,
        export_features=export_features,
        write_images=False,
        export_images=False,
        **kwargs
    )


def extract_features_generator(
        write_features: bool = False,
        export_features: bool = True,
        **kwargs
):
    """
    Compute features from regions of interest in images. This generator is a wrapper around
    :func:`mirp.extractFeaturesAndImages.extract_features_and_images_generator`.

    Parameters
    ----------
    write_features: bool, default: False
        Determines whether features computed from images should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    export_features: bool, default: True
        Determines whether features computed from images should be returned by the function.

    **kwargs:
        Keyword arguments passed to :func:`mirp.extractFeaturesAndImages.extract_features_and_images_generator`.

    Returns
    -------
    None | list[Any]
        List of feature tables, if ``export_features=True``.

    See Also
    --------
    :func:`mirp.extractFeaturesAndImages.extract_features_and_images_generator`

    """
    yield from extract_features_and_images_generator(
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
    """
    Process images and masks. This function is a wrapper around
    :func:`mirp.extractFeaturesAndImages.extract_features_and_images`.

    Parameters
    ----------
    write_images: bool, default: True
        Determines whether processed images and masks should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    export_images: bool, default: False
        Determines whether processed images and masks should be returned by the function.

    **kwargs:
        Keyword arguments passed to :func:`mirp.extractFeaturesAndImages.extract_features_and_images`.

    Returns
    -------
    None | list[Any]
        List of feature tables, if ``export_images=True``.

    See Also
    --------
    :func:`mirp.extractFeaturesAndImages.extract_features_and_images`

    """
    return extract_features_and_images(
        write_features=False,
        export_features=False,
        write_images=write_images,
        export_images=export_images,
        **kwargs
    )


def extract_images_generator(
        write_images: bool = False,
        export_images: bool = True,
        **kwargs
):
    """
    Process images and masks. This generator is a wrapper around
    :func:`mirp.extractFeaturesAndImages.extract_features_and_images_generator`.

    Parameters
    ----------
    write_images: bool, default: True
       Determines whether processed images and masks should be written to the directory indicated by the
       ``write_dir`` keyword argument.

    export_images: bool, default: False
       Determines whether processed images and masks should be returned by the function.

    **kwargs:
       Keyword arguments passed to :func:`mirp.extractFeaturesAndImages.extract_features_and_images_generator`.

    Yields
    ------
    None | list[Any]
       List of feature tables, if ``export_images=True``.

    See Also
    --------
    :func:`mirp.extractFeaturesAndImages.extract_features_and_images_generator`

    """
    yield from extract_features_and_images_generator(
        write_features=False,
        export_features=False,
        write_images=write_images,
        export_images=export_images,
        **kwargs
    )


def extract_features_and_images(
        image_export_format: str = "dict",
        num_cpus: None | int = None,
        **kwargs
):
    """
    Processes images and computes features from regions of interest.

    Parameters
    ----------
    image_export_format: {"dict", "native", "numpy"}, default: "numpy"
        Return format for processed images and masks. ``"dict"`` returns dictionaries of images and masks as numpy
        arrays and associated characteristics. ``"native"`` returns images and masks in their internal format.
        ``"numpy"`` returns images and masks in numpy format. This argument is only used if ``export_images=True``.

    num_cpus: int, optional, default: None
        Number of CPU nodes that should be used for parallel processing. Image processing and feature computation can be
        parallelized using the ``ray`` package. If a ray cluster is defined by the user, this cluster will be used
        instead. By default, images are processed sequentially.

    **kwargs:
        Keyword arguments passed for importing images and masks (
        :func:`mirp.importData.importImageAndMask.import_image_and_mask`) and configuring settings:

        * general settings (:class:`mirp.settings.settingsGeneral.GeneralSettingsClass`)
        * image post-processing (:class:`mirp.settings.settingsImageProcessing.ImagePostProcessingClass`)
        * image perturbation / augmentation (:class:`mirp.settings.settingPerturbation.ImagePerturbationSettingsClass`)
        * image interpolation / resampling (
          :class:`mirp.settings.settingsInterpolation.ImageInterpolationSettingsClass` and
          :class:`mirp.settings.settingsInterpolation.MaskInterpolationSettingsClass`)
        * mask resegmentation (:class:`mirp.settings.settingsMaskResegmentation.ResegmentationSettingsClass`)
        * image transformation (:class:`mirp.settings.settingsImageTransformation.ImageTransformationSettingsClass`)
        * feature computation / extraction (
          :class:`mirp.settings.settingsFeatureExtraction.FeatureExtractionSettingsClass`)

        See also the `Other Parameters` section below.

    Returns
    -------
    None | list[Any]
        List of features, images and masks, depending on ``export_features`` and ``export_images``.

    Other Parameters
    ----------------
    .. note::
        The parameters below can be provided as keyword arguments.

    write_dir: str, optional
        Path to directory where processed images, masks and feature tables should be written.

    write_features: bool, default: False
        Determines whether features computed from images should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    export_features: bool, default: False
        Determines whether features computed from images should be returned by the function.

    write_images: bool, default: False
        Determines whether processed images and masks should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    export_images: bool, default: False
        Determines whether processed images and masks should be returned by the function.

    image: Any
        A path to an image file, a path to a directory containing image files, a path to a config_data.xml
        file, a path to a csv file containing references to image files, a pandas.DataFrame containing references to
        image files, or a numpy.ndarray.

    mask: Any
        A path to a mask file, a path to a directory containing mask files, a path to a config_data.xml
        file, a path to a csv file containing references to mask files, a pandas.DataFrame containing references to
        mask files, or a numpy.ndarray.

    sample_name: str or list of str, default: None
        Name of expected sample names. This is used to select specific image files. If None, no image files are
        filtered based on the corresponding sample name (if known).

    image_name: str, optional, default: None
        Pattern to match image files against. The matches are exact. Use wildcard symbols ("*") to
        match varying structures. The sample name (if part of the file name) can also be specified using "#". For
        example, image_name = '#_*_image' would find John_Doe in John_Doe_CT_image.nii or John_Doe_001_image.nii.
        File extensions do not need to be specified. If None, file names are not used for filtering files and
        setting sample names.

    image_file_type: {"dicom", "nifti", "nrrd", "numpy", "itk"}, optional, default: None
        The type of file that is expected. If None, the file type is not used for filtering files.
        "itk" comprises "nifti" and "nrrd" file types.

    image_modality: {"ct", "pet", "pt", "mri", "mr", "generic"}, optional, default: None
        The type of modality that is expected. If None, modality is not used for filtering files. Note that only
        DICOM files contain metadata concerning modality.

    image_sub_folder: str, optional, default: None
        Fixed directory substructure where image files are located. If None, the directory substructure is not used
        for filtering files.

    mask_name: str or list of str, optional, default: None
        Pattern to match mask files against. The matches are exact. Use wildcard symbols ("*") to match varying
        structures. The sample name (if part of the file name) can also be specified using "#". For example,
        mask_name = '#_*_mask' would find John_Doe in John_Doe_CT_mask.nii or John_Doe_001_mask.nii. File extensions
        do not need to be specified. If None, file names are not used for filtering files and setting sample names.

    mask_file_type: {"dicom", "nifti", "nrrd", "numpy", "itk"}, optional, default: None
        The type of file that is expected. If None, the file type is not used for filtering files.
        "itk" comprises "nifti" and "nrrd" file types.

    mask_modality: {"rtstruct", "seg", "generic_mask"}, optional, default: None
        The type of modality that is expected. If None, modality is not used for filtering files.
        Note that only DICOM files contain metadata concerning modality. Masks from non-DICOM files are considered to
        be "generic_mask".

    mask_sub_folder: str, optional, default: None
        Fixed directory substructure where mask files are located. If None, the directory substructure is not used for
        filtering files.

    roi_name: str, optional, default: None
        Name of the regions of interest that should be assessed.

    association_strategy: {"frame_of_reference", "sample_name", "file_distance", "file_name_similarity",  "list_order", "position", "single_image"}
        The preferred strategy for associating images and masks. File association is preferably done using frame of
        reference UIDs (DICOM), or sample name (NIfTI, numpy). Other options are relatively frail, except for
        `list_order` which may be applicable when a list with images and a list with masks is provided and both lists
        are of equal length.

    stack_images: {"auto", "yes", "no"}, optional, default: "str"
        If image files in the same directory cannot be assigned to different samples, and are 2D (slices) of the same
        size, they might belong to the same 3D image stack. "auto" will stack 2D numpy arrays, but not other file types.
        "yes" will stack all files that contain 2D images, that have the same dimensions, orientation and spacing,
        except for DICOM files. "no" will not stack any files. DICOM files ignore this argument, because their stacking
        can be determined from metadata.

    stack_masks: {"auto", "yes", "no"}, optional, default: "str"
        If mask files in the same directory cannot be assigned to different samples, and are 2D (slices) of the same
        size, they might belong to the same 3D mask stack. "auto" will stack 2D numpy arrays, but not other file
        types. "yes" will stack all files that contain 2D images, that have the same dimensions, orientation and
        spacing, except for DICOM files. "no" will not stack any files. DICOM files ignore this argument,
        because their stacking can be determined from metadata.

    See Also
    --------
    * general settings (:class:`mirp.settings.settingsGeneral.GeneralSettingsClass`)
    * image post-processing (:class:`mirp.settings.settingsImageProcessing.ImagePostProcessingClass`)
    * image perturbation / augmentation (:class:`mirp.settings.settingPerturbation.ImagePerturbationSettingsClass`)
    * image interpolation / resampling (:class:`mirp.settings.settingsInterpolation.ImageInterpolationSettingsClass` and
      :class:`mirp.settings.settingsInterpolation.MaskInterpolationSettingsClass`)
    * mask resegmentation (:class:`mirp.settings.settingsMaskResegmentation.ResegmentationSettingsClass`)
    * image transformation (:class:`mirp.settings.settingsImageTransformation.ImageTransformationSettingsClass`)
    * feature computation / extraction (:class:`mirp.settings.settingsFeatureExtraction.FeatureExtractionSettingsClass`)

    """

    # Conditionally start a ray cluster.
    external_ray = ray.is_initialized()
    if not external_ray and num_cpus is not None and num_cpus > 1:
        ray.init(num_cpus=num_cpus)

    if ray.is_initialized():
        # Parallel processing.
        results = [
            _ray_extractor.remote(workflow=workflow, image_export_format=image_export_format)
            for workflow in _base_extract_features_and_images(**kwargs)
        ]

        results = ray.get(results)
        if not external_ray:
            ray.shutdown()

    else:
        # Sequential processing.
        workflows = list(_base_extract_features_and_images(**kwargs))
        results = [workflow.standard_extraction(image_export_format=image_export_format) for workflow in workflows]

    return results


def extract_features_and_images_generator(
        image_export_format: str = "dict",
        **kwargs
):
    """
    Processes images and computes features from regions of interest as a generator.

    Parameters
    ----------
    image_export_format: {"dict", "native", "numpy"}, default: "numpy"
        Return format for processed images and masks. ``"dict"`` returns dictionaries of images and masks as numpy
        arrays and associated characteristics. ``"native"`` returns images and masks in their internal format.
        ``"numpy"`` returns images and masks in numpy format. This argument is only used if ``export_images=True``.

    **kwargs:
        Keyword arguments passed for importing images and masks (
        :func:`mirp.importData.importImageAndMask.import_image_and_mask`) and configuring settings:

        * general settings (:class:`mirp.settings.settingsGeneral.GeneralSettingsClass`)
        * image post-processing (:class:`mirp.settings.settingsImageProcessing.ImagePostProcessingClass`)
        * image perturbation / augmentation (:class:`mirp.settings.settingPerturbation.ImagePerturbationSettingsClass`)
        * image interpolation / resampling (
          :class:`mirp.settings.settingsInterpolation.ImageInterpolationSettingsClass` and
          :class:`mirp.settings.settingsInterpolation.MaskInterpolationSettingsClass`)
        * mask resegmentation (:class:`mirp.settings.settingsMaskResegmentation.ResegmentationSettingsClass`)
        * image transformation (:class:`mirp.settings.settingsImageTransformation.ImageTransformationSettingsClass`)
        * feature computation / extraction (
          :class:`mirp.settings.settingsFeatureExtraction.FeatureExtractionSettingsClass`)

        See also the `Other Parameters` section below.

    Yields
    ------
    None | list[Any]
        List of features, images and masks, depending on ``export_features`` and ``export_images``.

    Other Parameters
    ----------------
    .. note::
        The parameters below can be provided as keyword arguments.

    write_dir: str, optional
        Path to directory where processed images, masks and feature tables should be written.

    write_features: bool, default: False
        Determines whether features computed from images should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    export_features: bool, default: False
        Determines whether features computed from images should be returned by the function.

    write_images: bool, default: False
        Determines whether processed images and masks should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    export_images: bool, default: False
        Determines whether processed images and masks should be returned by the function.

    image: Any
        A path to an image file, a path to a directory containing image files, a path to a config_data.xml
        file, a path to a csv file containing references to image files, a pandas.DataFrame containing references to
        image files, or a numpy.ndarray.

    mask: Any
        A path to a mask file, a path to a directory containing mask files, a path to a config_data.xml
        file, a path to a csv file containing references to mask files, a pandas.DataFrame containing references to
        mask files, or a numpy.ndarray.

    sample_name: str or list of str, default: None
        Name of expected sample names. This is used to select specific image files. If None, no image files are
        filtered based on the corresponding sample name (if known).

    image_name: str, optional, default: None
        Pattern to match image files against. The matches are exact. Use wildcard symbols ("*") to
        match varying structures. The sample name (if part of the file name) can also be specified using "#". For
        example, image_name = '#_*_image' would find John_Doe in John_Doe_CT_image.nii or John_Doe_001_image.nii.
        File extensions do not need to be specified. If None, file names are not used for filtering files and
        setting sample names.

    image_file_type: {"dicom", "nifti", "nrrd", "numpy", "itk"}, optional, default: None
        The type of file that is expected. If None, the file type is not used for filtering files.
        "itk" comprises "nifti" and "nrrd" file types.

    image_modality: {"ct", "pet", "pt", "mri", "mr", "generic"}, optional, default: None
        The type of modality that is expected. If None, modality is not used for filtering files. Note that only
        DICOM files contain metadata concerning modality.

    image_sub_folder: str, optional, default: None
        Fixed directory substructure where image files are located. If None, the directory substructure is not used
        for filtering files.

    mask_name: str or list of str, optional, default: None
        Pattern to match mask files against. The matches are exact. Use wildcard symbols ("*") to match varying
        structures. The sample name (if part of the file name) can also be specified using "#". For example,
        mask_name = '#_*_mask' would find John_Doe in John_Doe_CT_mask.nii or John_Doe_001_mask.nii. File extensions
        do not need to be specified. If None, file names are not used for filtering files and setting sample names.

    mask_file_type: {"dicom", "nifti", "nrrd", "numpy", "itk"}, optional, default: None
        The type of file that is expected. If None, the file type is not used for filtering files.
        "itk" comprises "nifti" and "nrrd" file types.

    mask_modality: {"rtstruct", "seg", "generic_mask"}, optional, default: None
        The type of modality that is expected. If None, modality is not used for filtering files.
        Note that only DICOM files contain metadata concerning modality. Masks from non-DICOM files are considered to
        be "generic_mask".

    mask_sub_folder: str, optional, default: None
        Fixed directory substructure where mask files are located. If None, the directory substructure is not used for
        filtering files.

    roi_name: str, optional, default: None
        Name of the regions of interest that should be assessed.

    association_strategy: {"frame_of_reference", "sample_name", "file_distance", "file_name_similarity",  "list_order", "position", "single_image"}
        The preferred strategy for associating images and masks. File association is preferably done using frame of
        reference UIDs (DICOM), or sample name (NIfTI, numpy). Other options are relatively frail, except for
        `list_order` which may be applicable when a list with images and a list with masks is provided and both lists
        are of equal length.

    stack_images: {"auto", "yes", "no"}, optional, default: "str"
        If image files in the same directory cannot be assigned to different samples, and are 2D (slices) of the same
        size, they might belong to the same 3D image stack. "auto" will stack 2D numpy arrays, but not other file types.
        "yes" will stack all files that contain 2D images, that have the same dimensions, orientation and spacing,
        except for DICOM files. "no" will not stack any files. DICOM files ignore this argument, because their stacking
        can be determined from metadata.

    stack_masks: {"auto", "yes", "no"}, optional, default: "str"
        If mask files in the same directory cannot be assigned to different samples, and are 2D (slices) of the same
        size, they might belong to the same 3D mask stack. "auto" will stack 2D numpy arrays, but not other file
        types. "yes" will stack all files that contain 2D images, that have the same dimensions, orientation and
        spacing, except for DICOM files. "no" will not stack any files. DICOM files ignore this argument,
        because their stacking can be determined from metadata.

    See Also
    --------
    * general settings (:class:`mirp.settings.settingsGeneral.GeneralSettingsClass`)
    * image post-processing (:class:`mirp.settings.settingsImageProcessing.ImagePostProcessingClass`)
    * image perturbation / augmentation (:class:`mirp.settings.settingPerturbation.ImagePerturbationSettingsClass`)
    * image interpolation / resampling (:class:`mirp.settings.settingsInterpolation.ImageInterpolationSettingsClass` and
      :class:`mirp.settings.settingsInterpolation.MaskInterpolationSettingsClass`)
    * mask resegmentation (:class:`mirp.settings.settingsMaskResegmentation.ResegmentationSettingsClass`)
    * image transformation (:class:`mirp.settings.settingsImageTransformation.ImageTransformationSettingsClass`)
    * feature computation / extraction (:class:`mirp.settings.settingsFeatureExtraction.FeatureExtractionSettingsClass`)

    """

    workflows = list(_base_extract_features_and_images(**kwargs))
    for workflow in workflows:
        yield workflow.standard_extraction(image_export_format=image_export_format)


@ray.remote
def _ray_extractor(workflow: StandardWorkflow, image_export_format="dict"):
    return workflow.standard_extraction(image_export_format=image_export_format)


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
