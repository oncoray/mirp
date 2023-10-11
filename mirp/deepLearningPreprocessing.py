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
        export_images: bool = False,
        write_images: bool = True,
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

    image_export_format: {"dict", "native", "numpy"}, default: "numpy"
        Return format for processed images and masks. ``"dict"`` returns dictionaries of images and masks as numpy
        arrays and associated characteristics. ``"native"`` returns images and masks in their internal format.
        ``"numpy"`` returns images and masks in numpy format. This argument is only used if ``export_images=True``.

    write_file_format: {"nifti", "numpy"}, default: "numpy"
        File format for processed images and masks. ``"nifti"`` writes images and masks in the NIfTI file format,
        and ``"numpy"`` writes images and masks as numpy files. This argument is only used if ``write_images=True``.

    export_images: bool, default: False
        Determines whether processed images and masks should be returned by the function.

    write_images: bool, default: True
        Determines whether processed images and masks should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    num_cpus: int, optional, default: None
        Number of CPU nodes that should be used for parallel processing. Image and mask processing can be
        parallelized using the ``ray`` package. If a ray cluster is defined by the user, this cluster will be used
        instead. By default, image and mask processing are processed sequentially.

    **kwargs:
        Keyword arguments passed for importing images and masks (
        :func:`mirp.importData.importImageAndMask.import_image_and_mask`) and configuring settings (notably
        :class:`mirp.settings.settingsImageProcessing.ImagePostProcessingClass`,
        :class:`mirp.settings.settingsPerturbation.ImagePerturbationSettingsClass`), among others. See also the
        `Other Parameters` section below.

    Returns
    -------
    None | list[Any]
        List of images and masks in the format indicated by ``image_export_format``, if ``export_images=True``.

    Other Parameters
    ----------------
    .. note::
        The parameters below can be provided as keyword arguments.

    write_dir: str, optional
        Path to directory where processed images and masks should be written.

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

    intensity_normalisation: {"none", "range", "relative_range", "quantile_range", "standardisation"}, default: "none"
        Specifies the algorithm used to normalise intensities in the image. Will use only intensities in voxels
        masked by the tissue mask (of present). The following are possible:

        * "none": no normalisation
        * "range": normalises intensities based on a fixed mapping against the ``intensity_normalisation_range``
          parameter, which is interpreted to represent an intensity range.
        * "relative_range": normalises intensities based on a fixed mapping against the ``intensity_normalisation_range``
          parameter, which is interpreted to represent a relative intensity range.
        * "quantile_range": normalises intensities based on a fixed mapping against the
          ``intensity_normalisation_range`` parameter, which is interpreted to represent a quantile range.
        * "standardisation": normalises intensities by subtraction of the mean intensity and division by the standard
          deviation of intensities.

    .. note::
        intensity normalisation may remove any physical meaning of intensity units.

    intensity_normalisation_range: list of float, optional
        Required for "range", "relative_range", and "quantile_range" intensity normalisation methods, and defines the
        intensities that are mapped to the [0.0, 1.0] range during normalisation. The default range depends on the
        type of normalisation method:

        * "range": [np.nan, np.nan]: the minimum and maximum intensity value present in the image are used to set the
          mapping range.
        * "relative_range": [0.0. 1.0]: the minimum (0.0) and maximum (1.0) intensity value present in the image are
          used to set the mapping range.
        * "quantile_range": [0.025, 0.975] the 2.5th and 97.5th percentiles of the intensities in the image are used
          to set the mapping range.

        The lower end of the range is mapped to 0.0 and the upper end to 1.0. However, if intensities below the lower
        end or above the upper end are present in the image, values below 0.0 or above 1.0 may be encountered after
        normalisation. Use ``intensity_normalisation_saturation`` to cap intensities after normalisation to a
        specific range.

    intensity_normalisation_saturation: list of float, optional, default: [np.nan, np.nan]
        Defines the start and endpoint for the saturation range. Normalised intensities that lie outside this
        range are mapped to the limits of the saturation range, e.g. with a range of [0.0, 0.8] all values greater
        than 0.8 are assigned a value of 0.8. np.nan can be used to define limits where the intensity values should
        not be saturated.

    tissue_mask_type: {"none", "range", "relative_range"}, optional, default: "relative_range"
        Type of algorithm used to produce an approximate tissue mask of the tissue. Such masks can be used to select
         pixels for bias correction and intensity normalisation by excluding non-tissue voxels.

    tissue_mask_range: list of float, optional
        Range values for creating an approximate mask of the tissue. Required for "range" and "relative_range"
        options. Default: [0.02, 1.00] (``"relative_range"``); [np.nan, np.nan] (``"range"``; effectively all voxels
        are considered to represent tissue).

        interpolate: bool, optional, default: False
        Controls whether interpolation of images to a common grid is performed at all.

    spline_order: int, optional, default: 3
        Sets the spline order used for spline interpolation. mirp uses `scipy.ndimage.map_coordinates
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage
        .map_coordinates>`_ internally. Spline orders 0, 1, and 3 refer to nearest neighbour, linear interpolation
        and cubic interpolation, respectively.

    new_spacing: float or list of float or list of list of float:
        Sets voxel spacing after interpolation. A single value represents the spacing that will be applied in all
        directions. Non-uniform voxel spacing may also be provided, but requires 3 values for z, y, and x directions
        (if `by_slice = False`) or 2 values for y and x directions (otherwise).

        Multiple spacings may be defined by creating a nested list, e.g. [[1.0], [1.5], [2.0]] to resample the
        same image multiple times to different (here: isotropic) voxel spacings, namely 1.0, 1.5 and 2.0. Units
        are defined by the headers of the image files. These are typically millimeters for radiological images.

    anti_aliasing: bool, optional, default: true
        Determines whether to perform anti-aliasing, which is done to mitigate aliasing artifacts when downsampling.

    smoothing_beta: float, optional, default: 0.98
        Determines the smoothness of the Gaussian filter used for anti-aliasing. A value of 1.00 equates to no
        anti-aliasing, with lower values producing increasingly smooth imaging. Values above 0.90 are recommended.

        perturbation_noise_repetitions: int, optional, default: 0
        Number of repetitions where noise is randomly added to the image. A value of 0 means that no noise will be
        added.

    perturbation_noise_level: float, optional, default: None
        Set the noise level in intensity units. This determines the width of the normal distribution used to generate
        random noise. If None (default), noise is determined from the image itself.

    perturbation_rotation_angles: float or list of float, optional, default: 0.0
        Angles (in degrees) over which the image and mask are rotated. This rotation is only in the x-y (axial)
        plane. Multiple angles can be provided to create images with different rotations.

    perturbation_translation_fraction: float or list of float, optional, default: 0.0
        Sub-voxel translation distance fractions of the interpolation grid. This forces the interpolation grid to
        shift slightly and interpolate at different points. Multiple values can be provided. All values should be
        between 0.0 and 1.0.

    perturbation_roi_adapt_type: {"fraction", "distance"}, optional, default: "distance"
        Determines how the mask is grown or shrunk. Can be either "fraction" or "distance". "fraction" is used to
        grow or shrink the mask by a certain fraction (see the ``perturbation_roi_adapt_size`` parameter).
        "distance" is used to grow or shrink the mask by a certain physical distance, defined using the
        ``perturbation_roi_adapt_size`` parameter.

    perturbation_roi_adapt_size: float or list of float, optional, default: 0.0
        Determines the extent of growth/shrinkage of the ROI mask. The use of this parameter depends on the
        growth/shrinkage type (``perturbation_roi_adapt_type``), For "distance", this parameter defines
        growth/shrinkage in physical units, typically mm. For "fraction", this parameter defines growth/shrinkage in
        volume fraction (e.g. a value of 0.2 grows the mask by 20%). For either type, positive values indicate growing
        the mask, whereas negative values indicate its shrinkage. Multiple values can be provided to perturb the
        volume of the mask.

    perturbation_roi_adapt_max_erosion: float, optional, default: 0.8
        Limits shrinkage of the mask by distance-based adaptations to avoid forming empty masks. Defined as fraction of
        the original volume, e.g. a value of 0.8 prevents shrinking the mask below 80% of its original volume. Only
        used when ``perturbation_roi_adapt_type=="distance"``.

    perturbation_randomise_roi_repetitions: int, optional, default: 0.0
        Number of repetitions where the mask is randomised using supervoxel-based randomisation.

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

    if export_images:
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
        export_images: bool = True,
        write_images: bool = False,
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

    image_export_format: {"dict", "native", "numpy"}, default: "numpy"
        Return format for processed images and masks. ``"dict"`` returns dictionaries of images and masks as numpy
        arrays and associated characteristics. ``"native"`` returns images and masks in their internal format.
        ``"numpy"`` returns images and masks in numpy format. This argument is only used if ``export_images=True``.

    write_file_format: {"nifti", "numpy"}, default: "numpy"
        File format for processed images and masks. ``"nifti"`` writes images and masks in the NIfTI file format,
        and ``"numpy"`` writes images and masks as numpy files. This argument is only used if ``write_images=True``.

    export_images: bool, default: True
        Determines whether processed images and masks should be returned by the function.

    write_images: bool, default: False
        Determines whether processed images and masks should be written to the directory indicated by the
        ``write_dir`` keyword argument.

    **kwargs:
        Keyword arguments passed for importing images and masks (
        :func:`mirp.importData.importImageAndMask.import_image_and_mask`) and configuring settings (notably
        :class:`mirp.settings.settingsImageProcessing.ImagePostProcessingClass`,
        :class:`mirp.settings.settingsPerturbation.ImagePerturbationSettingsClass`), among others. See also the
        `Other Parameters` section below.

    Yields
    -------
    None | list[Any]
        List of images and masks in the format indicated by ``image_export_format``, if ``export_images=True``.

    Other Parameters
    ----------------
    .. note::
        The parameters below can be provided as keyword arguments.

    write_dir: str, optional
        Path to directory where processed images and masks should be written.

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

    intensity_normalisation: {"none", "range", "relative_range", "quantile_range", "standardisation"}, default: "none"
        Specifies the algorithm used to normalise intensities in the image. Will use only intensities in voxels
        masked by the tissue mask (of present). The following are possible:

        * "none": no normalisation
        * "range": normalises intensities based on a fixed mapping against the ``intensity_normalisation_range``
          parameter, which is interpreted to represent an intensity range.
        * "relative_range": normalises intensities based on a fixed mapping against the ``intensity_normalisation_range``
          parameter, which is interpreted to represent a relative intensity range.
        * "quantile_range": normalises intensities based on a fixed mapping against the
          ``intensity_normalisation_range`` parameter, which is interpreted to represent a quantile range.
        * "standardisation": normalises intensities by subtraction of the mean intensity and division by the standard
          deviation of intensities.

    .. note::
        intensity normalisation may remove any physical meaning of intensity units.

    intensity_normalisation_range: list of float, optional
        Required for "range", "relative_range", and "quantile_range" intensity normalisation methods, and defines the
        intensities that are mapped to the [0.0, 1.0] range during normalisation. The default range depends on the
        type of normalisation method:

        * "range": [np.nan, np.nan]: the minimum and maximum intensity value present in the image are used to set the
          mapping range.
        * "relative_range": [0.0. 1.0]: the minimum (0.0) and maximum (1.0) intensity value present in the image are
          used to set the mapping range.
        * "quantile_range": [0.025, 0.975] the 2.5th and 97.5th percentiles of the intensities in the image are used
          to set the mapping range.

        The lower end of the range is mapped to 0.0 and the upper end to 1.0. However, if intensities below the lower
        end or above the upper end are present in the image, values below 0.0 or above 1.0 may be encountered after
        normalisation. Use ``intensity_normalisation_saturation`` to cap intensities after normalisation to a
        specific range.

    intensity_normalisation_saturation: list of float, optional, default: [np.nan, np.nan]
        Defines the start and endpoint for the saturation range. Normalised intensities that lie outside this
        range are mapped to the limits of the saturation range, e.g. with a range of [0.0, 0.8] all values greater
        than 0.8 are assigned a value of 0.8. np.nan can be used to define limits where the intensity values should
        not be saturated.

    tissue_mask_type: {"none", "range", "relative_range"}, optional, default: "relative_range"
        Type of algorithm used to produce an approximate tissue mask of the tissue. Such masks can be used to select
         pixels for bias correction and intensity normalisation by excluding non-tissue voxels.

    tissue_mask_range: list of float, optional
        Range values for creating an approximate mask of the tissue. Required for "range" and "relative_range"
        options. Default: [0.02, 1.00] (``"relative_range"``); [np.nan, np.nan] (``"range"``; effectively all voxels
        are considered to represent tissue).

        interpolate: bool, optional, default: False
        Controls whether interpolation of images to a common grid is performed at all.

    spline_order: int, optional, default: 3
        Sets the spline order used for spline interpolation. mirp uses `scipy.ndimage.map_coordinates
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage
        .map_coordinates>`_ internally. Spline orders 0, 1, and 3 refer to nearest neighbour, linear interpolation
        and cubic interpolation, respectively.

    new_spacing: float or list of float or list of list of float:
        Sets voxel spacing after interpolation. A single value represents the spacing that will be applied in all
        directions. Non-uniform voxel spacing may also be provided, but requires 3 values for z, y, and x directions
        (if `by_slice = False`) or 2 values for y and x directions (otherwise).

        Multiple spacings may be defined by creating a nested list, e.g. [[1.0], [1.5], [2.0]] to resample the
        same image multiple times to different (here: isotropic) voxel spacings, namely 1.0, 1.5 and 2.0. Units
        are defined by the headers of the image files. These are typically millimeters for radiological images.

    anti_aliasing: bool, optional, default: true
        Determines whether to perform anti-aliasing, which is done to mitigate aliasing artifacts when downsampling.

    smoothing_beta: float, optional, default: 0.98
        Determines the smoothness of the Gaussian filter used for anti-aliasing. A value of 1.00 equates to no
        anti-aliasing, with lower values producing increasingly smooth imaging. Values above 0.90 are recommended.

        perturbation_noise_repetitions: int, optional, default: 0
        Number of repetitions where noise is randomly added to the image. A value of 0 means that no noise will be
        added.

    perturbation_noise_level: float, optional, default: None
        Set the noise level in intensity units. This determines the width of the normal distribution used to generate
        random noise. If None (default), noise is determined from the image itself.

    perturbation_rotation_angles: float or list of float, optional, default: 0.0
        Angles (in degrees) over which the image and mask are rotated. This rotation is only in the x-y (axial)
        plane. Multiple angles can be provided to create images with different rotations.

    perturbation_translation_fraction: float or list of float, optional, default: 0.0
        Sub-voxel translation distance fractions of the interpolation grid. This forces the interpolation grid to
        shift slightly and interpolate at different points. Multiple values can be provided. All values should be
        between 0.0 and 1.0.

    perturbation_roi_adapt_type: {"fraction", "distance"}, optional, default: "distance"
        Determines how the mask is grown or shrunk. Can be either "fraction" or "distance". "fraction" is used to
        grow or shrink the mask by a certain fraction (see the ``perturbation_roi_adapt_size`` parameter).
        "distance" is used to grow or shrink the mask by a certain physical distance, defined using the
        ``perturbation_roi_adapt_size`` parameter.

    perturbation_roi_adapt_size: float or list of float, optional, default: 0.0
        Determines the extent of growth/shrinkage of the ROI mask. The use of this parameter depends on the
        growth/shrinkage type (``perturbation_roi_adapt_type``), For "distance", this parameter defines
        growth/shrinkage in physical units, typically mm. For "fraction", this parameter defines growth/shrinkage in
        volume fraction (e.g. a value of 0.2 grows the mask by 20%). For either type, positive values indicate growing
        the mask, whereas negative values indicate its shrinkage. Multiple values can be provided to perturb the
        volume of the mask.

    perturbation_roi_adapt_max_erosion: float, optional, default: 0.8
        Limits shrinkage of the mask by distance-based adaptations to avoid forming empty masks. Defined as fraction of
        the original volume, e.g. a value of 0.8 prevents shrinking the mask below 80% of its original volume. Only
        used when ``perturbation_roi_adapt_type=="distance"``.

    perturbation_randomise_roi_repetitions: int, optional, default: 0.0
        Number of repetitions where the mask is randomised using supervoxel-based randomisation.

    """
    workflows = list(_base_deep_learning_preprocessing(
        export_images=export_images,
        write_images=write_images,
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
        write_images: bool = False,
        export_images: bool = True,
        write_dir: None | str = None,
        **kwargs
):
    from mirp.importData.importImageAndMask import import_image_and_mask
    from mirp.settings.importConfigurationSettings import import_configuration_settings

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
        raise TypeError(f"The 'settings' argument is expected to be a path to a configuration xml file, "
                        f"a SettingsClass object, or a list thereof. Found: {type(settings)}.")

    if not write_images:
        write_dir = None

    if write_images and write_dir is None:
        raise ValueError("write_dir argument should be provided for writing images and masks to.")

    if not write_images and not export_images:
        raise ValueError(f"write_images and export_images arguments cannot both be False.")

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
