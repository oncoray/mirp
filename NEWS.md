# Version 2.3.0

## Major changes

- The proper ancient feature computation code running in the background of MIRP has been completely refactored. We 
  moved from a functional backend where all features were computed per feature family to a more flexible 
  object-oriented approach. Although this change is not visible at the user-end, it offers several new possibilities:
  - Single features can now be computed. In addition, for some features (e.g. percentile statistics), a flexible 
    percentile value could be passed.
  - Creation of feature maps.
  - Output of features and their metadata to machine-readable formats, instead of just tabular data.
  
  **Important**: Though the *name* of features in the tabular exports has not changed, their *ordering* may have. 
  Avoid using column position when processing or analysing feature data.

- Apparent diffusion coefficient (ADC) maps and multi-frame DICOM objects in general are now supported.
- Planar imaging, i.e. computed radiography, digital X-ray and digital mammography DICOM files are now supported.
  
# Version 2.2.4

## Fixes

- Masks can now be plotted in images without causing an error when using `matplotlib` version 3.9.0 or later.

# Version 2.2.3

## Minor changes

- Tables with feature values now contain extra columns. These columns specify the file name (for non-DICOM input), 
  the directory path of the image and masks and several DICOM tags for identifying the input.

- MIRP now checks whether there are potential problems between the frames of reference of image and mask files.

## Fixes

- Fixed an error that occurs when attempting to create a deep copy `ImageITKFile` objects.

# Version 2.2.2

## Minor changes

- `show` method of `GenericImage` and subclasses now indicate if a user-provided `slice_id` is out-of-volume and 
  select the nearest slice instead.

- Naming of branches in the settings `xml` file now matches that of their respective settings classes. `xml` files 
  with the previous branch names still function.

- Errors encountered during file import and handling are now more descriptive.
- `extract_mask_labels` and `extract_image_parameters` now export extra information from DICOM metadata, e.g. series 
  UID.

## Documentation

- Added a new tutorial on applying image filters to images.
- Added documentation on the feature naming system.
- Added documentation on the design of MIRP.

## Fixes

- Computing features related to the minimum volume enclosing ellipsoid no longer produces warnings due to the use of 
  deprecated `numpy.matrix` class.

# Version 2.2.1

## Minor changes

- If mask-related parameters are not provided for computing features or processing of images for deep learning, a 
  mask is generated that covers the entire image.

- Add fall-back methods for missing installation of the `ray` package for parallel processing. This can happen when 
  a python version is not supported by the `ray` package. `ray` is now a conditional dependency, until that package 
  is released for python `3.12`.

- The default export format for `deep_learning_processing` and `deep_learning_processing_generator` is now `dict`, 
  because the sample name is important for matching against observed outcomes.

- `write_file` arguments of `extract_mask_labels` and `extract_image_parameters` were deprecated as these were 
  redundant.

## Fixes

- Streamlined importing and reading DICOM files results in faster processing of DICOM-based imaging.

- Fixed an indexing issue when attempting to split masks into bulk and rim sections in a slice-wise fashion.

- Fixed an indexing issue in Rank's method for noise estimation.

- Fixed incorrectly named image parameters file export. Instead of `mask_labels.csv`, image parameters are now 
  correctly exported to `image_metadata.csv`.

# Version 2.2.0

## Major changes

- Added support for intensity scaling using the `intensity_scaling` parameter. Intensity scaling multiplies 
  intensities by a scalar value. Intensity scaling occurs after intensity normalisation (if any) and prior to adding 
  noise (if any). For example, intensity scaling can be used after intensity normalisation to scale intensities to a 
  different range. `intensity_normalisation = "range"` with `intensity_scaling = 1000.0` maps image intensities to 
  [1000.0, 0.0] instead of [1.0, 0.0].

- Added support for intensity transformation filters: square root (`"pyradiomics_square_root"`), square 
  (`"pyradiomics_square"`), logarithm (`"pyradiomics_logarithm"`) and exponential (`"pyradiomics_exponential"`). 
  These implementations are based on the definitions in the `pyradiomics` 
  [documentation](https://pyradiomics.readthedocs.io/en/latest/radiomics.html#module-radiomics.imageoperations). 
  Since these filters do not currently have an IBSI reference standard, these are mostly intended for reproducing 
  and validating radiomics models based on features extracted from pyradiomics.

- Modules were renamed according to the PEP8 standard. This does not affect the documented public interface, but may 
  affect external extensions. Public and private parts of the API are now indicated. 

## Minor changes

- Added support for Python version 3.10 using `typing-extensions`.
- Several changes were made to ensure proper functioning of MIRP with future versions of `pandas`.
- Some changes were made prevent deprecation warnings in future version of `numpy`.

# Version 2.1.1

## Fixes

- Fixed missing merge changes from version 2.1.0 to the main branch.
- Fixed reading of `mask_name` from data xml files.
- `image_name` and `mask_name` configuration parameters are now parsed as single strings if only one value is 
  specified to match argument-based configuration.
- Fixed and updated several exception messages.
- Filter kernel names, specified using `filter_kernels` in xml files, are now correctly parsed as strings instead of 
  floats.

# Version 2.1.0

## Major changes

- Added support for SEG DICOM files for segmentation.

- Added support for processing RTDOSE files.

- It is now possible to combine and split masks, and to select the largest mask or mask slice, as part of the image
  processing workflow. Masks can be combines by setting `mask_merge = True`, which merges all available masks for an
  image into a single mask. This can be useful when, e.g., multiple regions of interest should be assessed as a single
  (possibly internally disconnected) mask. Masks are split using `mask_split = True`, which separates every disconnected
  region into its own mask that is assessed separately. This is used for splitting multiple lesions inside a single mask
  into multiple separate masks. The largest region of interest in each mask is selected by 
  `mask_select_largest_region = True`. This can be used when, e.g., only the largest lesion of multiple lesions should be
  assessed. Sometimes, only the largest slice (i.e. the slice containing most of the voxels in a mask) should be
  assessed. This is done using `mask_select_largest_slice = True`. This also forces `by_slice = True`.

  These mask operations are implemented in the following order: combination -> splitting -> largest region -> 
  largest slice.

- Masks from an RT-structure file that shares a frame of reference with an image but does not have a one-to-one 
  mapping to its voxel space can now be processed. This facilitates processing of masks from RT structure sets that 
  are, e.g., defined on CT images but applied to co-registered PET imaging, or from one MR sequence to another. 

## Fixes

- Providing a mask consisting of boolean values in a numpy array no longer incorrectly throws an error.
- Configuration parameters from `xml` files are now processed in the same manner as parameters defined as function 
  arguments. The same default values are now used, independent of the parameter source. This fixes a known issue where
  outlier-based resegmentation would occur by default using `xml` files, whereas the intended default is that no
  resegmentation takes place.
- Masks can now be exported to the file system without throwing an error.
- DICOM files from frontal or sagittal view data are now correctly processed.

# Version 2.0.1

## Minor changes

- Randomisation in MIRP now uses the generator-based methods in `numpy.random`, replacing the legacy functions.
  The generator is seeded so that results are reproducible. The seed depends on input image, mask and configuration
  parameters, if applicable. 

## Fixes

- Numpy arrays can now be used as direct input without throwing a `FileNotFoundError`.
- Relaxed check on orientation matrix when importing images, preventing errors when the l2-norm is around 1.000 but not
  to high precision.
- To prevent high loads through internal multithreading in `numpy` and other libraries when using `ray` for parallel
  processing, each ray thread is now initialised with environment parameters that prevent multi-threading.

# Version 2.0.0

## Major changes

- MIRP was previously configured using two `xml` files: [`config_data.xml`](mirp/config_data.xml) for configuring
  directories, data to be read, etc., and [`config_settings.xml`](mirp/config_settings.xml) for configuring experiments.
  While these two files can still be used, MIRP can now be configured directly, without using these files.

- The main functions of MIRP (`mainFunctions.py`) have all been re-implemented.
  - `mainFunctions.extract_features` is now `extractFeaturesAndImages.extract_features` (functional form) or
    `extractFeaturesAndImages.extract_features_generator` (generator). The replacements allow for both writing
    feature values to a directory and returning them as function output. 
  - `mainFunctions.extract_images_to_nifti` is now `extractFeaturesAndImages.extract_images` (functional form) or
     `extractFeaturesAndImages.extract_images_generator` (generator). The replacements allow for both writing 
     images to a directory (e.g., in NIfTI or numpy format) and returning them as function output.
  - `mainFunctions.extract_images_for_deep_learning` has been replaced by 
    `deepLearningPreprocessing.deep_learning_preprocessing` (functional form) and 
    `deepLearningPreprocessing.deep_learning_preprocessing_generator` (generator).
  - `mainFunctions.get_file_structure_parameters` and `mainFunctions.parse_file_structure` are deprecated, as the
    the file import system used in version 2 no longer requires a rigid directory structure.
  - `mainFunctions.get_roi_labels` is now `extractMaskLabels.extract_mask_labels`.
  - `mainFunctions.get_image_acquisition_parameters` is now `extractImageParameters.extract_image_parameters`.

- MIRP previously relied on `ImageClass` and `RoiClass` objects. These have been completely replaced by `GenericImage`
  (and its subclasses, e.g. `CTImage`) and `BaseMask` objects, respectively. New image modalities can be added as
  subclass of `GenericImage` in the `mirp.images` submodule.

- File import, e.g. from DICOM or NIfTI files, in was previously implemented in an ad-hoc manner, and required a rigid
  directory structure. Now, file import is implemented using an object-oriented approach, and directory structures 
  are more flexible. File import of new modalities can be implemented as a relevant subclass of `ImageFile`.

- MIRP uses type hinting, and makes use of the `Self` type hint introduced in Python 3.11. MIRP therefore requires 
  Python 3.11 or later.

## Minor changes
- MIRP now uses the `ray` package for parallel processing.

# Version 1.3.0

## Minor changes
- `SimpleITK` has been removed as a dependency. Handling of non-DICOM imaging is now done through `itk` itself.
- Rotation - as a perturbation or augmentation operation - is now performed as part of the interpolation process. 
  Previously, rotation was implemented using `scipy.ndimage.rotate`. This, combined with any translation or 
  interpolation operation would involve two interpolation steps. Aside from removing a computationally intensive 
  step, this also prevents unnecessary image degradation through the interpolation process. The new implementation 
  operates using affine matrix transformations.
- Discretisation of intensities after filtering (i.e. intensities of response maps) now uses a *fixed bin number* 
  method with 16 bins by default. Previously, no default was set, which could lead to unintended results. These 
  parameters can be manually specified using the `response_map_discretisation_method`, 
  `response_map_discretisation_bin_width`, and `response_map_discretisation_n_bins` arguments; or alternatively 
  using the `discretisation_method`, `discretisation_bin_width` and `discretisation_n_bins` parameters of the 
  `img_transform` section of the settings configuration file. 

## Fixes

- Fixed a deprecation warning caused by `slic` of the `scikit-image` module.
- Fixed incorrect merging of contours of the same region of interest (ROI) in the same slice. Previously, each contour 
  was converted to a mask individually, and merged with the segmentation mask using `OR` operations. This functions 
  perfectly for contours that represent separate objects spatially. However, holes in RTSTRUCT objects are not 
  always represented by a single contour. They can also be represented by a separate contour (of the same region of 
  interest) that is contained within a larger contour. For those RTSTRUCT objects, holes would disappear. This has 
  now been fixed by first collecting all contours of a ROI for each slice, prior to converted them to a segmentation 
  mask.

# Version 1.2.0

## Major changes
- Updated filter implementations to the current (August 2022) IBSI 2 guidelines.
- Settings read from the configuration files are now parsed and checked prior to starting computations. This is a 
  preliminary to command-line configuration of experiments in future versions. Several `xml` tags were renamed or 
  deprecated. Most renamed tags are soft-deprecated, and support backward compatibility. The following tags will now 
  throw deprecation warnings:
  - `new_non_iso_spacing` has been deprecated. Non-isotropic spacing can be set using the existing `new_spacing` 
    argument.
  - `glcm_merge_method` has been deprecated and merged into `glcm_spatial_method`.
  - `glrlm_merge_method` has likewise been deprecated and merged into `glrlm_spatial_method`.
  - `log_average` has been deprecated. The same effect can be achieved by giving the 
    `laplacian_of_gaussian_pooling_method` the value `mean`.

## Minor changes
- It is now possible to compute features for multiple images for the same subject and modality.

## Fixes
- White-space is properly stripped from the names of regions of interest.
- Several issues related to one-voxel ROI were resolved.
- Computing no features or features that do not require discretisation do no longer prompt providing for a 
  discretisation method.
- Computing no features from, e.g., the base image no longer generate errors.
- Fixed an issue where rotated masks were not returned correctly.
- A number of other fixes were made to improve stability.

# Version 1.1

## Major changes
- The `extract_images_for_deep_learning` and underlying functions have been reworked.
- The `deep_learning` section of the settings configuration xml file have been deprecated in favour of function 
  arguments.