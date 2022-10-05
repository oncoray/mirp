# Version 1.3.0

## Major changes

## Minor changes
- `SimpleITK` has been removed as a dependency. Handling of non-DICOM imaging is now done through `itk` itself.

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