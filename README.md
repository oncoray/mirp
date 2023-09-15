<img src="icon/mirp.svg" align="right" width="120"/>

# Medical Image Radiomics Processor

The Medical Image Radiomics Processor (MIRP) is an IBSI-compliant python package for medical image analysis.
MIRP focuses on radiomics applications and supports computation of features for conventional radiomics
and image processing for deep-learning applications.

## Installing MIRP
...

## Transitioning to version 2

Version 2 is a major refactoring of the previous code base. For users this brings the following noticeable changes:

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

For advanced users and developers, the following changes are relevant:
- MIRP previously relied on `ImageClass` and `RoiClass` objects. These have been completely replaced by `GenericImage`
  (and its subclasses, e.g. `CTImage`) and `BaseMask` objects, respectively. New image modalities can be added as
  subclass of `GenericImage` in the `mirp.images` submodule.
- File import, e.g. from DICOM or NIfTI files, in version 1 was implemented in an ad-hoc manner, and required a rigid
  directory structure. Since version 2, file import is implemented using an object-oriented approach, and directory
  structures are more flexible. File import of new modalities can be implemented as a relevant subclass of `ImageFile`.
- MIRP uses type hinting, and makes use of the `Self` type hint introduced in Python 3.11. MIRP 
  therefore requires Python 3.11 or later.

## Examples

# Citation info
If you use MIRP, please cite the following work:
```Zwanenburg A, Leger S, Agolli L, Pilz K, Troost EG, Richter C, Löck S. Assessing robustness of radiomic features by image perturbation. Scientific reports. 2019 Jan 24;9(1):614.```

# Developers and contributors

MIRP is developed by:
* Alex Zwanenburg

We would like thank the following contributors:
* Stefan Leger
* Sebastian Starke
