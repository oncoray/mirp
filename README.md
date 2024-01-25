<img src="https://raw.githubusercontent.com/oncoray/mirp/master/icon/mirp.svg" align="right" width="120"/>

# Medical Image Radiomics Processor

Medical Image Radiomics Processor (MIRP) is an IBSI-compliant python package for medical image analysis.
MIRP focuses on radiomics applications and supports computation of features for conventional radiomics
and image processing for deep-learning applications.

## Documentation

Documentation can be found here: https://oncoray.github.io/mirp/

## Installing MIRP
MIRP is available from PyPI and can be installed using `pip`, or other installer tools:

```commandline
pip install mirp
```

## Transitioning to version 2

Version 2 is a major refactoring of the previous code base. For users this brings the following noticeable changes:

- MIRP was previously configured using two `xml` files: [`config_data.xml`](mirp/config_data.xml) for configuring
  directories, data to be read, etc., and [`config_settings.xml`](mirp/config_settings.xml) for configuring experiments.
  While these two files can still be used, MIRP can now be configured directly, without using these files.
- The main functions of MIRP (`mainFunctions.py`) have all been re-implemented.
  - `mainFunctions.extract_features` is now `extract_features` (functional form) or
    `extract_features_generator` (generator). The replacements allow for both writing
    feature values to a directory and returning them as function output. 
  - `mainFunctions.extract_images_to_nifti` is now `extract_images` (functional form) or
     `extract_images_generator` (generator). The replacements allow for both writing 
     images to a directory (e.g., in NIfTI or numpy format) and returning them as function output.
  - `mainFunctions.extract_images_for_deep_learning` has been replaced by 
    `deep_learning_preprocessing` (functional form) and 
    `deep_learning_preprocessing_generator` (generator).
  - `mainFunctions.get_file_structure_parameters` and `mainFunctions.parse_file_structure` are deprecated, as the
    the file import system used in version 2 no longer requires a rigid directory structure.
  - `mainFunctions.get_roi_labels` is now `extract_mask_labels`.
  - `mainFunctions.get_image_acquisition_parameters` is now `extract_image_parameters`.

For advanced users and developers, the following changes are relevant:
- MIRP previously relied on `ImageClass` and `RoiClass` objects. These have been completely replaced by `GenericImage`
  (and its subclasses, e.g. `CTImage`) and `BaseMask` objects, respectively. New image modalities can be added as
  subclass of `GenericImage` in the `mirp.images` submodule.
- File import, e.g. from DICOM or NIfTI files, in version 1 was implemented in an ad-hoc manner, and required a rigid
  directory structure. Since version 2, file import is implemented using an object-oriented approach, and directory
  structures are more flexible. File import of new modalities can be implemented as a relevant subclass of `ImageFile`.
- MIRP uses type hinting, and makes use of the `Self` type hint introduced in Python 3.11. MIRP 
  therefore requires Python 3.11 or later.
- MIRP now uses the `ray` package for parallel processing.

## Examples - Computing Radiomics Features

MIRP can be used to compute quantitative features from regions of interest in images in an IBSI-compliant manner 
using a standardized workflow This requires both images and masks. MIRP can process DICOM, NIfTI, NRRD and numpy 
images. Masks are DICOM radiotherapy structure sets (RTSTRUCT), or volumetric data with integer labels (e.g. 1, 2, 
etc.).

Below is a minimal working example for extracting features from a single image file and its mask.

```python
from mirp import extract_features

feature_data = extract_features(
    image="path to image",
    mask="path to mask",
    base_discretisation_method="fixed_bin_number",
    base_discretisation_n_bins=32
)
```
Instead of providing the path to the image (`"path_to_image"`), a numpy image can be provided, and the same goes for 
`"path to mask"`. The disadvantage of doing so is that voxel spacing cannot be determined. 

MIRP also supports processing images and masks for multiple samples (e.g., patients). The syntax is much the same, 
but depending on the file type and directory structure, additional arguments need to be specified. For example, 
assume that files are organised in subfolders for each sample, i.e. `main_folder / sample_name / subfolder`. The 
minimal working example is then:

```python
from mirp import extract_features

feature_data = extract_features(
    image="path to main image directory",
    mask="path to main mask directory",
    image_sub_folder="image subdirectory structure relative to main image directory",
    mask_sub_folder="mask subdirectory structure relative to main mask directory",
    base_discretisation_method="fixed_bin_number",
    base_discretisation_n_bins=32
)
```
The above example will compute features sequentially. MIRP supports parallel processing using the `ray` package. 
Feature computation can be parallelized by specifying the `num_cpus` argument, e.g. `num_cpus=2` for two CPU threads.

## Examples - Image Preprocessing for Deep Learning
Deep learning-based radiomics is an alternative to using predefined quantitative features. MIRP supports 
preprocessing of images and masks using the same standardized workflow that is used for computing features.

Below is a minimal working example for preprocessing deep learning images. Note that MIRP uses the numpy notation 
for indexing, i.e. indices are ordered [*z*, *y*, *x*].

```python
from mirp import deep_learning_preprocessing

processed_images = deep_learning_preprocessing(
    image="path to image",
    mask="path to mask",
    crop_size=[50, 224, 224]
)
```

## Examples - Summarising Image Metadata

MIRP can also summarise image metadata. This is particularly relevant for DICOM files that have considerable 
metadata. Other files, e.g. NIfTI, only have metadata related to position and spacing of the image.

Below is a minimal working example for extracting metadata from a single image file.
```python
from mirp import extract_image_parameters

image_parameters = extract_image_parameters(
    image="path to image"
)
```

MIRP also supports extracting metadata from multiple files. For example, assume that files are organised in 
subfolders for each sample, i.e. `main_folder / sample_name / subfolder`. The minimal working example is then:
```python
from mirp import extract_image_parameters

image_parameters = extract_image_parameters(
    image="path to main image directory",
    image_sub_folder="image subdirectory structure relative to main image directory"
)
```

## Examples - Finding labels

MIRP can identify which labels are present in masks. For a single mask file, labels can be retrieved as follows:
```python
from mirp import extract_mask_labels

mask_labels = extract_mask_labels(
    mask="path to mask"
)
```

MIRP supports extracting labels from multiple masks. For example, assume that files are organised in subfolders for 
each sample, i.e. `main_folder / sample_name / subfolder`. The minimal working example is then:
```python
from mirp import extract_mask_labels
mask_labels = extract_mask_labels(
    mask="path to main mask directory",
    mask_sub_folder="mask subdirectory structure relative to main mask directory"
)
```

# Citation info
A publication for MIRP is forthcoming. For now, please cite the following work:
```Zwanenburg A, Leger S, Agolli L, Pilz K, Troost EG, Richter C, Löck S. Assessing robustness of radiomic features by image perturbation. Scientific reports. 2019 Jan 24;9(1):614.```

# Developers and contributors

MIRP is developed by:
* Alex Zwanenburg

We would like thank the following contributors:
* Stefan Leger
* Sebastian Starke
