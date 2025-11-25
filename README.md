<img src="https://raw.githubusercontent.com/oncoray/mirp/master/icon/mirp.svg" align="right" width="120"/>

![GitHub License](https://img.shields.io/github/license/oncoray/mirp)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mirp)
[![PyPI - Version](https://img.shields.io/pypi/v/mirp)](https://pypi.org/project/mirp/)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/oncoray/mirp/auto-test-dependencies_timed.yml)
[![JOSS](https://joss.theoj.org/papers/165c85b1ecad891550a21b12c8b2e577/status.svg)](https://joss.theoj.org/papers/165c85b1ecad891550a21b12c8b2e577)

# Medical Image Radiomics Processor

MIRP is a python package for quantitative analysis of medical images. It focuses on processing images for integration
with radiomics workflows. These workflows either use quantitative features computed using MIRP, or directly use MIRP
to process images as input for neural networks and other deep learning models.

MIRP offers the following main functionality:

- [Extract and collect metadata](https://oncoray.github.io/mirp/image_metadata.html) from medical images.
- [Find and collect labels or names](https://oncoray.github.io/mirp/mask_labels.html) of regions of interest from image 
  segmentations.
- [Compute quantitative features](https://oncoray.github.io/mirp/quantitative_image_analysis.html) from regions of interest in medical images.
- [Process images for deep learning](https://oncoray.github.io/mirp/deep_learning.html).

## Tutorials

We currently offer the following tutorials:

- [Computing quantitative features from MR images](https://oncoray.github.io/mirp/tutorial_compute_radiomics_features_mr.html)
- [Applying filters to images](https://oncoray.github.io/mirp/tutorial_apply_image_filter.html)

## Documentation

Documentation can be found here: https://oncoray.github.io/mirp/

## Supported Python and OS

MIRP currently supports the following Python versions and operating systems: 

| Python | Linux     | Win       | OSX       |
|--------|-----------|-----------|-----------|
| 3.10   | Supported | Supported | Supported |
| 3.11   | Supported | Supported | Supported |
| 3.12   | Supported | Supported | Supported |
| 3.13   | Supported | Supported | Supported |

## Supported imaging and mask types

MIRP currently supports the following image and mask types:

| Data format | Data type | Supported modality                              |
|-------------|-----------|-------------------------------------------------|
| DICOM       | image     | CT, MR (incl. ADC, DCE), PT, RTDOSE, CR, DX, MG |
| DICOM       | mask      | RTSTRUCT, SEG                                   |
| NIfTI       | any       | any                                             |
| NRRD        | any       | any                                             |
| numpy       | any       | any                                             |
| MIRP-native | any       | any                                             | 

NIfTI, NRRD, and numpy files support any kind of (single-channel) image. MIRP cannot process RGB or 4D images. 
MIRP-native images and masks can be produced by functions such as `extract_images`, and then used as input. 

## Installing MIRP
MIRP is available from PyPI and can be installed using `pip`, or other installer tools:

```commandline
pip install mirp
```

## Examples - Computing radiomics features

MIRP can be used to compute quantitative features from regions of interest in images in an IBSI-compliant manner 
using a standardized workflow This requires both images and masks. MIRP can process DICOM, NIfTI, NRRD and numpy 
images. Masks are DICOM radiotherapy structure sets (RTSTRUCT), DICOM segmentation (SEG) or volumetric data with 
integer labels (e.g. 1, 2, etc.).

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

## Examples - Image preprocessing for deep learning
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

## Examples - Summarising image metadata

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

## Examples - Using MIRP native images and mask

MIRP supports exporting images and masks in its native, internal format. This is specified using the 
`image_export_format="native"` argument, e.g. in `extract_images(.., image_export_format="native")` or 
`extract_features_and_images(..., image_export_format="native"`). The resulting images and masks can be used again
as input, e.g. `extract_features(image=native_images, masks=native_masks, ...)`, with `native_images` and 
`native_masks` being the images and masks in the native format, respectively.
  
This allows for external processing of the contents of images and masks, such as performing gamma corrections. The 
image and mask contents are retrieved using the `get_voxel_grid` method, and set using the `set_voxel_grid` method.
`set_voxel_grid` expects a `numpy.ndarray` of the same shape and type (`float` for images, `bool` for masks) as the 
original.

```python
from mirp import extract_images, extract_features

results = extract_images(
    image="path to image",
    mask="path to mask",
    image_export_format="native"
)

image = results[0][0][0]
mask = results[0][1][0]

# Obtain the numpy.ndarray.
voxel_grid = image.get_voxel_grid()

# Divide intensities by 2.
image.set_voxel_grid(voxel_grid=voxel_grid / 2.0)

features = extract_features(
    image=image,
    mask=mask,
    base_discretisation_method="fixed_bin_number",
    base_discretisation_n_bins=32
)[0]
```

# Compatibility

MIRP is compliant with the Image Biomarker Standardisation Initiative's reference standards for image processing and
feature  computation and for image filters. Compliance is checked automatically as part
of the software tests.

The reference standards do not cover all possible image pre-processing steps, features or image filters. There are three
pre-processing steps in MIRP that are used by default and that currently lack reference standards, and thus may be
implemented differently in other software packages, or be absent entirely. These are:

1. **Anti-aliasing**: Downsampling, i.e. resampling from a high-resolution scan to a lower-resolution image leads
to aliasing artifacts because the interpolation algorithms will primary use local information from the voxels directly
adjacent to each interpolation point. To counteract this, MIRP by default uses a Gaussian
anti-aliasing filter prior to resampling. This filters smooths local information, prevent anti-aliasing artifacts. To
turn anti-aliasing off set `anti_aliasing = False`.

2. **Use of tissue masks for intensity normalisation**: Intensity normalisation is a pre-processing step commonly
used for harmonising clinical MRI sequences. These methods use information from intensities in a scan. Generally,
information from voxels outside the studied patient (air voxels) are not relevant. By default, MIRP tries to exclude
such voxels by generating a tissue mask based on the intensity distribution, and use only information from voxels
related containing patient tissue for normalisation. To turn tissue masks off, set `tissue_mask_type = "none"`.

3. **Conversion of PET values to SUV**: If PET scans are provided in the DICOM format, by default MIRP will use
information contained in the DICOM headers to compute body-weight corrected standardised uptake values. To use raw PET
information, set `pet_suv_conversion = "none"`. Other types of standardised uptake value can be computed using other
options.

# Citation info
MIRP has been published in *Journal of Open Source Software*:
```Zwanenburg A, Löck S. MIRP: A Python package for standardised radiomics. J Open Source Softw. 2024;9: 6413. doi:10.21105/joss.06413```

# Contributing
If you have ideas for improving MIRP, please read the short [contribution guide](./CONTRIBUTING.md).

# Developers and contributors

MIRP is developed by:
* Alex Zwanenburg

We would like thank the following contributors:
* Stefan Leger
* Sebastian Starke
