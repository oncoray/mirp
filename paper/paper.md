---
title: 'MIRP: A Python package for standardised radiomics'
tags:
  - Python
  - radiomics
  - medical imaging
authors:
  - name: Alex Zwanenburg
    orcid: 0000-0002-0342-9545
    affiliation: "1, 2"
  - name: Steffen Löck
    orcid: 0000-0002-0342-9545
    affiliation: "2, 3, 4"
affiliations:
  - name: "National Center for Tumor Diseases Dresden (NCT/UCC), Germany: German Cancer Research Center (DKFZ), Heidelberg, Germany; Faculty of Medicine and University Hospital Carl Gustav Carus, TUD Dresden University of Technology, Dresden, Germany; Helmholtz-Zentrum Dresden-Rossendorf (HZDR), Dresden, Germany"
    index: 1
  - name: OncoRay – National Center for Radiation Research in Oncology, Faculty of Medicine and University Hospital Carl
      Gustav Carus, TUD Dresden University of Technology, Helmholtz-Zentrum Dresden-Rossendorf, Dresden, Germany
    index: 2
  - name: German Cancer Consortium (DKTK), Partner Site Dresden, and German Cancer Research Center (DKFZ), Heidelberg, Germany
    index: 3
  - name: Department of Radiotherapy and Radiation Oncology, Faculty of Medicine and University Hospital Carl Gustav Carus, TUD Dresden University of Technology, Dresden, Germany
    index: 4
date: 7 February 2024
bibliography: paper.bib
---

# Summary

Medical imaging provides non-invasive anatomical and functional visualisation of the human body. It is used 
clinically for diagnostic, prognostic, treatment planning and other purposes. Many current uses of medical imaging 
involve qualitative or semi-quantitive assessment by experts. Radiomics seeks to automate analysis of medical imaging for 
clinical decision support. At its core, radiomics involves the extraction and machine learning-based analysis of 
quantitive features from medical images [@Lambin2017-tf]. However, very few--if any--radiomics tools have been translated to the 
clinic [@Huang2022-mi]. One of the essential prerequisites for translation is reproducibility and validation in 
external settings [@OConnor2017-iv]. This can be facilitated through the use of standardised radiomics software. 
Here we present `mirp`, a Python package for standardised processing of medical imaging and computation of 
quantitative features. Researchers can use `mirp` for their own radiomics analyses or to reproduce and validate 
radiomics analyses of others.

# Statement of need

Lack of standardised radiomics software is one of the reasons for poor translation of radiomics tools to the clinic.
The Image Biomarker Standardisation Initiative (IBSI) has created reference standards for radiomics software: 1. a 
reference standard for basic image processing and feature extraction steps [@Zwanenburg2020-go]; and 2. a reference 
standard for image filters [@Whybra2024-yb]. There is currently a lack of fully IBSI-compliant radiomics 
packages in Python, which is important for the radiomics field because Python is used to interface with machine learning
and deep learning packages such as `scikit-learn` and `pytorch`. `mirp` is intended to fill this gap. It provides a 
user-friendly API for standardised image processing for deep learning-based radiomics as well as standardised 
image processing and feature extraction for machine learning-based radiomics.

`mirp` is intended to be used by researchers in the radiomics field to perform their own radiomics analyses on the 
one hand, and to externally reproduce and validate results of other researchers. It was originally created in 2016 and 
regularly updated to conform with the IBSI reference standards and to improve usability. Previous versions of `mirp`
were used by e.g. @Leger2017-si, @Zwanenburg2019-jg, @Shahzadi2022-wk and @Bettinelli2022-ml. Recently, `mirp` was used 
as a radiomics engine for the `RadTract` Python package [@Neher2024-ze]. The latest major version (2) provides major
improvements in user experience with a unified API, better documentation, and easier installation using `pip`.

`mirp` follows an end-to-end design principle and abstracts away intermediate steps for the user. In this sense it 
is not a toolkit such as `scikit-image` or `opencv`, that offer a set of functions for performing different operations
on images. Because of the end-to-end design, relatively few functions are exposed to the user:

- `mirp.deep_learning_preprocessing`: For reading and processing images as input for deep learning networks.
- `mirp.extract_features`: For reading and processing images, and computing radiomics features as input for machine 
  learning algorithms.
- `mirp.extract_images`: For reading and processing images and exporting them.
- `mirp.extract_features_and_images`: For reading and processing images, computing radiomics features and 
  simultaneous export of both processed images and quantitative features computed from them.
- `mirp.extract_image_parameters`: For reading images and extracting their relevant metadata.
- `mirp.extract_mask_labels`: For reading masks and extracting their labels.

Of the above, `mirp.deep_learning_preprocessing`, `mirp.extract_features`, `mirp.extract_images` and 
`mirp.extract_features_and_images` allow for parallel processing using the `ray` package. These functions also have 
generator-based variants that yield output one-by-one. 

`mirp` supports standard medical imaging formats as input, notably DICOM, NIfTI and NRRD. It also supports `numpy` 
arrays as a generic fallback option, and has specific routines for handling computed tomography (CT), positron 
emission tomography (PET), magnetic resonance imaging (MRI) and radiotherapy dose maps. It supports both 
radiotherapy structure sets (RTSTRUCT) and segmentation (SEG) files as DICOM-based masks.

In conclusion, `mirp` offers a much-needed solution for standardized radiomics. With its user-friendly Python 
interface, researchers can conduct radiomics analyses and, crucially, reproduce and validate the work of others, 
bringing us one step closer to harnessing the full potential of medical imaging in improving patient care.

# Alternatives

`mirp` is not the only package available for image processing and feature extraction for radiomics analyses. Commonly 
used alternatives are listed in Table 1.

|                          | `mirp`                                    | `pyradiomics`                                        | `CERR`                                 | `LIFEx`                               | `radiomics`                                       |
|--------------------------|-------------------------------------------|------------------------------------------------------|----------------------------------------|---------------------------------------|---------------------------------------------------| 
| Version                  | 2.1.0                                     | 3.1.0                                                | unknown                                | 7.4.0                                 | unknown                                           |
| Last updated             | 2/2024                                    | 5/2023                                               | 2/2024                                 | 6/2023                                | 11/2019                                           |
| License                  | EUPL-1.2                                  | BSD-3                                                | LGPL-2.1                               | custom                                | GPL-3.0                                           |
| Programming language     | Python                                    | Python                                               | MATLAB                                 | Java                                  | MATLAB                                            |
| IBSI-1 compliant         | yes                                       | partial                                              | yes                                    | yes                                   | no claim                                          |
| IBSI-2 compliant         | yes                                       | no claim                                             | yes                                    | yes                                   | no claim                                          |
| Graphical user interface | no                                        | no                                                   | yes                                    | yes                                   | no                                                |
| Website                  | [GitHub](https://github.com/oncoray/mirp) | [GitHub](https://github.com/AIM-Harvard/pyradiomics) | [GitHub](https://github.com/cerr/CERR) | [website](https://www.lifexsoft.org/) | [GitHub](https://github.com/mvallieres/radiomics) | 
Table 1: Comparison of `mirp` with other popular alternatives. Note that compliance with the first and second 
set of reference standards of the Image Biomarker Standardisation Initiative (IBSI-1 and IBSI-2, respectively) is 
based on claims of the developers, and not verified by the authors.

# Acknowledgements

We acknowledge contributions from Stefan Leger and Sebastian Starke. Development of the initial version of `mirp` was 
financially supported by the Federal Ministry of Education and Research (BMBF-0371N52).

# References