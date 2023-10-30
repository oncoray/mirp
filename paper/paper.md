Instructions
------------

OSS welcomes submissions from broadly diverse research areas. For this reason, we require that authors include in the paper some sentences that explain the software functionality and domain of use to a non-specialist reader. We also require that authors explain the research applications of the software. The paper should be between 250-1000 words. Authors submitting papers significantly longer than 1000 words may be asked to reduce the length of their paper.

Your paper should include:

- [x] A list of the authors of the software and their affiliations, using the correct format (see the example below). 
- [ ] A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience. 
- [ ] A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. 
- [ ] A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline. 
- [ ] Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. 
- [ ] Acknowledgement of any financial support.

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
 - name: National Center for Tumor Diseases (NCT/UCC) Dresden, Germany: German Cancer Research Center (DKFZ), Heidelberg, Germany; Faculty of Medicine and University Hospital Carl Gustav Carus, Technische Universität Dresden, Dresden, Germany, and; Helmholtz Association/Helmholtz-Zentrum Dresden–Rossendorf (HZDR), Dresden, Germany
   index: 1
 - name: OncoRay—National Center for Radiation Research in Oncology, Faculty of Medicine and University Hospital Carl Gustav Carus, Technische Universität Dresden, Helmholtz-Zentrum Dresden–Rossendorf, Dresden, Germany
   index: 2
 - name: German Cancer Research Center (DKFZ), Heidelberg and German Cancer Consortium (DKTK) Partner Site Dresden, Dresden, Germany
   index: 3
 - name: Department of Radiotherapy and Radiation Oncology, Faculty of Medicine and University Hospital Carl Gustav Carus, Technische Universität Dresden, Dresden, Germany
   index: 4
date: 30 November 2023
bibliography: paper.bib
---

# Summary

Medical imaging provides non-invasive anatomical and functional visualisation of the human body.
It is used for diagnostics, prognostics and treatment planning.
Many current uses of medical imaging involve qualitative or semi-quantitive assessment by experts.
Radiomics seeks to automate analysis of medical imaging for clinical decision support.
At its core, radiomics involves the extraction and machine learning-based analysis of quantitive features from medical images.
However, very few--if any--radiomics tools have been translated to the clinic.
One of the essential prerequisites for translation is reproducibility and validation in external settings.
This can be facilitated through the use of standardised radiomics software.
`mirp` is a Python package for standardised processing of medical imaging and computation of quantitative features.
Researchers can use `mirp` for their own radiomics analyses or to reproduce and validate radiomics of others.

# Statement of need

Lack of standardised radiomics software is one of the reasons for poor translation of radiomics tools to the clinic.
The Image Biomarker Standardisation Initiative has created reference standards for radiomics software: 1. a 
reference standard for basic image processing and feature extraction steps; and 2. a reference standard for image 
filters. There is currently a lack of fully IBSI-compliant radiomics packages in Python. Python is important for the 
radiomics field as commonly used machine learning and deep learning packages such as `scikit-learn` and `pytorch` 
are interfaced using Python. `mirp` facilitates both by offering a user-friendly API for standardised image processing 
and  feature extraction for machine learning-based radiomics, and standardised image processing for deep 
learning-based radiomics.

Intended user

History

# Alternatives

The most relevant Python-based alternative is `pyradiomics`. However, this package is not fully IBSI-compliant.

`mirp` is a Python package for standardised processing of medical imaging and computation of quantitative features.


It is intended for research use, and offers a user-friendly but highly configurable functional or generator-based API.
`mirp` integrates with the `pydicom` and `itk` packages for importing DICOM and other image file types (NIfTI, NRRD), respectively.
`mirp` converts such images to `numpy` arrays and internally stores these with associated metadata.
The `ray` package facilitates parallel image processing and feature computation.

Other packages offer similar core functionality.
Among these, the `pyradiomics` package is the most prominent and widely used.
`mirp` has several distinct advantages:
- Two reference standards for radiomics software have been published by the Biomarker Standardardisation Initiative (IBSI).
  The first IBSI reference standard concerns basic image processing and feature extraction, whereas the second reference standard concerns image filters.
  `mirp` is fully compliant with both standards, and integrates compliance tests within its test suite.
  This allows for validation of radiomics studies conducted using `mirp` using other IBSI-compliant packages.
- `mirp` is flexible regarding its file input, allowing users to provide directories, paths to files, as well as `numpy` arrays, using the same API.
- `mirp` additionally provides functions for extracting relevant image metadata, for identifying mask labels present in files, and for stand-alone image processing as part of a deep-learning pipeline.
- `mirp` can accelerate computation using parallel processing of multiple images and masks.
- `mirp` integrates conversion of positron emission tomography images to standardised uptake values that are used in clinics.
- `mirp` allows for bias-field correction of magnetic resonance images.
- `mirp` allows for perturbation or augmentation of images and masks as part of image processing workflow. 

Since its initial development in 2016, `mirp` has been used in a number of scientific publications.
Th


`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References