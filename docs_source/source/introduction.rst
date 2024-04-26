Introduction
============

What is radiomics?
------------------
Radiomics is the process of using medical images for, e.g., diagnosing disease or predicting how patients respond to
a treatment. Radiomics involves the use of computer algorithms to process medical images and predicting such outcomes.
There are two major radiomics branches, defined by what algorithms are used to predict outcomes. The first branch
(historically) of radiomics is characterised by the use of quantitative (handcrafted) features that are then used by
machine learning algorithms for tabular data. The second, more recent, branch uses deep learning algorithms to
directly learn from images themselves.

For more details, see reviews by Lambin et al. [Lambin2017]_ and van Timmeren et al. [vanTimmeren2020]_.

What is MIRP?
-------------

Medical Image Radiomics Processor (MIRP) is a python package for medical image analysis that is compliant with the
reference standards of the Image Biomarker Standardisation Initiative (IBSI) [Zwanenburg2020]_,
[Whybra2024]_. MIRP focuses on radiomics applications and supports computation of features for conventional
radiomics and image processing for deep-learning applications.

Why MIRP?
---------

In radiomics, image processing and feature computation are part of a larger workflow that also includes machine
learning. Python has some of the most commonly used machine learning packages, such *scikit-learn* and *pytorch*.
However, there was no Python package for image processing and feature computation that was fully compliant with the
IBSI reference standards -- i.e. a package whose output is reproducible by other IBSI-compliant software. MIRP fills
this gap.

Contact
-------
If you have any questions or run into issues, please visit the MIRP `GitHub repository <https://github
.com/oncoray/mirp>`_.

References
----------

.. [Lambin2017] Lambin P, Leijenaar RTH, Deist TM, Peerlings J, de Jong EEC, van Timmeren J, et al. Radiomics: the
  bridge between medical imaging and personalized medicine. Nat Rev Clin Oncol. 2017;14: 749-762.
  doi:`10.1038/nrclinonc.2017.141 <https://doi.org/10.1038/nrclinonc.2017.141>`_

.. [vanTimmeren2020] van Timmeren JE, Cester D, Tanadini-Lang S, Alkadhi H, Baessler B. Radiomics in medical
  imaging-“how-to” guide and critical reflection. Insights Imaging. 2020;11: 91.
  doi:`10.1186/s13244-020-00887-2 <https://doi.org/10.1186/s13244-020-00887-2>`_

.. [Zwanenburg2020] Zwanenburg A, Vallieres M, Abdalah MA, Aerts HJWL, Andrearczyk V, Apte A, et al. The Image
  Biomarker Standardization Initiative: Standardized Quantitative Radiomics for High-Throughput Image-based
  Phenotyping. Radiology. 2020;295: 328-338.
  doi:`10.1148/radiol.2020191145 <https://doi.org/10.1148/radiol.2020191145>`_

.. [Whybra2024] Whybra P, Zwanenburg A, Andrearczyk V, Schaer R, Apte AP, Ayotte A, et al. The Image Biomarker
  Standardization Initiative: Standardized Convolutional Filters for Reproducible Radiomics and Enhanced Clinical
  Insights. Radiology. 2024;310: e231319. doi:`10.1148/radiol.231319 <https://doi.org/10.1148/radiol.231319>`_