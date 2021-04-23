# Medical Image Radiomics Processor

The Medical Image Radiomics Processor (MIRP) is an IBSI-compliant radiomics processor that is used to extract image biomarkers from medical imaging. MIRP is an end-to-end framework featuring parallel processing that is configured using `xml` files.

# Citation info
If you use MIRP, please cite the following work:
```Zwanenburg A, Leger S, Agolli L, Pilz K, Troost EG, Richter C, Löck S. Assessing robustness of radiomic features by image perturbation. Scientific reports. 2019 Jan 24;9(1):614.```

# Usage
To be documented.

# Developers

MIRP was developed by:
* Alex Zwanenburg
* Stefan Leger
* Sebastian Starke

# New in version 1.1
The `extract_images_for_deep_learning` and underlying functions have been reworked. The `deep_learning` section of the settings configuration xml file have been deprecated in favour of function arguments.