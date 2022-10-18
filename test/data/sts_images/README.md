Subset of the soft-tissue sarcoma dataset
===

This is a subset of the dataset was used to validate the findings in the IBSI 1 study. The `GTV_Mass` contour was used.

The images are available in both DICOM and NIfTI formats, and consists of the image itself (image) and its segmentation (mask).
The segmentation in DICOM format is an RTSTRUCT and needs to be converted to a voxel mask, whereas in the NIfTI format, the mask is already a voxel mask.
Consider using the NIfTI mask in case conversion of in-plane polygons to a mask is not supported.

## License
The images are licensed under the Creative Commons Attribution 3.0 Unported Licence. To view a copy of this license, visit https://creativecommons.org/licenses/by/3.0/ or send a letter to Creative Commons, PO Box 1866, Mount View, CA 94042, USA.

## Acknowledgments
This dataset is based on the images and contours of the Soft-tissue-Sarcoma collection that was uploaded to the Cancer Imaging Archive, see citation onformation.

Alex Zwanenburg:
* converted raw counts in PET images to SUV values.
* performed N4 bias field correction on T1weighted MR images.
* normalised the MR images based on subcutaneous fat intensities.
* isolated the `GTV_Mass` contour.
* cropped the image 5 cm around the contour.
* exported the image and contour to DICOM format.
* converted the image and the mask from DICOM and RTSTRUCT formats to NIfTI format.

The above operations were performed using MIRP (https://github.com/oncoray/mirp).

## Citation information
Please include the following citations when using this dataset:

* Vallières, M., Freeman, C. R., Skamene, S. R., El Naqa, I. (2015). A radiomics model from joint FDG-PET and MRI texture features for the prediction of lung metastases in soft-tissue sarcomas of the extremities. The Cancer Imaging Archive. DOI: 10.7937/K9/TCIA.2015.7GO2GSKS
* Vallières, M., Freeman, C. R., Skamene, S. R., El Naqa, I. (2015). A radiomics model from joint FDG-PET and MRI texture features for the prediction of lung metastases in soft-tissue sarcomas of the extremities. Physics in Medicine and Biology. DOI: 10.1088/0031-9155/60/14/5471
* Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., Prior, F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: 10.1007/s10278-013-9622-7