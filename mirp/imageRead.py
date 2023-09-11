import os
import warnings
import pandas as pd

from typing import Union
from mirp.dicomImport import read_dicom_image_series, read_dicom_rt_struct, read_roi_names, get_all_dicom_headers
from mirp.imageClass import ImageClass
from mirp.settings.settingsGeneric import SettingsClass


def find_regions_of_interest(roi_folder, subject):

    # Obtain file names and ROI names
    file_names, roi_names = read_roi_names(dcm_folder=roi_folder)

    if len(roi_names) == 0:
        warnings.warn(f"No ROI segmentations were found for the current subject ({subject}).")

    roi_table = pd.DataFrame({"subject": subject, "file_name": file_names, "roi": roi_names})

    return roi_table


def find_imaging_parameters(image_folder,
                            modality,
                            subject,
                            plot_images,
                            write_folder,
                            roi_folder=None,
                            registration_image_folder=None,
                            settings: Union[None, SettingsClass] = None,
                            roi_names=None):
    """
    :param image_folder: path; path to folder containing image data.
    :param modality: string; identifies modality of the image in the image folder.
    :param subject: string; name of the subject.
    :param plot_images: bool; flag to set image extraction. An image is created at the center of each ROI.
    :param write_folder: path; path to folder where the analysis should be written.
    :param roi_folder: path; path to folder containing the region of interest definitions.
    :param registration_image_folder: path; path to folder containing image data on which the region of interest was
     originally created. If None, it is assumed that the image in image_folder was used to the define the roi.
    :param settings:
    :param roi_names:
    :return:
    """

    from mirp.imagePlot import plot_image
    from mirp.imageMetaData import get_meta_data
    from mirp.imageProcess import estimate_image_noise

    # Read DICOM series
    img_obj: ImageClass = read_dicom_image_series(image_folder=image_folder, modality=modality)

    # Load registration image
    if registration_image_folder == image_folder or registration_image_folder is None:
        img_reg_obj = img_obj
    else:
        img_reg_obj: ImageClass = read_dicom_image_series(image_folder=image_folder, modality=modality)

    # Load segmentations
    roi_list = read_dicom_rt_struct(dcm_folder=roi_folder, image_object=img_reg_obj, roi=roi_names)

    # Load dicom headers for all slices in the image object.
    dcm_list = get_all_dicom_headers(image_folder=image_folder,
                                     modality=modality,
                                     sop_instance_uid=img_obj.slice_table.sop_instance_uid.values)

    # Parse metadata
    metadata_table = get_meta_data(dcm_list=dcm_list, modality=modality)

    # Add sample identifier, folder and image noise
    metadata_table["subject"] = subject
    metadata_table["folder"] = image_folder
    metadata_table["noise"] = estimate_image_noise(img_obj=img_obj, settings=None, method="chang")

    # Find the segmentation range.
    if settings is None:
        g_range = None
    else:
        g_range = settings.roi_resegment.intensity_range

    # Plot images
    if isinstance(plot_images, str):
        if plot_images == "single":
            plot_image(img_obj=img_obj, roi_list=roi_list, slice_id="roi_center", file_path=write_folder,
                       file_name=subject + "_" + modality,
                       g_range=g_range)
        elif plot_images == "all_roi":
            plot_image(img_obj=img_obj, roi_list=roi_list, slice_id="all_roi", file_path=write_folder,
                       file_name=subject + "_" + modality,
                       g_range=g_range)
        elif plot_images == "all":
            plot_image(img_obj=img_obj, roi_list=roi_list, slice_id="all", file_path=write_folder,
                       file_name=subject + "_" + modality,
                       g_range=g_range)

    elif isinstance(plot_images, bool):
        if plot_images:
            plot_image(img_obj=img_obj, roi_list=roi_list, slice_id="roi_center", file_path=write_folder,
                       file_name=subject + "_" + modality,
                       g_range=settings.roi_resegment.intensity_range)

    else:
        raise TypeError("plot_image is expected to be a string or boolean.")

    # Write table to single file for case-by-case analysis
    metadata_table.to_frame().T.to_csv(
        path_or_buf=os.path.normpath(os.path.join(write_folder, subject + "_" + modality + "_meta_data.csv")),
        sep=";", na_rep="NA", index=False, decimal=".")

    return metadata_table.to_frame().T

