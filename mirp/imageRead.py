import os
import warnings

import itk
import numpy as np
import pandas as pd
import pydicom

from typing import Union
from mirp.dicomImport import read_dicom_image_series, read_dicom_rt_struct, read_roi_names, get_all_dicom_headers
from mirp.imageClass import ImageClass
from mirp.imageMetaData import get_itk_dicom_meta_tag
from mirp.settings.settingsGeneric import SettingsClass
from mirp.itkImport import read_itk_image, read_itk_segmentations


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


def load_image(image_folder,
               modality=None,
               roi_folder=None,
               registration_image_folder=None,
               image_name=None,
               roi_names=None,
               registration_image_name=None):

    # Import image
    img_obj: ImageClass = import_image(folder=image_folder,
                                       modality=modality,
                                       name_contains=image_name)

    # Load registration image
    if registration_image_folder == image_folder or registration_image_folder is None:
        img_reg_obj = img_obj
    else:
        img_reg_obj: ImageClass = import_image(folder=registration_image_folder,
                                               modality=modality,
                                               name_contains=registration_image_name)

    # Load segmentation
    if roi_names is not None:
        roi_list = import_segmentations(folder=roi_folder,
                                        roi_names=roi_names,
                                        image_object=img_reg_obj)
    else:
        roi_list = []

    return img_obj, roi_list


def import_image(folder, modality=None, name_contains=None):
    # Check folder contents, keep only files that are recognised as DICOM images or other image files.
    file_list = os.listdir(folder)
    file_list = [file_name for file_name in file_list if not os.path.isdir(os.path.join(folder, file_name))]

    # Find DICOM files
    dcm_file_list = [file_name for file_name in file_list if file_name.lower().endswith(".dcm")]

    # Find other image formats
    other_file_list = [file_name for file_name in file_list if file_name.lower().endswith((".nii", ".nii.gz", ".nrrd"))]

    if len(dcm_file_list) > 0:
        img_obj = read_dicom_image_series(image_folder=folder, modality=modality)

    elif len(other_file_list) > 0:
        img_obj = read_itk_image(image_folder=folder, modality=modality, name_contains=name_contains)

    else:
        raise FileNotFoundError(f"Could not find image files in the indicated folder: {folder}")

    return img_obj


def import_segmentations(folder, image_object, roi_names):
    # Check folder contents, keep only files that are recognised as DICOM images or other image files.
    file_list = os.listdir(folder)
    file_list = [file_name for file_name in file_list if not os.path.isdir(os.path.join(folder, file_name))]

    # Find DICOM files
    dcm_file_list = [file_name for file_name in file_list if file_name.lower().endswith(".dcm")]

    # Find other image formats
    other_file_list = [file_name for file_name in file_list if file_name.lower().endswith((".nii", ".nii.gz", ".nrrd"))]

    if len(dcm_file_list) > 0:
        # Attempt to obtain segmentations from DICOM
        roi_list = read_dicom_rt_struct(dcm_folder=folder, image_object=image_object, roi=roi_names)

        if len(roi_list) == 0 and len(other_file_list) > 0:
            # Attempt to obtain segmentation masks from other types of image file.
            roi_list = read_itk_segmentations(image_folder=folder, roi=roi_names)

    elif len(other_file_list) > 0:
        # Attempt to obtain segmentation masks from other types of image file.
        roi_list = read_itk_segmentations(image_folder=folder, roi=roi_names)

    else:
        roi_list = []

    if len(roi_list) == 0:
        warnings.warn(f"No segmentations were imported from {folder}. This could be because the folder does not contain segmentations, "
                      f"or none of the segmentations matches the roi_names argument.")

    return roi_list


def read_basic_image_characteristics(image_folder, folder_contains=None):

    # Check folder contents, keep only files that are recognised as images.
    file_list = os.listdir(image_folder)
    file_list = [file_name for file_name in file_list if not os.path.isdir(os.path.join(image_folder, file_name))]
    file_list = [file_name for file_name in file_list if file_name.lower().endswith((".dcm", ".ima", ".nii", ".nii.gz", ".nrrd"))]

    # File characteristics (file name, image type, modality, etc)
    list_char = []

    for file_name in file_list:

        # Check whether file name contains proper characters
        file_name = check_file_name(file_name=file_name, file_path=image_folder)

        # Set path to current file
        file_path = os.path.normpath(os.path.join(image_folder, file_name))

        # Set file type
        if file_name.lower().endswith((".dcm", ".ima")):
            img_file_type = "dicom"
        elif file_name.lower().endswith((".nii", ".nii.gz")):
            img_file_type = "nifti"
        elif file_name.lower().endswith(".nrrd"):
            img_file_type = "nrrd"
        else:
            img_file_type = "unknown"

        # Try to read file and get voxel grid data
        try:
            itk_img = itk.imread(filename=file_path)
            import_successful = True
        except:
            import_successful = False

        # If simple ITK was able to read the data
        if import_successful:

            # Load spatial data: note that simple ITK reads in (x,y,z) order
            img_origin = np.array(itk_img.GetOrigin())
            img_spacing = np.array(itk_img.GetSpacing())
            img_dimension = np.array(itk_img.GetSize())

            # Determine modality
            if img_file_type == "dicom":
                # From file meta information
                img_modality = itk_img.GetMetaData("0008|0060")
            else:
                # From file name
                if "MR" in file_name:
                    img_modality = "MR"
                elif "PET" in file_name:
                    img_modality = "PT"
                elif ("PT" in file_name) and ("PTV" not in file_name):
                    img_modality = "PT"
                elif ("CT" in file_name) and ("CTV" not in file_name):
                    img_modality = "CT"
                else:
                    if folder_contains is not None and len(file_list) == 1:
                        img_modality = folder_contains
                    else:
                        img_vox = itk.GetArrayFromImage(itk_img)
                        if (np.min(img_vox) == 0 or np.min(img_vox) == 1) and np.max(img_vox) == 1:
                            img_modality = "SEG"
                        else:
                            img_modality = "unknown"

            # In DICOM, update spacing with slice thickness as z-spacing
            if img_file_type == "dicom":
                img_spacing[2] = get_itk_dicom_meta_tag(itk_img=itk_img,
                                                        tag="0018|0050",
                                                        tag_type="float",
                                                        default=2.0)

            # Set characteristics
            df_char_curr = pd.Series({"file_name":        file_name,
                                      "file_path":        file_path,
                                      "file_type":        img_file_type,
                                      "modality":         img_modality,
                                      "size_x":           img_dimension[0],
                                      "size_y":           img_dimension[1],
                                      "size_z":           img_dimension[2],
                                      "spacing_x":        img_spacing[0],
                                      "spacing_y":        img_spacing[1],
                                      "spacing_z":        img_spacing[2],
                                      "pos_x":            img_origin[0],
                                      "pos_y":            img_origin[1],
                                      "pos_z":            img_origin[2]})

            # Append data frame to list
            list_char.append(df_char_curr)

        else:
            # Parse data where Simple ITK fails
            if not file_name.lower().endswith('.dcm'): continue
            if not file_name.lower().endswith('.dcm'): continue

            # Read DICOM file using pydicom
            dcm = pydicom.dcmread(image_folder + "/" + file_name, stop_before_pixels=True, force=True)

            # Determine modality
            img_modality = dcm.Modality

            if dcm.Modality == "RTSTRUCT":
                df_char_curr = pd.Series({"file_name": file_name,
                                          "file_path": file_path,
                                          "file_type": img_file_type,
                                          "modality": img_modality,
                                          "size_x": -1,
                                          "size_y": -1,
                                          "size_z": -1,
                                          "spacing_x": np.nan,
                                          "spacing_y": np.nan,
                                          "spacing_z": np.nan,
                                          "pos_x": np.nan,
                                          "pos_y": np.nan,
                                          "pos_z": np.nan})

                # Append data frame to list
                list_char.append(df_char_curr)

    # Concatenate list of data frames to single data frame
    df_char = pd.concat(list_char, axis=1).T

    return df_char


def check_file_name(file_name, file_path):
    """Checks file name and replaces non-ASCII characters."""

    # Check if name contains non-ASCII characters by attempting to encode file name as ascii
    try:
        file_name.encode("ascii")
    except UnicodeError:
        old_path = os.path.normpath(os.path.join(file_path, file_name))
        file_name = file_name.replace("ß", "ss")
        file_name = file_name.replace("ä", "ae")
        file_name = file_name.replace("Ä", "AE")
        file_name = file_name.replace("ö", "oe")
        file_name = file_name.replace("Ö", "OE")
        file_name = file_name.replace("ü", "ue")
        file_name = file_name.replace("Ü", "UE")
        new_path = os.path.normpath(os.path.join(file_path, file_name))
        os.rename(old_path, new_path)

    return file_name
