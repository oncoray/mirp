import logging
import os
import warnings

import SimpleITK as sitk
import numpy as np
import pandas as pd
import pydicom

from mirp.dicomImport import read_dicom_image_series, read_dicom_rt_struct, read_roi_names
from mirp.imageClass import ImageClass
from mirp.imageMetaData import get_sitk_dicom_meta_tag
from mirp.itkImport import read_itk_image, read_itk_segmentations
# Monkey patch for dicom files with implicit VR tags that also export meta file tags (0x0002 0x----) as meta data.
# This seems to have happened for some RayStation exports
# from mirp.pydicom_fix import read_dataset
# pydicom.filereader.read_dataset = read_dataset

from pydicom.filereader import read_dataset


def find_regions_of_interest(roi_folder, subject):

    # Obtain file names and ROI names
    file_names, roi_names = read_roi_names(dcm_folder=roi_folder)

    if len(roi_names) == 0:
        warnings.warn(f"No ROI segmentations were found for the current subject ({subject}).")

    roi_table = pd.DataFrame({"subject": subject, "file_name": file_names, "roi": roi_names})

    return roi_table


def find_imaging_parameters(image_folder, modality, subject, plot_images, write_folder,
                            roi_folder=None, registration_image_folder=None, settings=None, roi_names=None):
    """
    :param image_folder: path; path to folder containing image data.
    :param modality: string; identifies modality of the image in the image folder.
    :param subject: string; name of the subject.
    :param plot_images: bool; flag to set image extraction. An image is created at the center of each ROI.
    :param write_folder: path; path to folder where the analysis should be written.
    :param roi_folder: path; path to folder containing the region of interest definitions.
    :param registration_image_folder: path; path to folder containing image data on which the region of interest was originally created. If None, it is assumed that the image in
    image_folder was used to the define the roi.
    :param settings:
    :param roi_names:
    :return:
    """

    # TODO: make it so that advanced meta-data can actually be obtained.

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

    # metadata_table =


def find_imaging_parameters_deprecated(image_folder, modality, subject, plot_images, write_folder, roi_folder=None, roi_reg_img_folder=None, settings=None, roi_names=None):
    """
    :param image_folder: path; path to folder containing image data.
    :param modality: string; identifies modality of the image in the image folder.
    :param subject: string; name of the subject.
    :param plot_images: bool; flag to set image extraction. An image is created at the center of each ROI.
    :param write_folder: path; path to folder where the analysis should be written.
    :param roi_folder: path; path to folder containing the region of interest definitions.
    :param roi_reg_img_folder: path; path to folder containing image data on which the region of interest was originally created. If None, it is assumed that the image in
    image_folder was used to the define the roi.
    :param settings:
    :param roi_names:
    :return:
    """

    from mirp.imagePlot import plot_image
    from mirp.imageMetaData import get_meta_data
    from mirp.imageProcess import estimate_image_noise

    # # Convert a single input modality to list
    # if type(modality) is str: modality = [modality]

    # Read image characteristics
    df_img_char = read_basic_image_characteristics(image_folder=image_folder)

    # Remove non-modality objects
    df_img_char = df_img_char.loc[np.logical_and(df_img_char.modality.isin([modality]), df_img_char.file_type.isin(["dicom"]))]
    if len(df_img_char) == 0:
        logging.warning("No dicom images with modality %s were found for %s.", modality[0], subject)
        return None

    # Check if image parameters need to be read within roi slices
    if roi_names is not None:
        img_obj, roi_list = load_image(image_folder=image_folder, roi_folder=roi_folder, registration_image_folder=roi_reg_img_folder,
                                       modality=modality, roi_names=roi_names)

        # Register rois to image
        for ii in np.arange(len(roi_list)):
            roi_list[ii].register(img_obj=img_obj)
    else:
        roi_list = None

    # Read meta tags
    if modality in ["CT", "PT", "MR"]:
        df_meta = get_meta_data(image_file_list=df_img_char.file_path.values.tolist(), modality=modality)
    else:
        logging.warning("Dicom images could not be analysed for provided modality.")
        return None

    df_meta["subject"] = subject
    df_meta["folder"] = image_folder

    if roi_names is not None:
        df_meta["noise"] = estimate_image_noise(img_obj=img_obj, settings=settings, method="chang")

    # Plot images
    if isinstance(plot_images, str):
        if plot_images == "single":
            plot_image(img_obj=img_obj, roi_list=roi_list, slice_id="roi_center", file_path=write_folder, file_name=subject + "_" + modality[0],
                       g_range=settings.roi_resegment.g_thresh)
        elif plot_images == "all_roi":
            plot_image(img_obj=img_obj, roi_list=roi_list, slice_id="all_roi", file_path=write_folder, file_name=subject + "_" + modality[0],
                       g_range=settings.roi_resegment.g_thresh)
        elif plot_images == "all":
            plot_image(img_obj=img_obj, roi_list=roi_list, slice_id="all", file_path=write_folder, file_name=subject + "_" + modality[0],
                       g_range=settings.roi_resegment.g_thresh)
    else:
        if plot_images:
            plot_image(img_obj=img_obj, roi_list=roi_list, slice_id="roi_center", file_path=write_folder, file_name=subject + "_" + modality[0],
                       g_range=settings.roi_resegment.g_thresh)

    # Write table to single file for case-by-case analysis
    df_meta.to_frame().T.to_csv(path_or_buf=os.path.normpath(os.path.join(write_folder, subject + "_" + modality[0] + "_meta_data.csv")), sep=";", na_rep="NA", index=False,
                                decimal=".")

    return df_meta.to_frame().T


def load_image(image_folder, modality=None, roi_folder=None, registration_image_folder=None, image_name=None, roi_names=None, registration_image_name=None):

    # Import image
    img_obj: ImageClass = import_image(folder=image_folder, modality=modality, name_contains=image_name)

    # Load registration image
    if registration_image_folder == image_folder or registration_image_folder is None:
        img_reg_obj = img_obj
    else:
        img_reg_obj: ImageClass = import_image(folder=registration_image_folder, modality=modality, name_contains=registration_image_name)

    # Load segmentation
    if roi_names is not None:
        roi_list = import_segmentations(folder=roi_folder, roi_names=roi_names, image_object=img_reg_obj)
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
        if file_name.lower().endswith((".dcm", ".ima")): img_file_type = "dicom"
        elif file_name.lower().endswith((".nii", ".nii.gz")): img_file_type = "nifti"
        elif file_name.lower().endswith(".nrrd"): img_file_type = "nrrd"
        else: img_file_type = "unknown"

        # Try to read file and get voxel grid data
        try:
            sitk_img = sitk.ReadImage(file_path)
            import_successful = True
        except:
            import_successful = False

        # If simple ITK was able to read the data
        if import_successful:

            # Load spatial data: note that simple ITK reads in (x,y,z) order
            img_origin    = np.array(sitk_img.GetOrigin())
            img_spacing   = np.array(sitk_img.GetSpacing())
            img_dimension = np.array(sitk_img.GetSize())

            # Determine modality
            if img_file_type == "dicom":
                # From file meta information
                img_modality = sitk_img.GetMetaData("0008|0060")
            else:
                # From file name
                if "MR" in file_name:    img_modality = "MR"
                elif "PET" in file_name: img_modality = "PT"
                elif ("PT" in file_name) and ("PTV" not in file_name): img_modality = "PT"
                elif ("CT" in file_name) and ("CTV" not in file_name): img_modality = "CT"
                else:
                    if folder_contains is not None and len(file_list) == 1:
                        img_modality = folder_contains
                    else:
                        img_vox = sitk.GetArrayFromImage(sitk_img)
                        if (np.min(img_vox) == 0 or np.min(img_vox) == 1) and np.max(img_vox) == 1:
                            img_modality = "SEG"
                        else:
                            img_modality = "unknown"

            # In DICOM, update spacing with slice thickness as z-spacing
            if img_file_type == "dicom":
                img_spacing[2] = get_sitk_dicom_meta_tag(sitk_img=sitk_img, tag="0018|0050", tag_type="float", default=2.0)

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

            # Read DICOM file using pydicom
            dcm = pydicom.dcmread(image_folder + "/" + file_name, stop_before_pixels=True, force=True)

            # Determine modality
            img_modality = dcm.Modality

            if dcm.Modality == "RTSTRUCT":
                df_char_curr = pd.Series({"file_name": file_name,
                                          "file_path": file_path,
                                          "file_type": img_file_type,
                                          "modality":  img_modality,
                                          "size_x":    -1,
                                          "size_y":    -1,
                                          "size_z":    -1,
                                          "spacing_x": np.nan,
                                          "spacing_y": np.nan,
                                          "spacing_z": np.nan,
                                          "pos_x":     np.nan,
                                          "pos_y":     np.nan,
                                          "pos_z":     np.nan})

                # Append data frame to list
                list_char.append(df_char_curr)

    # Concatenate list of data frames to single data frame
    df_char = pd.concat(list_char, axis=1).T

    return df_char


def check_file_name(file_name, file_path):
    """Checks file name and replaces non-ASCII characters. This prevents crashes with SimpleITK readImage function"""

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
