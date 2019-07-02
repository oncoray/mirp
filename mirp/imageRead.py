import logging
import os

import SimpleITK as sitk
import numpy as np
import pandas as pd
import pydicom

from mirp.imageClass import ImageClass
from mirp.imageMetaData import get_sitk_dicom_meta_tag, get_pydicom_meta_tag, get_rtstruct_roi_names
from mirp.imageSUV import SUVscalingObj, suv_list_update
# Monkey patch for dicom files with implicit VR tags that also export meta file tags (0x0002 0x----) as meta data.
# This seems to have happened for some RayStation exports
from mirp.pydicom_fix import read_dataset
from mirp.roiClass import RoiClass, merge_roi_objects
from mirp.utilities import parse_roi_name

pydicom.filereader.read_dataset = read_dataset


def find_regions_of_interest(roi_folder, subject):

    # Read roi characteristics
    df_char = read_basic_image_characteristics(image_folder=roi_folder, folder_contains=None)

    # Remove non-roi objects from the list
    df_char = df_char.loc[df_char.modality.isin(["RTSTRUCT", "SEG"]), ]

    if len(df_char) == 0:
        logging.warning("No segmentation files were found for %s.", subject)
        return None

    # Find roi names in files on path
    file_names, roi_names = read_segment_names(df_char)

    # Add subject, file names and roi names to data frame
    df_roi = pd.DataFrame({"subject":   subject,
                           "file_name": file_names,
                           "roi":       roi_names})

    return df_roi


def find_imaging_parameters(image_folder, modality, subject, plot_images, write_folder, roi_folder=None, roi_reg_img_folder=None, settings=None, roi_names=None):
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

    # Convert a single input modality to list
    if type(modality) is str: modality = [modality]

    # Read image characteristics
    df_img_char = read_basic_image_characteristics(image_folder=image_folder)

    # Remove non-modality objects
    df_img_char = df_img_char.loc[np.logical_and(df_img_char.modality.isin(modality), df_img_char.file_type.isin(["dicom"]))]
    if len(df_img_char) == 0:
        logging.warning("No dicom images with modality %s were found for %s.", modality[0], subject)
        return None

    # Check if image parameters need to be read within roi slices
    if roi_names is not None:
        img_obj, roi_list = load_image(image_folder=image_folder, settings=settings, roi_folder=roi_folder,
                                       roi_reg_img_folder=roi_reg_img_folder, modality=modality, roi_names=roi_names)

        # Register rois to image
        for ii in np.arange(len(roi_list)):
            roi_list[ii].register(img_obj=img_obj)
    else:
        roi_list = None

    # Read meta tags
    if modality[0] in ["CT", "PT", "MR"]:
        df_meta = get_meta_data(image_file_list=df_img_char.file_path.values.tolist(), modality=modality[0])
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


def load_image(image_folder, settings, roi_folder=None, roi_reg_img_folder=None, modality=None, roi_names=None):

    # Provide standard set of modalities
    # if modality is None: modality = ("CT", "PT", "MR")

    # Convert a single input modality to list
    if type(modality) is str or modality is None: modality = [modality]

    ####################################################################################################################
    # Load images
    ####################################################################################################################

    # Read image characteristics
    df_char = read_basic_image_characteristics(image_folder=image_folder, folder_contains=modality[0])

    # Load image object from file
    if np.any(np.logical_and(df_char.file_type.isin(["dicom", "nifti", "nrrd"]), df_char.modality.isin(modality))):
        img_obj = import_image(df_char=df_char.loc[np.logical_and(df_char.file_type.isin(["dicom", "nifti", "nrrd"]), df_char.modality.isin(modality)),], modality=modality[0])
    else:
        img_obj = None
        logging.error("No image files were read.")

    ####################################################################################################################
    # Load image for roi registration
    ####################################################################################################################
    if roi_reg_img_folder == image_folder or roi_reg_img_folder is None:
        img_reg_obj = img_obj
    else:
        # Read registration image characteristics
        df_char = read_basic_image_characteristics(image_folder=roi_reg_img_folder, folder_contains=modality[0])

        # Load registration image object from file
        if np.any(df_char.file_type.isin(["dicom", "nifti", "nrrd"])):
            img_reg_obj = import_image(df_char=df_char.loc[df_char.file_type.isin(["dicom", "nifti", "nrrd"]),], modality=modality[0])
        else:
            img_reg_obj = None
            logging.warning("Registration image for ROIs was not found.")

    ####################################################################################################################
    # Load segmentations
    ####################################################################################################################

    # Read roi characteristics
    df_char = read_basic_image_characteristics(image_folder=roi_folder, folder_contains="SEG")

    # Load segmentations
    if np.any(df_char.modality.isin(["RTSTRUCT", "SEG"])):
        roi_list = import_segmentation(df_char=df_char.loc[df_char.modality.isin(["RTSTRUCT", "SEG"]),], img_obj=img_reg_obj,
                                       settings=settings, req_roi_names=roi_names)
    else:
        roi_list = []
        logging.error("No segmentation files were read.")

    return img_obj, roi_list


def read_basic_image_characteristics(image_folder, folder_contains=None):

    # Check folder contents
    file_list = os.listdir(image_folder)

    # File characteristics (file name, image type, modality, etc)
    list_char = []

    for file_name in file_list:

        # Inspect files to determine image characteristics
        if file_name.lower().endswith((".dcm", ".ima", ".nii", ".nii.gz", ".nrrd")):

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


def import_image(df_char, modality):
    # Read image from file

    # Perform consistency checks - x and y grid dimensions
    if np.any(np.logical_or(df_char.size_x.values - df_char.size_x.values[0] != 0,
                            df_char.size_y.values - df_char.size_y.values[0] != 0)):
        logging.error("Mismatching voxel grid dimensions while reading images from file.")

    # Perform consistency checks - x and y voxel dimensions
    if np.any(np.logical_or(df_char.spacing_x.values - df_char.spacing_x.values[0] != 0.0,
                            df_char.spacing_y.values - df_char.spacing_y.values[0] != 0.0)):
        logging.error("Mismatching voxel dimensions while reading images from file.")

    # Setup empty voxel object - assume that images are stacked in the z-dimension
    img_dims = np.array([np.sum(df_char.size_z), df_char.size_y[0], df_char.size_x[0]])
    img_vox  = np.zeros(img_dims, dtype=np.float32)

    # Setup empty slice-position array
    img_slice_pos = np.zeros(img_dims[0], dtype=np.float32)

    # Sort data frame by increasing slice position
    df_char = df_char.sort_values(by="pos_z").reset_index(drop=True)

    # Set image origin and spacing from sorted pandas frame
    img_origin = np.array([df_char.pos_z.values[0], df_char.pos_y.values[0], df_char.pos_x.values[0]])

    # Check if spacing is correct. Slice thickness may be different from what is expected.
    if len(df_char) > 1:
        actual_spacing_z = np.median(np.abs(np.diff(df_char.pos_z.values)))
        if np.around(actual_spacing_z - df_char.spacing_z.values[0], decimals=5) != 0.0:
            logging.warning("Slice thickness %s is inconsistent with actual spacing %s. z-spacing is replaced.", str(df_char.spacing_z.values[0]), actual_spacing_z)
            img_spacing = np.array([actual_spacing_z, df_char.spacing_y.values[0], df_char.spacing_x.values[0]])
        else:
            img_spacing = np.array([df_char.spacing_z.values[0], df_char.spacing_y.values[0], df_char.spacing_x.values[0]])
    else:
        img_spacing = np.array([df_char.spacing_z.values[0], df_char.spacing_y.values[0], df_char.spacing_x.values[0]])

    # Set initial offset for slices: the offset increases after adding slices from files
    slice_offset = 0

    # Collect imaging-specific data
    suv_obj_list = []
    if modality == "PT":
        for ii in np.arange(0, len(df_char)):
            if df_char.file_type.values[ii] == "dicom":
                suv_obj_list += [SUVscalingObj(dcm=pydicom.dcmread(df_char.file_path.values[ii], stop_before_pixels=True, force=True))]

        # Update list, e.g. to find the start of acquisition
        suv_obj_list = suv_list_update(suv_obj_list=suv_obj_list)

    flag_image_missing = False

    # Iterate over files
    for ii in np.arange(0, len(df_char)):

        # Read image
        sitk_img = sitk.ReadImage(df_char.file_path.values[ii])

        # Set slice range
        slice_range = np.arange(0, df_char.size_z.values[ii]) + slice_offset

        # Get slice voxels
        slice_vox = sitk.GetArrayFromImage(sitk_img).astype(dtype=np.float32)

        # SUV conversion
        if modality == "PT" and df_char.file_type.values[ii] == "dicom":
            # Get suv_scale
            suv_scale = suv_obj_list[ii].get_scale_factor(suv_normalisation="bw")

            if suv_scale is not None:
                slice_vox *= suv_scale
            else:
                slice_vox = None

        # Add to voxel object; fill with NaN if the range could not be read
        if slice_vox is not None:
            img_vox[slice_range, :, :] = slice_vox
        else:
            flag_image_missing = True

        # Update slice pos
        img_slice_pos[slice_range] = df_char.pos_z.values[ii] + np.arange(0, df_char.size_z.values[ii]) * df_char.spacing_z.values[ii]

        # Update offset
        slice_offset += df_char.size_z.values[ii]

    # Update image direction
    img_orientation = np.array(sitk_img.GetDirection())[::-1]

    # Create image
    img_obj = ImageClass(voxel_grid=img_vox, origin=img_origin, spacing=img_spacing, slice_z_pos=img_slice_pos,
                         orientation=img_orientation, modality=df_char.modality[0], spat_transform="base", no_image=flag_image_missing)

    # Collect garbage
    del sitk_img

    return img_obj


def import_segmentation(df_char, img_obj, settings, req_roi_names=None):
    # Reads segmentation files

    # Empty roi list
    roi_list = []

    for ii in np.arange(0, len(df_char)):

        # Parse according to file type and modality
        if df_char.modality.values[ii] == "SEG" and df_char.file_type.values[ii] != "dicom":
            # Roi map from voxel volume masks (non-dicom)
            roi_list += import_segment_from_volume(df_char=df_char.iloc[[ii]], req_roi_names=req_roi_names)

        elif df_char.modality.values[ii] == "SEG" and df_char.file_type.values[ii] == "dicom":
            # TODO: Roi map from mask voxel volumes (dicom SEG)
            logging.warning("Cannot parse dicom segmentation file: %s", df_char.file_path.values[ii])  # requires future implementation

        elif df_char.modality.values[ii] == "RTSTRUCT" and df_char.file_type.values[ii] == "dicom":
            # Roi map from dicom segmentation contours (RT structures)
            roi_list += import_segment_from_contour(df_char=df_char.iloc[[ii]], img_obj=img_obj, settings=settings, req_roi_names=req_roi_names)

        else:
            logging.warning("Cannot parse segmentation file:%s", df_char.file_path.values[ii])

    return roi_list


def import_segment_from_volume(df_char, req_roi_names):
    # Read segmentations from voxel volumes

    roi_list = []

    # Iterate over the roi
    for current_roi in req_roi_names:

        # Parse the current roi to identify the individual rois
        combined_roi_names = parse_roi_name(roi=current_roi)

        merge_roi_list = []

        # Iterate over individual rois
        for roi in combined_roi_names:
            for ii in np.arange(0, len(df_char)):

                # Find the roi name as given by the file
                file_roi_name = df_char.file_name.values[ii].lower().split(".")[0]

                # Skip files that do not match the current roi name
                if roi.lower() != file_roi_name:
                    continue

                # Load image file using simple itk
                sitk_img = sitk.ReadImage(df_char.file_path.values[ii])
                roi_map = sitk.GetArrayFromImage(sitk_img) * 1.0 > 0.0

                # Read roi spatial identifiers: note that because simpleitk reads (x,y,z), order is inverted
                roi_origin = np.array(sitk_img.GetOrigin())[::-1]
                roi_spacing = np.array(sitk_img.GetSpacing())[::-1]
                roi_dimension = np.array(sitk_img.GetSize())[::-1]
                roi_orientation = np.array(sitk_img.GetDirection())[::-1]

                # Calculate z position
                roi_slice_z_pos = roi_origin[0] + np.arange(0, roi_dimension[0]) * roi_spacing[0]

                # Create roi object using roi_map and roi_name
                roi_map_obj = ImageClass(voxel_grid=roi_map, origin=roi_origin, spacing=roi_spacing, orientation=roi_orientation,
                                         slice_z_pos=roi_slice_z_pos)
                merge_roi_list += [RoiClass(name=current_roi, contour=None, roi_mask=roi_map_obj)]

        # Combine ROI objects to a single ROI
        roi_list += [merge_roi_objects(roi_list=merge_roi_list)]

    return roi_list


def import_segment_from_contour(df_char, img_obj, settings, req_roi_names=None):
    # Reads contours and perform segmentation

    roi_list = []

    # Contours cannot be properly segmented without an origin
    if np.any(np.isnan(img_obj.origin)):
        return roi_list

    for current_roi_name in req_roi_names:

        # Identify rois in a combined name
        combined_roi_names = parse_roi_name(roi=current_roi_name)

        # Local roi container list
        merge_roi_list = []

        for individual_roi_name in combined_roi_names:
            for ii in np.arange(0, len(df_char)):

                # Read the dicom file
                dcm = pydicom.dcmread(df_char.file_path.values[ii], stop_before_pixels=True, force=True)

                # Find roi names in the current dicom file
                rtstruct_roi_names, rtstruct_roi_numbers = get_rtstruct_roi_names(dcm=dcm, with_roi_number=True)

                # See if the roi name is in the current dicom file
                if individual_roi_name not in rtstruct_roi_names:
                    continue

                # Determine if there is a roi contour sequence (0x3006, 0x0039)
                if not get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x3006, 0x0039)):
                    continue

                # Find the roi number corresponding to individual_roi_name
                individual_roi_number = [rtstruct_roi_numbers[ii] for ii in range(len(rtstruct_roi_numbers)) if rtstruct_roi_names[ii] == individual_roi_name][0]

                # Find which roi contour sequence corresponds has the individual_roi_number
                for roi_contour_elem in dcm[0x3006, 0x0039]:

                    # Determine if the Reference ROI number tag exists (3006, 0084)
                    if not get_pydicom_meta_tag(dcm_seq=roi_contour_elem, tag=(0x3006, 0x0084)):
                        continue

                    # Skip if current roi contour element does not contain the requested roi
                    if get_pydicom_meta_tag(dcm_seq=roi_contour_elem, tag=(0x3006, 0x0084), tag_type="str") != individual_roi_number:
                        continue

                    # Determine if the contour sequence element exists (3006, 0040)
                    if not get_pydicom_meta_tag(dcm_seq=roi_contour_elem, tag=(0x3006, 0x0040)):
                        continue

                    # Load the contour sequence
                    contour_sequence = roi_contour_elem[0x3006, 0x0040]

                    # Empty contour_list
                    contour_data_list = []

                    # Iterate over contours in the contour_sequence
                    for current_contour_sequence in contour_sequence:

                        # Check if the geometric type exists (3006, 0042)
                        if not get_pydicom_meta_tag(dcm_seq=current_contour_sequence, tag=(0x3006, 0x0042), test_tag=True):
                            continue

                        # Check if the geometric type equals "CLOSED_PLANAR"
                        if get_pydicom_meta_tag(dcm_seq=current_contour_sequence, tag=(0x3006, 0x0042), tag_type="str") != "CLOSED_PLANAR":
                            continue

                        # Check if contour data exists (3006, 0050)
                        if not get_pydicom_meta_tag(dcm_seq=current_contour_sequence, tag=(0x3006, 0x0050), test_tag=True):
                            continue

                        contour_data = np.array(get_pydicom_meta_tag(dcm_seq=current_contour_sequence, tag=(0x3006, 0x0050), tag_type="mult_float"), dtype=np.float64)
                        contour_data = contour_data.reshape((-1, 3))

                        # Determine if there is an offset (3006, 0045)
                        contour_offset = np.array(get_pydicom_meta_tag(dcm_seq=current_contour_sequence, tag=(0x3006, 0x0045), tag_type="mult_float", default=[0.0, 0.0, 0.0]),
                                                  dtype=np.float64)

                        # Remove the offset from the data
                        contour_data -= contour_offset

                        # Add contour data to the contour list
                        contour_data_list += [contour_data]

                    # Check if the contour_data_list contains data
                    if len(contour_data_list) > 0:
                        # Create a ROIClass object for the current roi
                        roi_obj = RoiClass(name=individual_roi_name, contour=contour_data_list)

                        # Convert contour into segmentation object
                        roi_obj.create_mask_from_contours(img_obj=img_obj, settings=settings)

                        # Append the roi_obj to the list
                        merge_roi_list += [roi_obj]

        # Combine ROI objects to a single ROI
        if len(merge_roi_list) > 0:
            roi_list += [merge_roi_objects(roi_list=merge_roi_list)]

    return roi_list


def read_segment_names(df_char):

    roi_names = []
    file_names = []

    # Iterate over files and add roi names
    for ii in np.arange(0, len(df_char)):

        # Parse according to file type and modality
        if df_char.modality.values[ii] == "SEG" and df_char.file_type.values[ii] != "dicom":
            # Roi name from file name (non-dicom)
            file_name_split = df_char.file_name.values[ii].lower().split(".")

            # Add roi names and corresponding file names
            roi_names  += [file_name_split[0]]
            file_names += [df_char.file_name.values[ii]]

        elif df_char.modality.values[ii] == "SEG" and df_char.file_type.values[ii] == "dicom":
            # Roi map from mask voxel volumes (dicom)
            logging.warning("Cannot parse dicom segmentation file: %s",
                            df_char.file_path.values[ii])  # requires future implementation

        elif df_char.modality.values[ii] == "RTSTRUCT" and df_char.file_type.values[ii] == "dicom":
            # Roi map from dicom segmentation contours (RT structures)

            # Empty place holder
            dcm_roi = get_rtstruct_roi_names(image_file=df_char.file_path.values[ii])

            # Add roi names and file name
            if len(dcm_roi) > 0:
                roi_names += dcm_roi
                file_names += [df_char.file_name.values[ii]] * len(dcm_roi)

        else:
            logging.warning("Cannot parse segmentation file: %s", df_char.file_path.values[ii])

    return file_names, roi_names


# def getCTDicomMetaTags(df_char, roi_list):
#     """Reads meta tags for CT data sets. The assumption is that df_char only contains a single contiguous series or single file"""
#
#     import copy
#
#     def getCTMetaTags(df_char, roi_name="none"):
#         # Set empty meta data frame
#         df_meta = pd.DataFrame({"modality":              "NA",
#                                 "roi":                   roi_name,
#                                 "manufacturer":          "NA",
#                                 "scanner_model":         "NA",
#                                 "tube_potential":        np.nan,
#                                 "mean_tube_current":     np.nan,
#                                 "mean_exposure_time":    np.nan,
#                                 "mean_exposure":         np.nan,
#                                 "reconstruction_kernel": "NA",
#                                 "slice_thickness":       np.nan,
#                                 "pixel_spacing_x":       np.nan,
#                                 "pixel_spacing_y":       np.nan,
#                                 "volume":                np.nan},
#                                index=[0])
#
#         # Get the number of slices
#         n_slices = len(df_char) * 1.0
#
#         # Set 0-valued placeholders
#         tube_current = 0.0
#         exposure_time = 0.0
#         exposure = 0.0
#
#         for ii in np.arange(0, len(df_char)):
#             # Load as simple itk image
#             sitk_img = sitk.ReadImage(df_char.file_path[ii])
#
#             # Tube current in mA
#             tube_current += get_sitk_dicom_meta_tag(sitk_img=sitk_img, tag="0018|1151", tag_type="float", default=np.nan)
#
#             # Exposure time in milliseconds
#             exposure_time += get_sitk_dicom_meta_tag(sitk_img=sitk_img, tag="0018|1150", tag_type="float", default=np.nan)
#
#             # Radiation exposure in mAs
#             exposure += get_sitk_dicom_meta_tag(sitk_img=sitk_img, tag="0018|1152", tag_type="float", default=np.nan)
#
#         # Modality
#         df_meta.ix[0, "modality"] = get_sitk_dicom_meta_tag(sitk_img=sitk_img, tag="0008|0060", tag_type="str", default="NA")
#
#         # Manufacturer
#         df_meta.ix[0, "manufacturer"] = get_sitk_dicom_meta_tag(sitk_img=sitk_img, tag="0008|0070", tag_type="str", default="NA")
#
#         # Scanner model
#         df_meta.ix[0, "scanner_model"] = get_sitk_dicom_meta_tag(sitk_img=sitk_img, tag="0008|1090", tag_type="str", default="NA")
#
#         # Tube potential in kVP
#         df_meta.ix[0, "tube_potential"] = get_sitk_dicom_meta_tag(sitk_img=sitk_img, tag="0018|0060", tag_type="float", default=np.nan)
#
#         # Reconstruction kernel
#         df_meta.ix[0, "reconstruction_kernel"] = get_sitk_dicom_meta_tag(sitk_img=sitk_img, tag="0018|1210", tag_type="str", default="NA")
#
#         # Reconstructed slice thickness in mm
#         df_meta.ix[0, "slice_thickness"] = get_sitk_dicom_meta_tag(sitk_img=sitk_img, tag="0018|0050", tag_type="float", default=np.nan)
#
#         # Pixel spacing (columns)
#         try:    df_meta.ix[0, "pixel_spacing_x"] = sitk_img.GetSpacing()[0]
#         except: pass
#
#         # Pixel spacing (rows)
#         try:    df_meta.ix[0, "pixel_spacing_y"] = sitk_img.GetSpacing()[1]
#         except: pass
#
#         # Tube current in mA
#         if tube_current > 0.0:
#             df_meta.ix[0, "mean_tube_current"] = tube_current / n_slices
#
#         # Mean exposure time per slice in milliseconds
#         if exposure_time > 0.0:
#             df_meta.ix[0, "mean_exposure_time"] = exposure_time / n_slices
#
#         # Radiation exposure in mAs
#         if exposure > 0.0:
#             df_meta.ix[0, "mean_exposure"] = exposure / n_slices
#
#         return df_meta
#
#     meta_list = [getCTMetaTags(df_char, roi_name="none")]
#
#     # Iterate over rois (if present)
#     if len(roi_list) > 0:
#         for roi_obj in roi_list:
#
#             # Determine z-positions corresponding to slices contain roi voxels
#             z_ind, y_ind, x_ind = np.where(roi_obj.roi.get_voxel_grid())
#             z_ind = np.unique(z_ind)
#             roi_pos_z = np.around(roi_obj.roi.slice_z_pos[z_ind], 5)
#             img_pos_z = np.around(df_char.pos_z.values.astype(np.float32), 5)
#
#             # Select the slices that cover the ROI
#             df_char_roi = copy.deepcopy(df_char)
#             df_char_roi = df_char_roi.loc[np.isin(img_pos_z, roi_pos_z), ].reset_index(drop=True)
#
#             # Determine imaging parameters
#             df_roi_meta = getCTMetaTags(df_char_roi, roi_name=roi_obj.name)
#
#             # Determine (approximate) volume of the ROI
#             df_roi_meta.ix[0, "volume"] = np.sum(roi_obj.roi.get_voxel_grid()) * np.prod(roi_obj.roi.spacing)
#
#             # Add to the list
#             meta_list.append(df_roi_meta)
#
#     # Cast to data frame
#     df_meta = pd.concat(meta_list)
#
#     return df_meta


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
