import os
import warnings
from copy import deepcopy
from typing import Union

import pydicom
import pandas as pd
import numpy as np

from mirp.imageSUV import SUVscalingObj
from mirp.imageMetaData import get_pydicom_meta_tag, has_pydicom_meta_tag
from mirp.imageClass import ImageClass
from mirp.roiClass import RoiClass
from pydicom.tag import Tag
from pydicom import FileDataset

from utilities import parse_roi_name


def read_dicom_image_series(image_folder, modality=None, series_uid=None):

    # Obtain a list with image files
    file_list = _find_dicom_image_series(image_folder=image_folder, allowed_modalities=["CT", "PT", "MR"],
                                         modality=modality, series_uid=series_uid)

    # Obtain slice positions for each file
    image_position_z = []
    for file_name in file_list:

        # Read DICOM header
        dcm = pydicom.dcmread(os.path.join(image_folder, file_name), stop_before_pixels=True, force=True,
                              specific_tags=[Tag(0x0020, 0x0032)])

        # Obtain the z position
        image_position_z += [get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x0032), tag_type="mult_float")[2]]

    # Order ascending position (DICOM: z increases from feet to head)
    file_table = pd.DataFrame({"file_name": file_list, "position_z": image_position_z}).sort_values(by="position_z")

    # Obtain DICOM metadata from the bottom slice. This will be used to fill out all the different details.
    dcm = pydicom.dcmread(os.path.join(image_folder, file_table.file_name.values[0]), stop_before_pixels=True, force=True)

    # Find the number of rows (y) and columns (x) in the data set.
    n_x = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0028, 0x011), tag_type="int")
    n_y = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0028, 0x010), tag_type="int")

    # Create an empty voxel grid. Use z, y, x ordering for consistency within MIRP.
    voxel_grid = np.zeros((len(file_table), n_y, n_x), dtype=np.float32)

    # Read all dicom slices in order.
    slice_dcm_list = [pydicom.dcmread(os.path.join(image_folder, file_name), stop_before_pixels=False, force=True) for file_name in file_table.file_name.values]

    # Iterate over the different slices to fill out the voxel_grid.
    for ii, file_name in enumerate(file_table.file_name.values):

        # Read the dicom file and extract the slice grid
        slice_dcm = slice_dcm_list[ii]
        slice_grid = slice_dcm.pixel_array.astype(np.float32)

        # Update with scale and intercept. These may change per slice.
        rescale_intercept = get_pydicom_meta_tag(dcm_seq=slice_dcm, tag=(0x0028, 0x1052), tag_type="float", default=0.0)
        rescale_slope = get_pydicom_meta_tag(dcm_seq=slice_dcm, tag=(0x0028, 0x1053), tag_type="float", default=1.0)
        slice_grid = slice_grid * rescale_slope + rescale_intercept

        # Convert all images to SUV at admin
        if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str") == "PT":
            suv_conversion_object = SUVscalingObj(dcm=slice_dcm)
            scale_factor = suv_conversion_object.get_scale_factor(suv_normalisation="bw")

            # Convert to SUV
            slice_grid *= scale_factor

            # Update the DICOM header
            slice_dcm = suv_conversion_object.update_dicom_header(dcm=slice_dcm)

        # Store in voxel grid
        voxel_grid[ii, :, :] = slice_grid

    # Obtain the image origin from the dicom header (note: z, y, x order)
    image_origin = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x0032), tag_type="mult_float", default=np.array([0.0, 0.0, 0.0]))[::-1]

    # Obtain the image spacing from the dicom header and slice positions.
    image_pixel_spacing = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0028, 0x0030), tag_type="mult_float")
    image_slice_thickness = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0050), tag_type="float", default=None)

    if len(file_table) > 1:
        # Slice spacing can be determined from the slice positions
        image_slice_spacing = np.median(np.abs(np.diff(file_table.position_z.values)))

        if image_slice_thickness is None:
            # TODO: Update slice thickness tag in dcm
            pass
        else:
            # Warn the user if there is a mismatch between slice thickness and the actual slice spacing.
            if not np.around(image_slice_thickness - image_slice_spacing, decimals=5) == 0.0:
                warnings.warn(f"Mismatch between slice thickness ({image_slice_thickness}) and actual slice spacing ({image_slice_spacing}). The actual slice spacing will be "
                              f"used.", UserWarning)

    elif image_slice_thickness is not None:
        # There is only one slice, and we use the slice thickness as parameter.
        image_slice_spacing = image_slice_thickness

    else:
        # There is only one slice and the slice thickness is unknown. In this situation, we use the pixel spacing
        image_slice_spacing = np.max(image_pixel_spacing)

    # Combine pixel spacing and slice spacing into the voxel spacing, using z, y, x order.
    image_spacing = np.array([image_slice_spacing, image_pixel_spacing[1], image_pixel_spacing[0]])

    # Obtain image orientation and add the 3rd dimension
    image_orientation = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x0037), tag_type="mult_float")
    image_orientation += [0.0, 0.0, 1.0]

    # Revert to z, y, x order
    image_orientation = image_orientation[::-1]

    # Create an ImageClass object and store dicom meta-data
    img_obj = ImageClass(voxel_grid=voxel_grid,
                         origin=image_origin,
                         spacing=image_spacing,
                         slice_z_pos=file_table.position_z.values,
                         orientation=image_orientation,
                         modality=get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str"),
                         spat_transform="base",
                         no_image=False,
                         metadata=slice_dcm_list[0],
                         metadata_sop_instances=[get_pydicom_meta_tag(dcm_seq=slice_dcm, tag=(0x0008, 0x0018), tag_type="str") for slice_dcm in slice_dcm_list])

    return img_obj


def read_dicom_rt_struct(dcm_folder, image_object: Union[ImageClass, None] = None, series_uid=None, roi=None, frame_of_ref_uid=None):

    # Parse to list
    if roi is not None:
        if isinstance(roi, str):
            roi = [roi]

    # If the user does not specify a frame of reference UID, we attempt to read the frame of reference UID from the
    # image. These UIDs should be the same if the segmentation is to be realiably registered. The Series UID may
    # be the same as well, but not necessarily.
    if frame_of_ref_uid is None and image_object is not None:
        if image_object.metadata is not None:
            frame_of_ref_uid = image_object.get_metadata(tag=(0x0020, 0x0052), tag_type="str")

    # Find a list of RTSTRUCT files
    file_list = _find_dicom_image_series(image_folder=dcm_folder, allowed_modalities=["RTSTRUCT"],
                                         modality="RTSTRUCT", series_uid=series_uid, frame_of_ref_uid=frame_of_ref_uid)

    # Obtain DICOM metadata from RT structure set.
    dcm_file = os.path.join(dcm_folder, file_list[0])
    dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True, force=True)

    # Obtain ROI names and numbers
    roi_names, roi_numbers = _find_dicom_roi_names(dcm=dcm, with_roi_number=True)

    # Check rois and roi names
    if roi is None:
        # No user-defined rois
        roi = roi_names

    # Deparse rois and flatten
    deparsed_roi = [parse_roi_name(roi=current_roi) for current_roi in roi]
    deparsed_roi = [deparsed_roi_name for combined_roi_name in deparsed_roi for deparsed_roi_name in combined_roi_name]

    # Check if all are included in roi_names
    missing_roi = np.setdiff1d(ar1=np.array(deparsed_roi), ar2=np.array(roi_names)).tolist()
    if len(missing_roi) == len(deparsed_roi):
        warnings.warn(f"None of the ROIs could be found in the RT structure set ({dcm_file}).")
        return []

    elif len(missing_roi) > 0:
        warnings.warn(f"Some ROIs could not be found in the RT structure set ({dcm_file}): {missing_roi}")

    # Only keep those entries in the dicom data that contain any of the requested rois
    requested_roi_numbers = [roi_number for ii, roi_number in enumerate(roi_numbers) if roi_names[ii] in deparsed_roi]
    dcm = _filter_rt_structure_set(dcm=dcm, roi_numbers=requested_roi_numbers)

    # Check if there is an input image to recreate the voxel-based imaging.
    if image_object is None:
        return dcm

    else:
        roi_list = [_convert_rtstruct_to_segmentation(dcm=dcm, roi=current_roi, image_object=image_object) for current_roi in roi]
        return [roi_obj for roi_obj in roi_list if roi_obj is not None]


def read_roi_names(dcm_folder):

    # Placeholder list
    roi_names = []
    file_name_list = []

    # Find a list of RTSTRUCT files
    file_list = _find_dicom_image_series(image_folder=dcm_folder, allowed_modalities=["RTSTRUCT"],
                                         modality="RTSTRUCT", series_uid=None, frame_of_ref_uid=None)

    # Open DICOM files and read names
    for file_name in file_list:
        dcm_file = os.path.join(dcm_folder, file_name)
        dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True, force=True)

        # Obtain ROI names
        roi_names += _find_dicom_roi_names(dcm=dcm, with_roi_number=False)
        file_name_list += [file_name] * len(roi_names)

    return file_name_list, roi_names


def _find_dicom_image_series(image_folder, allowed_modalities, modality=None, series_uid=None, frame_of_ref_uid=None):

    # Check folder contents, keep only files that are recognised as DICOM images.
    file_list = os.listdir(image_folder)
    file_list = [file_name for file_name in file_list if not os.path.isdir(os.path.join(image_folder, file_name))]
    file_list = [file_name for file_name in file_list if file_name.lower().endswith(".dcm")]

    if len(file_list) == 0:
        raise FileNotFoundError(f"The image folder does not contain any DICOM files.")

    # Modality and series UID
    series_modality = []
    series_series_uid = []
    series_FOR_uid = []

    # Identify modality of the files
    for file_name in file_list:
        # Read DICOM header using pydicom
        dcm = pydicom.dcmread(os.path.join(image_folder, file_name), stop_before_pixels=True, force=True)

        # Read modality
        series_modality += [get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str")]

        # Read series UID
        series_series_uid += [get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x000e), tag_type="str")]

        # Frame of reference UID
        series_FOR_uid += [_get_frame_of_reference_uid(dcm=dcm)]

    # User-provided modality
    if modality is not None:

        if modality.lower() in ["ct"]:
            requested_modality = ["CT"]
        elif modality.lower() in ["pet", "pt"]:
            requested_modality = ["PT"]
        elif modality.lower() in ["mri", "mr"]:
            requested_modality = ["MR"]
        elif modality.lower() in ["rtstruct", "structure_set"]:
            requested_modality = ["RTSTRUCT"]
        else:
            raise ValueError(f"Unknown modality requested. Available choices are CT, PT, MR, RTSTRUCT. Found: {modality}")

        if not any([tmp_modality in allowed_modalities for tmp_modality in requested_modality]):
            raise ValueError(f"The selected modality ({modality}) cannot be used within the current context. This error can occur when attempting to image files instead of"
                             "segmentations when segmentations are intended.")

    else:
        # Any supported modality
        requested_modality = allowed_modalities

    # Filter file list
    file_list = [file_list[ii] for ii, file_modality in enumerate(series_modality) if file_modality in requested_modality]
    series_series_uid = [series_series_uid[ii] for ii, file_modality in enumerate(series_modality) if file_modality in requested_modality]
    series_FOR_uid = [series_FOR_uid[ii] for ii, file_modality in enumerate(series_modality) if file_modality in requested_modality]

    # Check if the requested modality was found.
    if len(file_list) == 0:
        raise ValueError(f"The DICOM folder does not contain any DICOM images with a currently supported modality ({allowed_modalities}).")

    # Check uniqueness of series UID
    if len(list(set(series_series_uid))) > 1 and series_uid is None and frame_of_ref_uid is None:
        raise ValueError(f"Multiple series UID were found in the DICOM folder. Please select one using the series_uid argument. Found: {list(set(series_series_uid))}")

    elif len(list(set(series_series_uid))) > 1 and series_uid is None and frame_of_ref_uid is not None:
        series_uid = [series_series_uid[ii] for ii, file_FOR_uid in enumerate(series_FOR_uid) if file_FOR_uid == frame_of_ref_uid]

        if len(list(set(series_uid))) > 1:
            raise ValueError(f"Multiple series UID that share a frame of reference UID were found in the DICOM folder. Please select one using the series_uid argument."
                             f"Found: {list(set(series_uid))}")
        else:
            series_uid = series_uid[0]

    elif series_uid is not None:

        # Check if the series_uid exists
        if series_uid not in series_series_uid:
            raise ValueError(f"The requested series UID ({series_uid}) was not found in the DICOM folder. Found: {list(set(series_series_uid))}")

    else:
        series_uid = series_series_uid[0]

    # Check uniqueness of FOR_uid
    if len(list(set(series_FOR_uid))) > 1 and frame_of_ref_uid is None and series_uid is None:
        raise ValueError(f"Multiple series with different frame of reference UIDs were found in the DICOM folder.")

    elif len(list(set(series_FOR_uid))) > 1 and frame_of_ref_uid is None and series_uid is not None:
        frame_of_ref_uid = [series_FOR_uid[ii] for ii, file_series_uid in enumerate(series_series_uid) if file_series_uid == series_uid]

        if len(list(set(frame_of_ref_uid))) > 1:
            raise ValueError(f"Multiple frame of reference UIDs where found for the same DICOM series. This may indicate corruption of the DICOM files.")
        else:
            frame_of_ref_uid = frame_of_ref_uid[0]

    elif frame_of_ref_uid is not None:

        if frame_of_ref_uid not in series_FOR_uid:
            raise ValueError(f"The requested frame of reference UID ({frame_of_ref_uid}) was not found in the DICOM folder. Found: {list(set(series_FOR_uid))}")

    else:
        frame_of_ref_uid = series_FOR_uid[0]

    # Filter series with the particular frame of reference uid and series uid
    return [file_list[ii] for ii in range(len(file_list)) if series_series_uid[ii] == series_uid and series_FOR_uid[ii] == frame_of_ref_uid]


def _get_frame_of_reference_uid(dcm):

    # Try to obtain a frame of reference UID
    if has_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x0052)):
        return get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x0052), tag_type="str")

    # For RT structure sets, the FOR UID may be tucked away in the Structure Set ROI Sequence
    if has_pydicom_meta_tag(dcm_seq=dcm, tag=(0x3006, 0x0020)):
        structure_set_roi_sequence = dcm[(0x3006, 0x0020)]

        for structure_set_roi_element in structure_set_roi_sequence:
            if has_pydicom_meta_tag(dcm_seq=structure_set_roi_element, tag=(0x3006, 0x0024)):
                return get_pydicom_meta_tag(dcm_seq=structure_set_roi_element, tag=(0x3006, 0x0024), tag_type="str")

    return None


def _find_dicom_roi_names(dcm, with_roi_number=False):

    # Placeholder roi_names list
    roi_names = []
    roi_sequence_numbers = []

    # Check if a Structure Set ROI Sequence exists (3006, 0020)
    if not get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x3006, 0x0020), test_tag=True):
        warnings.warn("The RT-structure set did not contain any ROI sequences.")

        if with_roi_number:
            return roi_names, roi_sequence_numbers
        else:
            return roi_names

    # Iterate over roi elements in the roi sequence
    for roi_element in dcm[0x3006, 0x0020]:

        # Check if the roi element contains a name (3006, 0026)
        if get_pydicom_meta_tag(dcm_seq=roi_element, tag=(0x3006, 0x0026), test_tag=True):
            roi_names += [get_pydicom_meta_tag(dcm_seq=roi_element, tag=(0x3006, 0x0026), tag_type="str")]
            roi_sequence_numbers += [get_pydicom_meta_tag(dcm_seq=roi_element, tag=(0x3006, 0x0022), tag_type="str")]

    if with_roi_number:
        return roi_names, roi_sequence_numbers
    else:
        return roi_names


def _filter_rt_structure_set(dcm, roi_numbers=None, roi_names=None):

    from pydicom.sequence import Sequence

    # We need to update a few sequences in the RT structure set:
    # * The Structure Set ROI sequence (3006, 0020)
    # * The ROI Contour sequence (3006, 0039)
    # * The RT ROI Observations Sequence (3006, 0080)

    # Initialise new sequences
    new_structure_set_roi_sequence = Sequence()
    new_roi_contour_sequence = Sequence()
    new_rt_roi_observations_sequence = Sequence() if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x3006, 0x0080), test_tag=True) else None

    if not get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x3006, 0x0020), test_tag=True):
        return None

    if not get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x3006, 0x0039), test_tag=True):
        return None

    # Check that either roi_numbers is provided, or roi_names is provided.
    if roi_numbers is None and roi_names is None:
        raise ValueError("Either the ROI Reference numbers or the ROI names should be provided.")

    elif roi_numbers is None and roi_names is not None:
        roi_names_available, roi_numbers_available = _find_dicom_roi_names(dcm=dcm, with_roi_number=True)
        roi_numbers = [roi_number for ii, roi_number in enumerate(roi_numbers_available) if roi_names_available[ii] in roi_names]

    if len(roi_numbers) == 0:
        return None

    for ii, current_roi_number in enumerate(roi_numbers):

        # Look through the existing Structure Set ROI sequence for matching roi_number
        for structure_set_elem in dcm[(0x3006, 0x0020)]:
            if not get_pydicom_meta_tag(dcm_seq=structure_set_elem, tag=(0x3006, 0x0022), test_tag=True):
                # No ROI number present.
                continue

            elif get_pydicom_meta_tag(dcm_seq=structure_set_elem, tag=(0x3006, 0x0022), tag_type="str") == current_roi_number:
                # ROI number matches
                new_structure_set_roi_sequence.append(structure_set_elem)

            else:
                continue

        # Look through existing ROI contour sequence for matching roi_number
        for roi_contour_elem in dcm[(0x3006, 0x0039)]:
            if not get_pydicom_meta_tag(dcm_seq=roi_contour_elem, tag=(0x3006, 0x0084), test_tag=True):
                # No ROI number present
                continue

            elif get_pydicom_meta_tag(dcm_seq=roi_contour_elem, tag=(0x3006, 0x0084), tag_type="str") == current_roi_number:
                # ROI number matches
                new_roi_contour_sequence.append(roi_contour_elem)

            else:
                continue

        if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x3006, 0x0080), test_tag=True):
            for roi_observation_elem in dcm[(0x3006, 0x0080)]:

                if not get_pydicom_meta_tag(dcm_seq=roi_observation_elem, tag=(0x3006, 0x0084), tag_type="str"):
                    # No ROI number present
                    continue

                elif get_pydicom_meta_tag(dcm_seq=roi_observation_elem, tag=(0x3006, 0x0084), tag_type="str") == current_roi_number:
                    # ROI number matches
                    new_rt_roi_observations_sequence.append(roi_observation_elem)

                else:
                    continue

    # Local copy of dcm
    dcm = deepcopy(dcm)

    # Add as data element
    if new_structure_set_roi_sequence is not None:
        dcm[(0x3006, 0x0020)].value = new_structure_set_roi_sequence

    if new_roi_contour_sequence is not None:
        dcm[(0x3006, 0x0039)].value = new_roi_contour_sequence

    if new_rt_roi_observations_sequence is not None:
        dcm[(0x3006, 0x0080)].value = new_rt_roi_observations_sequence

    return dcm


def _convert_rtstruct_to_segmentation(dcm: FileDataset, roi: str, image_object: ImageClass):

    # Deparse roi
    deparsed_roi = parse_roi_name(roi=roi)

    # Keep only structure data corresponding to the current ROIs.
    dcm = _filter_rt_structure_set(dcm=dcm, roi_names=deparsed_roi)

    # Check if the roi is found.
    if dcm is None:
        return None

    # Initialise a data list
    contour_data_list = []

    # Obtain segmentation
    for roi_contour_sequence in dcm[(0x3006, 0x0039)]:
        for contour_sequence in roi_contour_sequence[(0x3006, 0x0040)]:

            # Check if the geometric type exists (3006, 0042)
            if not get_pydicom_meta_tag(dcm_seq=contour_sequence, tag=(0x3006, 0x0042), test_tag=True):
                continue

            # Check if the geometric type equals "CLOSED_PLANAR"
            if get_pydicom_meta_tag(dcm_seq=contour_sequence, tag=(0x3006, 0x0042), tag_type="str") != "CLOSED_PLANAR":
                continue

            # Check if contour data exists (3006, 0050)
            if not get_pydicom_meta_tag(dcm_seq=contour_sequence, tag=(0x3006, 0x0050), test_tag=True):
                continue

            contour_data = np.array(get_pydicom_meta_tag(dcm_seq=contour_sequence, tag=(0x3006, 0x0050), tag_type="mult_float"), dtype=np.float64)
            contour_data = contour_data.reshape((-1, 3))

            # Determine if there is an offset (3006, 0045)
            contour_offset = np.array(get_pydicom_meta_tag(dcm_seq=contour_sequence, tag=(0x3006, 0x0045), tag_type="mult_float", default=[0.0, 0.0, 0.0]),
                                      dtype=np.float64)

            # Remove the offset from the data
            contour_data -= contour_offset

            # Add contour data to the contour list
            contour_data_list += [contour_data]

    if len(contour_data_list) > 0:
        # Create a new ROI object.
        roi_obj = RoiClass(name="+".join(deparsed_roi), contour=contour_data_list, metadata=dcm)

        # Convert contour into segmentation object
        roi_obj.create_mask_from_contours(img_obj=image_object, disconnected_segments="keep_as_is")

        return roi_obj

    else:
        return None
