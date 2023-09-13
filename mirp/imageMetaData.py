import datetime
import logging
import os
import random
from collections.abc import Iterable
from typing import Union

import numpy as np
import pandas as pd
import pydicom
from pydicom import FileDataset, datadict, Dataset
from pydicom.tag import Tag

from mirp.importData.utilities import parse_image_correction, convert_dicom_time, get_pydicom_meta_tag


def get_image_directory_meta_data(image_folder, subject):

    # Get image files
    image_file_list = [file_name for file_name in os.listdir(image_folder) if file_name.lower().endswith((".dcm", ".ima", ".nii", ".nii.gz", ".nrrd"))]

    meta_collect_list = []

    # Iterate over image files
    for image_file in image_file_list:

        # Determine image type
        image_file_type = get_image_type(image_file=os.path.join(image_folder, image_file))

        # Load dicom file
        if image_file_type == "dicom":
            # Avoid corrupted headers
            try:
                dcm = pydicom.dcmread(os.path.join(image_folder, image_file), stop_before_pixels=True, force=True)
            except OSError:
                logging.warning("DICOM header could not be read for %s.", os.path.join(image_folder, image_file))
                dcm = None
        else:
            dcm = None

        # Add file_path
        image_meta_list = [pd.Series({"file_path": os.path.join(image_folder, image_file)})]

        image_meta_list += [get_series_meta_data(dcm=dcm, shorten_uid=True)]
        image_meta_list += [get_basic_image_meta_data(dcm=dcm)]
        image_meta_list += [get_basic_ct_meta_data(dcm=dcm)]
        image_meta_list += [get_basic_pet_meta_data(dcm=dcm)]
        image_meta_list += [get_basic_mr_meta_data(dcm=dcm)]

        meta_collect_list += [pd.concat(image_meta_list, axis=0)]

    # Empty lists cannot be concatenated
    if len(meta_collect_list) > 0:
        df_meta = pd.concat(meta_collect_list, axis=1).T
    else:
        df_meta = []

    return df_meta


def get_meta_data(modality, dcm_list=None, image_folder=None):

    # Import locally to prevent circular references.
    from mirp.imageRead import get_all_dicom_headers

    if dcm_list is None and image_folder is None:
        raise ValueError("One of dcm_list and image_folder parameters needs to be provided.")

    elif dcm_list is None:
        dcm_list = get_all_dicom_headers(image_folder=image_folder, modality=modality)

    if len(dcm_list) == 0:
        return

    # Collect general information
    image_meta_list = [get_series_meta_data(dcm=dcm_list[0], shorten_uid=True)]
    image_meta_list += [get_basic_image_meta_data(dcm=dcm_list[0])]

    if modality == "CT":
        # Read basic meta data for CT imaging
        image_meta_list += [get_basic_ct_meta_data(dcm=dcm_list[0])]

        # Read advanced meta data for CT imaging per slice, then average
        adv_meta = [get_advanced_ct_meta_data(dcm=dcm) for dcm in dcm_list]
        image_meta_list += [pd.concat(adv_meta, axis=1).T.mean()]

    elif modality == "PT":
        image_meta_list += [get_basic_pet_meta_data(dcm=dcm_list[0])]
        image_meta_list += [get_advanced_pet_meta_data(dcm=dcm_list[0])]
    elif modality == "MR":
        image_meta_list += [get_basic_mr_meta_data(dcm=dcm_list[0])]

    return pd.concat(image_meta_list, axis=0)


def set_pydicom_meta_tag(dcm_seq: Union[FileDataset, Dataset], tag, value, force_vr=None):
    # Check tag
    if isinstance(tag, tuple):
        tag = Tag(tag[0], tag[1])

    elif isinstance(tag, list):
        tag = Tag(tag[0], tag[2])

    elif isinstance(tag, Tag):
        pass

    else:
        raise TypeError(f"Metadata tag {tag} is not a pydicom Tag, or can be parsed to one.")

    # Read the default VR information for non-existent tags.
    vr, vm, name, is_retired, keyword = datadict.get_entry(tag)

    if vr == "DS":
        # Decimal string (16-byte string representing decimal value)
        if isinstance(value, Iterable):
            value = [f"{x:.16f}"[:16] for x in value]
        else:
            value = f"{value:.16f}"[:16]

    if tag in dcm_seq and force_vr is None:
        # Update the value of an existing tag.
        dcm_seq[tag].value = value

    elif force_vr is None:
        # Add a new entry.
        dcm_seq.add_new(tag=tag, VR=vr, value=value)

    else:
        # Add a new entry
        dcm_seq.add_new(tag=tag, VR=force_vr, value=value)


def get_itk_dicom_meta_tag(itk_img, tag, tag_type, default=None):
    # Reads dicom tag

    # Initialise with default
    tag_value = default

    # Read from header using simple itk
    try:
        tag_value = itk_img.GetMetaData(tag)
    except:
        pass

    # Find empty entries
    if tag_value is not None:
        if tag_value == "":
            tag_value = default

    # Cast to correct type (meta tags are usually passed as strings)
    if tag_value is not None:

        # String
        if tag_type == "str":
            tag_value = str(tag_value)

        # Float
        elif tag_type == "float":
            tag_value = float(tag_value)

        # Integer
        elif tag_type == "int":
            tag_value = int(tag_value)

        # Boolean
        elif tag_type == "bool":
            tag_value = bool(tag_value)

    return tag_value


def get_series_meta_data(image_file=None, dcm=None, shorten_uid=False):

    if image_file is not None:
        # Determine image type
        image_file_type = get_image_type(image_file)

        if image_file_type == "dicom":
            # Load dicom file
            dcm = pydicom.dcmread(image_file, stop_before_pixels=True, force=True)
        else:
            dcm = None

    if dcm is not None:

        # Read study instance UID
        study_instance_uid = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x000D), tag_type="str", default="")

        # Read series instance UID
        series_instance_uid = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x000E), tag_type="str", default="")

        # Frame of reference UID
        if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x3006, 0x010), test_tag=True):
            frame_of_reference_uid = get_pydicom_meta_tag(dcm_seq=dcm[0x3006, 0x010][0], tag=(0x0020, 0x0052), tag_type="str", default="")
        else:
            frame_of_reference_uid = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x0052), tag_type="str", default="")

        # Read patient name
        patient_name = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0010, 0x0010), tag_type="str", default="")

        # Read study description
        study_description = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x1030), tag_type="str", default="")

        # Read series description
        series_description = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x103E), tag_type="str", default="")

        # Examined body part
        body_part_examined = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0015), tag_type="str", default="")

        # Read study date and time
        study_start_date = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0020), tag_type="str")
        study_start_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0030), tag_type="str", default="")

        # Read acquisition date and time
        acquisition_start_date = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0022), tag_type="str")
        acquisition_start_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0032), tag_type="str", default="")

        if shorten_uid:
            study_instance_uid = study_instance_uid[-6:]
            series_instance_uid = series_instance_uid[-6:]
            frame_of_reference_uid = frame_of_reference_uid[-6:]

        meta_data = pd.Series({"patient_name": patient_name,
                               "study_instance_uid": study_instance_uid,
                               "series_instance_uid": series_instance_uid,
                               "frame_of_reference_uid": frame_of_reference_uid,
                               "study_description": study_description,
                               "series_description": series_description,
                               "body_part_examined": body_part_examined,
                               "study_date": study_start_date,
                               "study_time": study_start_time,
                               "acquisition_date": acquisition_start_date,
                               "acquisition_time": acquisition_start_time})

    else:
        meta_data = pd.Series({"patient_name": "",
                               "study_instance_uid": "",
                               "series_instance_uid": "",
                               "frame_of_reference_uid": "",
                               "study_description": "",
                               "series_description": "",
                               "body_part_examined": "",
                               "study_date": "",
                               "study_time": "",
                               "acquisition_date": "",
                               "acquisition_time": ""})

    return meta_data


def get_basic_image_meta_data(image_file=None, dcm=None):

    if image_file is not None:
        # Determine image type
        image_file_type = get_image_type(image_file)

        if image_file_type == "dicom":
            # Load dicom file
            dcm = pydicom.dcmread(image_file, stop_before_pixels=True, force=True)
        else:
            dcm = None

    if dcm is not None:
        # Modality
        modality = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str")

        # Instance number
        instance_number = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0020, 0x0013), tag_type="int", default=-1)

        # Scanner type
        scanner_type = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x1090), tag_type="str", default="")

        # Scanner manufacturer
        manufacturer = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0070), tag_type="str", default="")

        # Slice thickness
        spacing_z = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0050), tag_type="float", default=np.nan)

        # Pixel spacing
        pixel_spacing = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0028, 0x0030), tag_type="mult_float")
        if pixel_spacing is not None:
            spacing_x = pixel_spacing[0]
            spacing_y = pixel_spacing[1]
        else:
            spacing_x = spacing_y = np.nan

        meta_data = pd.Series({"modality": modality,
                               "instance_number": instance_number,
                               "scanner_type": scanner_type,
                               "manufacturer": manufacturer,
                               "spacing_z": spacing_z,
                               "spacing_y": spacing_y,
                               "spacing_x": spacing_x})
    else:
        meta_data = pd.Series({"modality": "",
                               "instance_number": -1,
                               "scanner_type": "",
                               "manufacturer": "",
                               "spacing_z": np.nan,
                               "spacing_y": np.nan,
                               "spacing_x": np.nan})

    return meta_data


def get_basic_ct_meta_data(image_file=None, dcm=None):

    if image_file is not None:
        # Determine image type
        image_file_type = get_image_type(image_file)

        if image_file_type == "dicom":
            # Load dicom file
            dcm = pydicom.dcmread(image_file, stop_before_pixels=True, force=True)
        else:
            dcm = None

    meta_data = pd.Series({"image_type": "",
                           "kvp": np.nan,
                           "kernel": "",
                           "agent": ""})

    if dcm is not None:
        if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str") == "CT":

            # Image type
            image_type = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0008), tag_type="str", default="")

            # Peak kilo voltage output
            kvp = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0060), tag_type="float", default=np.nan)

            # Convolution kernel
            kernel = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x1210), tag_type="str", default="")

            # Contrast/bolus agent
            contrast_agent = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0010), tag_type="str", default="")

            meta_data = pd.Series({"image_type": image_type,
                                   "kvp": kvp,
                                   "kernel": kernel,
                                   "agent": contrast_agent})

    meta_data.index = "ct_" + meta_data.index

    return meta_data


def get_advanced_ct_meta_data(image_file=None, dcm=None):

    if image_file is not None:
        # Determine image type
        image_file_type = get_image_type(image_file)

        if image_file_type == "dicom":
            # Load dicom file
            dcm = pydicom.dcmread(image_file, stop_before_pixels=True, force=True)
        else:
            dcm = None

    meta_data = pd.Series({"tube_current": np.nan,
                           "exposure_time": np.nan,
                           "exposure": np.nan})

    if dcm is not None:
        if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str") == "CT":

            # Tube current in mA
            tube_current = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x1151), tag_type="float", default=np.nan)

            # Exposure time in milliseconds
            exposure_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x1150), tag_type="float", default=np.nan)

            # Radiation exposure in mAs
            exposure = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x1152), tag_type="float", default=np.nan)

            meta_data = pd.Series({"tube_current": tube_current,
                                   "exposure_time": exposure_time,
                                   "exposure": exposure})

    meta_data.index = "ct_" + meta_data.index

    return meta_data


def get_basic_pet_meta_data(image_file=None, dcm=None):
    if image_file is not None:
        # Determine image type
        image_file_type = get_image_type(image_file)

        if image_file_type == "dicom":
            # Load dicom file
            dcm = pydicom.dcmread(image_file, stop_before_pixels=True, force=True)
        else:
            dcm = None

    meta_data = pd.Series({"image_type": "",
                           "image_corrections": "",
                           "random_correction": "",
                           "attenuation_correction_method": "",
                           "scatter_correction_method": "",
                           "reconstruction_method": "",
                           "kernel": "",
                           "agent": ""})

    if dcm is not None:
        if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str") == "PT":

            # Image type
            image_type = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0008), tag_type="str", default="")

            # Image corrections
            image_corrections = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0028, 0x0051), tag_type="str", default="")

            # Randoms correction method
            random_correction_method = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x1100), tag_type="str", default="")

            # Attenuation correction method
            attenuation_correction_method = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x1101), tag_type="str", default="")

            # Scatter correction method
            scatter_correction_method = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x1105), tag_type="str", default="")

            # Reconstruction method
            reconstruction_method = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x1103), tag_type="str", default="")

            # Convolution kernel
            kernel = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x1210), tag_type="str", default="")

            # Radiopharmaceutical
            if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x0016), tag_type=None, test_tag=True):
                radiopharmaceutical = get_pydicom_meta_tag(dcm_seq=dcm[0x0054, 0x0016][0], tag=(0x0018, 0x0031), tag_type="str", default="")
            else:
                radiopharmaceutical = "unknown"

            meta_data = pd.Series({"image_type": image_type,
                                   "image_corrections": image_corrections,
                                   "random_correction": random_correction_method,
                                   "attenuation_correction": attenuation_correction_method,
                                   "scatter_correction": scatter_correction_method,
                                   "reconstruction_method": reconstruction_method,
                                   "kernel": kernel,
                                   "agent": radiopharmaceutical})

    meta_data.index = "pet_" + meta_data.index

    return meta_data


def get_advanced_pet_meta_data(image_file=None, dcm=None):
    if image_file is not None:
        # Determine image type
        image_file_type = get_image_type(image_file)

        if image_file_type == "dicom":
            # Load dicom file
            dcm = pydicom.dcmread(image_file, stop_before_pixels=True, force=True)
        else:
            dcm = None

    meta_data = pd.Series({"uptake_time": np.nan,
                           "frame_duration": np.nan,
                           "gender": "",
                           "weight": np.nan,
                           "height": np.nan,
                           "intensity_unit": "",
                           "decay_corr": "",
                           "attenuation_corr": "",
                           "scatter_corr": "",
                           "dead_time_corr": "",
                           "gantry_motion_corr": "",
                           "patient_motion_corr": "",
                           "count_loss_norm_corr": "",
                           "randoms_corr": "",
                           "radial_sampling_corr": "",
                           "sensitivy_calibr": "",
                           "detector_norm_corr": "",
                           "time_of_flight": "",
                           "reconstr_type": "",
                           "reconstr_algorithm": "",
                           "reconstr_is_iterative": "",
                           "n_iterations": np.nan,
                           "n_subsets": np.nan})

    if dcm is not None:
        if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str") == "PT":

            # Uptake time - acquisition start
            acquisition_ref_time = convert_dicom_time(date_str=get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0022), tag_type="str"),
                                                      time_str=get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0032), tag_type="str"))

            # Uptake time - administration (0018,1078) is the administration start DateTime
            if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x0016), test_tag=True):
                radio_admin_ref_time = convert_dicom_time(datetime_str=get_pydicom_meta_tag(dcm_seq=dcm[0x0054, 0x0016][0], tag=(0x0018, 0x1078), tag_type="str"))

                if radio_admin_ref_time is None:
                    # If unsuccessful, attempt determining administration time from (0x0018, 0x1072), which is the administration start time
                    radio_admin_ref_time = convert_dicom_time(date_str=get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0022), tag_type="str"),
                                                              time_str=get_pydicom_meta_tag(dcm_seq=dcm[0x0054, 0x0016][0], tag=(0x0018, 0x1072), tag_type="str"))
            else:
                radio_admin_ref_time = None

            if radio_admin_ref_time is None:
                # If neither (0x0018, 0x1078) or (0x0018, 0x1072) are present, attempt to read private tags.
                # GE tags - note that due to anonymisation, acquisition time may be different than reported.
                acquisition_ref_time = convert_dicom_time(
                    get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0009, 0x100d), tag_type="str"))
                radio_admin_ref_time = convert_dicom_time(
                    get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0009, 0x103b), tag_type="str"))

            if radio_admin_ref_time is not None and acquisition_ref_time is not None:

                day_diff = abs(radio_admin_ref_time - acquisition_ref_time).days
                if day_diff > 1:
                    # Correct for de-identification mistakes (i.e. administration time was de-identified correctly, but acquisition time not)
                    # We do not expect that the difference between the two is more than a day, or even more than a few hours at most.
                    if radio_admin_ref_time > acquisition_ref_time:
                        radio_admin_ref_time -= datetime.timedelta(days=day_diff)
                    else:
                        radio_admin_ref_time += datetime.timedelta(days=day_diff)

                if radio_admin_ref_time > acquisition_ref_time:
                    # Correct for overnight
                    radio_admin_ref_time -= datetime.timedelta(days=1)

                # Calculate uptake time in minutes
                uptake_time = ((acquisition_ref_time - radio_admin_ref_time).seconds / 60.0)
            else:
                uptake_time = np.nan

            # Frame duration (converted from milliseconds to seconds)
            frame_duration = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x1242), tag_type="float", default=np.nan) / 1000.0

            # Important data for SUV normalisation
            patient_gender = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0010, 0x0040), tag_type="str")
            patient_weight = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0010, 0x1030), tag_type="float")
            patient_height = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0010, 0x1020), tag_type="float")

            # The type of intensity represented by the pixels
            intensity_unit = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0054, 0x1001), tag_type="str")

            # Load image corrections for comparison in case correction tags are missing.
            image_corrections = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0028, 0x0051), tag_type="str", default="")
            image_corrections = image_corrections.replace(" ", "").replace("[", "").replace("]", "").replace("\'", "").split(sep=",")

            # Decay corrected DECY (0018,9758)
            decay_corrected = parse_image_correction(dcm_seq=dcm, tag=(0x0018, 0x9758), correction_abbr="DECY", image_corrections=image_corrections)

            # Attenuation corrected ATTN (0018,9759)
            attenuation_corrected = parse_image_correction(dcm_seq=dcm, tag=(0x0018, 0x9759), correction_abbr="ATTN", image_corrections=image_corrections)

            # Scatter corrected SCAT (0018,9760)
            scatter_corrected = parse_image_correction(dcm_seq=dcm, tag=(0x0018, 0x9760), correction_abbr="SCAT", image_corrections=image_corrections)

            # Dead time corrected DTIM (0018,9761)
            dead_time_corrected = parse_image_correction(dcm_seq=dcm, tag=(0x0018, 0x9761), correction_abbr="DTIM", image_corrections=image_corrections)

            # Gantry motion corrected MOTN (0018,9762)
            gantry_motion_corrected = parse_image_correction(dcm_seq=dcm, tag=(0x0018, 0x9762), correction_abbr="MOTN", image_corrections=image_corrections)

            # Patient motion corrected PMOT (0018,9763)
            patient_motion_corrected = parse_image_correction(dcm_seq=dcm, tag=(0x0018, 0x9763), correction_abbr="PMOT", image_corrections=image_corrections)

            # Count loss normalisation corrected CLN (0018,9764)
            count_loss_norm_corrected = parse_image_correction(dcm_seq=dcm, tag=(0x0018, 0x9764), correction_abbr="CLN", image_corrections=image_corrections)

            # Randoms corrected RAN (0018,9765)
            randoms_corrected = parse_image_correction(dcm_seq=dcm, tag=(0x0018, 0x9765), correction_abbr="RAN", image_corrections=image_corrections)

            # Non-uniform radial sampling corrected RADL (0018,9766)
            radl_corrected = parse_image_correction(dcm_seq=dcm, tag=(0x0018, 0x9766), correction_abbr="RADL", image_corrections=image_corrections)

            # Sensitivity calibrated DCAL (0018,9767)
            sensitivity_calibrated = parse_image_correction(dcm_seq=dcm, tag=(0x0018, 0x9767), correction_abbr="DCAL", image_corrections=image_corrections)

            # Detector normalisation correction NORM (0018,9768)
            detector_normalisation = parse_image_correction(dcm_seq=dcm, tag=(0x0018, 0x9768), correction_abbr="NORM", image_corrections=image_corrections)

            # Time of flight information (0018,9755)
            time_of_flight = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x9755), tag_type="str", default="")

            # Read reconstruction sequence (0018,9749)
            if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x9749), tag_type=None, test_tag=True):
                # Reconstruction type (0018,9749)(0018,9756)
                recon_type = get_pydicom_meta_tag(dcm_seq=dcm[0x0018, 0x9749][0], tag=(0x0018, 0x9756), tag_type="str", default="")

                # Reconstruction algorithm (0018,9749)(0018,9315)
                recon_algorithm = get_pydicom_meta_tag(dcm_seq=dcm[0x0018, 0x9749][0], tag=(0x0018, 0x9315), tag_type="str", default="")

                # Is an iterative method? (0018,9749)(0018,9769)
                is_iterative = get_pydicom_meta_tag(dcm_seq=dcm[0x0018, 0x9749][0], tag=(0x0018, 0x9769), tag_type="str", default="")

                # Number of iterations (0018,9749)(0018,9739)
                n_iterations = get_pydicom_meta_tag(dcm_seq=dcm[0x0018, 0x9749][0], tag=(0x0018, 0x9739), tag_type="int", default=np.nan)

                # Number of subsets (0018,9749)(0018,9740)
                n_subsets = get_pydicom_meta_tag(dcm_seq=dcm[0x0018, 0x9749][0], tag=(0x0018, 0x9740), tag_type="int", default=np.nan)
            else:
                recon_type = ""
                recon_algorithm = ""
                is_iterative = ""
                n_iterations = np.nan
                n_subsets = np.nan

            meta_data = pd.Series({"uptake_time": uptake_time,
                                   "frame_duration": frame_duration,
                                   "gender": patient_gender,
                                   "weight": patient_weight,
                                   "height": patient_height,
                                   "intensity_unit": intensity_unit,
                                   "decay_corr": decay_corrected,
                                   "attenuation_corr": attenuation_corrected,
                                   "scatter_corr": scatter_corrected,
                                   "dead_time_corr": dead_time_corrected,
                                   "gantry_motion_corr": gantry_motion_corrected,
                                   "patient_motion_corr": patient_motion_corrected,
                                   "count_loss_norm_corr": count_loss_norm_corrected,
                                   "randoms_corr": randoms_corrected,
                                   "radial_sampling_corr": radl_corrected,
                                   "sensitivy_calibr": sensitivity_calibrated,
                                   "detector_norm_corr": detector_normalisation,
                                   "time_of_flight": time_of_flight,
                                   "reconstr_type": recon_type,
                                   "reconstr_algorithm": recon_algorithm,
                                   "reconstr_is_iterative": is_iterative,
                                   "n_iterations": n_iterations,
                                   "n_subsets": n_subsets})

    meta_data.index = "pet_" + meta_data.index

    return meta_data


def get_basic_mr_meta_data(image_file=None, dcm=None):
    if image_file is not None:
        # Determine image type
        image_file_type = get_image_type(image_file)

        if image_file_type == "dicom":
            # Load dicom file
            dcm = pydicom.dcmread(image_file, stop_before_pixels=True, force=True)
        else:
            dcm = None

    meta_data = pd.Series({"image_type": "",
                           "scanning_sequence": "",
                           "scanning_sequence_variant": "",
                           "scanning_sequence_name": "",
                           "scan_options": "",
                           "acquisition_type": "",
                           "repetition_time": np.nan,
                           "echo_time": np.nan,
                           "echo_train_length": np.nan,
                           "inversion_time": np.nan,
                           "trigger_time": np.nan,
                           "magnetic_field_strength": np.nan,
                           "agent": ""})

    if dcm is not None:
        if get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0060), tag_type="str") == "MR":

            # Image type
            image_type = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0008, 0x0008), tag_type="str", default="")

            # Scanning sequence
            scanning_sequence = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0020), tag_type="str", default="")

            # Scanning sequence variant
            scanning_sequence_variant = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0021), tag_type="str", default="")

            # Sequence name
            scanning_sequence_name = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0024), tag_type="str", default="")

            # Scan options
            scan_options = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0022), tag_type="str", default="")

            # Acquisition type
            acquisition_type = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0023), tag_type="str", default="")

            # Repetition time
            repetition_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0080), tag_type="float", default=np.nan)

            # Echo time
            echo_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0081), tag_type="float", default=np.nan)

            # Echo train length
            echo_train_length = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0091), tag_type="float", default=np.nan)

            # Inversion time
            inversion_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0082), tag_type="float", default=np.nan)

            # Trigger time
            trigger_time = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x1060), tag_type="float", default=np.nan)

            # Contrast/bolus agent
            contrast_agent = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0010), tag_type="str", default=np.nan)

            # Magnetic field strength
            magnetic_field_strength = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x0018, 0x0087), tag_type="float", default=np.nan)

            meta_data = pd.Series({"image_type": image_type,
                                   "scanning_sequence": scanning_sequence,
                                   "scanning_sequence_variant": scanning_sequence_variant,
                                   "scanning_sequence_name": scanning_sequence_name,
                                   "scan_options": scan_options,
                                   "acquisition_type": acquisition_type,
                                   "repetition_time": repetition_time,
                                   "echo_time": echo_time,
                                   "echo_train_length": echo_train_length,
                                   "inversion_time": inversion_time,
                                   "trigger_time": trigger_time,
                                   "magnetic_field_strength": magnetic_field_strength,
                                   "agent": contrast_agent})

    meta_data.index = "mr_" + meta_data.index

    return meta_data


def get_image_type(image_file):
    # Determine image type
    if image_file.lower().endswith((".dcm", ".ima")):
        image_file_type = "dicom"
    elif image_file.lower().endswith((".nii", ".nii.gz")):
        image_file_type = "nifti"
    elif image_file.lower().endswith(".nrrd"):
        image_file_type = "nrrd"
    else:
        image_file_type = "unknown"

    return image_file_type


def create_new_uid(dcm: FileDataset):
    # Use series UID as the basis for generating new series and SOP instance UIDs
    series_uid = get_pydicom_meta_tag(dcm_seq=dcm, tag=(0x020, 0x000e), tag_type="str")

    # Set the minimum required length.
    min_req_len = 16

    # Determine the part of the series uid that should be removed
    split_series_uid = series_uid.split(".")
    s_length = np.array([len(s) for s in split_series_uid])
    s_cum_length = np.cumsum(s_length+1)
    s_cum_length[-1] -= 1

    # Strip parts of the string that are to be replaced
    s_keep = [s for ii, s in enumerate(split_series_uid) if s_cum_length[ii] < 64 - min_req_len]
    s_length = np.array([len(s) for s in s_keep])
    s_cum_length = np.cumsum(s_length + 1)
    s_cum_length[-1] -= 1

    # Generate new string
    available_length = 64 - s_cum_length[-1]

    # Initialise the randomiser
    random.seed()
    random_string = "".join([str(random.randint(1, 9)) for ii in range(available_length - 1)])
    s_keep += [random_string]

    new_uid = ".".join(s_keep)

    return new_uid
