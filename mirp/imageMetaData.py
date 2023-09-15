import datetime
import random

import numpy as np
import pandas as pd
import pydicom
from pydicom import FileDataset

from mirp.importData.utilities import parse_image_correction, convert_dicom_time, get_pydicom_meta_tag


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
