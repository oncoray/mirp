import datetime
import os.path
import fnmatch
import math
from typing import Union, List, Tuple, Optional, Any, Iterable
from os.path import split

import numpy as np
import pydicom
from pydicom import FileDataset, Dataset, datadict
from pydicom.tag import Tag


def supported_image_modalities(modality: Union[None, str] = None) -> List[str]:

    if isinstance(modality, str):
        modality = modality.lower()

    if modality is None:
        return ["ct", "pt", "mr", "generic"]

    elif modality == "ct":
        return ["ct"]

    elif modality in ["pt", "pet"]:
        return ["pt"]

    elif modality in ["mr", "mri"]:
        return ["mr"]

    elif modality == "generic":
        return ["generic"]

    else:
        raise ValueError(
            f"Encountered an unknown image modality: {modality}. The following image modalities are supported: "
            f"{', '.join(supported_image_modalities(None))}. The generic modality lacks special default parameters, "
            f"and can always be used.")


def stacking_dicom_image_modalities() -> List[str]:
    return ["ct", "pt", "mr"]


def supported_mask_modalities(modality: Union[None, str] = None) -> List[str]:

    if isinstance(modality, str):
        modality = modality.lower()

    if modality is None:
        return ["rtstruct", "seg", "generic_mask"]

    elif modality in ["rtstruct"]:
        return ["rtstruct"]

    elif modality in ["seg"]:
        return ["seg"]

    elif modality == "generic_mask":
        return ["generic_mask"]

    else:
        raise ValueError(
            f"Encountered an unknown mask modality: {modality}. The following mask modalities are supported: "
            f"{', '.join(supported_mask_modalities(None))}. The generic modality can always be used.")


def supported_file_types(file_type: Union[None, str] = None) -> List[str]:

    if isinstance(file_type, str):
        modality = file_type.lower()

    if file_type is None:
        return [".dcm", ".nii", ".nii.gz", ".nrrd", ".npy"]

    elif file_type == "dicom":
        return [".dcm"]

    elif file_type == "itk":
        return [".nii", ".nii.gz", ".nrrd"]

    elif file_type == "nifti":
        return [".nii", ".nii.gz"]

    elif file_type == "nrrd":
        return [".nrrd"]

    elif file_type == "numpy":
        return [".npy"]

    else:
        raise ValueError(
            f"Encountered an unknown file type argument: {file_type}. The following file types are supported: "
            f"{', '.join(supported_file_types(None))}.")


def flatten_list(unflattened_list):

    if len(unflattened_list) == 0:
        return unflattened_list

    if isinstance(unflattened_list[0], list):
        return flatten_list(unflattened_list[0]) + flatten_list(unflattened_list[1:])

    return unflattened_list[:1] + flatten_list(unflattened_list[1:])


def bare_file_name(
        x: Union[str, List[str]],
        file_extension: Union[str, List[str]]
) -> Union[str, List[str]]:
    """
    Strips provided extensions from the name of a file.
    :param x: One or more filenames or path to file names.
    :param file_extension: One or more extensions that should be stripped
    :return: One or more filenames from which the extension has been stripped.
    """
    return_list = True
    if isinstance(x, str):
        x = [x]
        return_list = False

    if isinstance(file_extension, str):
        file_extension = [file_extension]

    file_name = [os.path.basename(file_path) for file_path in x]

    for ii, current_file_name in enumerate(file_name):
        for extension in file_extension:
            if current_file_name.endswith(extension):
                file_name[ii] = current_file_name.removesuffix(extension)

                # If a file extension is found, remove it only once -- we want to avoid accidentally stripping the
                # filename more than necessary.
                break

    if return_list:
        return file_name
    else:
        return file_name[0]


def match_file_name(
        x: Union[str, List[str]],
        pattern: Union[str, List[str]],
        file_extension: Union[None, str, List[str]]
) -> Union[bool, List[bool]]:
    """
    Determine if any filename matches the provided pattern. fnmatch is used for matching, which allows for wildcards.
    :param x: a string or path that is the filename or a path to the file.
    :param pattern: a string or list of strings that should be tested.
    :param file_extension: None, string or list of strings representing the file extension. If provided, the extension
    is stripped from the filename prior to matching.
    :return: a (list of) boolean value(s). True if any pattern appears in the file name, and False if not.
    """
    return_list = True
    if isinstance(x, str):
        x = [x]
        return_list = False

    if isinstance(pattern, str):
        pattern = [pattern]

    file_name = [os.path.basename(file_path) for file_path in x]

    if file_extension is not None:
        file_name = bare_file_name(file_name, file_extension=file_extension)

    matches = np.zeros(len(file_name), dtype=bool)
    for current_pattern in pattern:
        current_pattern = current_pattern.replace("#", "*")
        matches = np.logical_or(matches, np.array([
            fnmatch.fnmatch(current_file_name, current_pattern)
            for current_file_name in file_name
        ]))

    if return_list:
        return matches
    else:
        return any(matches)


def isolate_sample_name(
        x: str,
        pattern: str,
        file_extenstion: Union[None, str, List[str]]
) -> Union[None, str]:

    # Pattern should only contain one sample name placeholder (#).
    if pattern.count("#") != 1:
        return None

    x = bare_file_name(x, file_extension=file_extenstion)

    # Determine where the sample name placeholder is compared to other wildcards.
    central_split_id = 0
    for current_character in pattern:
        if current_character == "#":
            break
        elif current_character == "*":
            central_split_id += 1

    pattern = pattern.replace("#", "*")
    if not fnmatch.fnmatch(x, pattern):
        return None

    pattern_split = pattern.split("*")
    # Use the fixed (non-wildcard) characters to reduce the string. This is done by stripping away parts to the left
    # or right of fixed characters based on their position relative to the sample name placeholder.
    for ii in range(0, central_split_id + 1):
        if pattern_split[ii] == "":
            continue
        x = x.split(pattern_split[ii], 1)[1]

    for ii in reversed(range(central_split_id + 1, len(pattern_split))):
        if pattern_split[ii] == "":
            continue
        x = x.rsplit(pattern_split[ii], 1)[0]

    if x == "":
        return None

    return x


def path_to_parts(x: str) -> List[str]:
    """
    Split a path into its components.

    Parameters
    ---------
    x: str
        A string that represents a file path.

    Returns
    -------
    path_parts: list of str
        A list of path components.
    """

    path_parts = []
    x_head = x
    while True:
        x_head, x_tail = split(x_head)
        if x_tail == "":
            path_parts += [x_head]
            break
        path_parts += [x_tail]

    return list(reversed(path_parts))


def dir_structure_contains_directory(
        x: str,
        pattern: Union[str, List[str]],
        ignore_dir: Union[None, str, List[str]]
) -> bool:
    """
    Identify if a path contains a directory matches any of pattern.
    :param x: a string or a path.
    :param pattern: a pattern that should be fully matched in the path.
    :param ignore_dir: any (partial) path that should be ignored. These are stripped from x prior to pattern matching.
    :return: a boolean value.
    """
    # Split x into parts.
    x = path_to_parts(x)

    # Strip the pattern to be ignored from x, if possible.
    if ignore_dir is not None:
        if not isinstance(ignore_dir, list):
            ignore_dir = [ignore_dir]

        for current_ignore_dir in ignore_dir:
            current_ignore_dir = path_to_parts(current_ignore_dir)

            # Find matching sequential elements.
            match_index: List[Union[None, int]] = [None for ii in range(len(current_ignore_dir))]
            for jj, ignore_elem in enumerate(current_ignore_dir):
                if match_index[0] is None:
                    if jj > 0:
                        # Break if the first element is not found at all.
                        break
                    for ii in range(len(x)):
                        if x[ii] == ignore_elem:
                            match_index[0] = ii
                            break
                else:
                    if match_index[jj - 1] + 1 > len(x) - 1:
                        # Break if we would exceed the length of x.
                        break

                    if x[match_index[jj - 1] + 1] == ignore_elem:
                        match_index[jj] = match_index[jj - 1] + 1
                    else:
                        # Each element must be sequential.
                        break

            if not any(match_elem is None for match_elem in match_index):
                x = [x_elem for ii, x_elem in enumerate(x) if ii not in match_index]

    if len(x) == 0 or x == "":
        return False

    if not isinstance(pattern, list):
        pattern = [pattern]

    # Find if any pattern is exactly matched.
    return any(x_elem in pattern for x_elem in x)


def compute_file_distance(
        x: Union[None, str],
        y: Union[None, str]
):
    if x is None or y is None:
        return math.inf

    try:
        common_path = os.path.commonpath([x, y])
    except ValueError:
        return math.inf

    # Split to
    split_common_path = path_to_parts(common_path)
    split_x = path_to_parts(x)

    return len(split_x) - len(split_common_path)


def parse_image_correction(
        dcm_seq: pydicom.Dataset,
        tag: Tuple[hex, hex],
        correction_abbr: str
) -> bool:
    """
    Parses image correction information. Indicates whether a specific type of PET correction was applied based on
    available information.

    Parameters
    ----------
    dcm_seq: pydicom.Dataset
        Set of DICOM metadata.

    tag: tag for image correction
        Tag for reading image correction from the enhanced PET set of tags.

    correction_abbr:
        Abbreviation for the specific correction in the image_corrections list.

    Returns
    -------
    bool, optional
    """
    image_corrections = get_pydicom_meta_tag(
        dcm_seq=dcm_seq,
        tag=(0x0028, 0x0051),
        tag_type="str"
    )
    if image_corrections is not None:
        image_corrections = image_corrections.replace(
            " ", "").replace(
            "[", "").replace(
            "]", "").replace(
            "\'", "").split(sep=",")

    # Read from enhanced PET.
    is_corrected = get_pydicom_meta_tag(dcm_seq=dcm_seq, tag=tag, tag_type="str")
    if is_corrected is None and image_corrections is None:
        is_corrected = "NO"
    elif is_corrected is None and correction_abbr in image_corrections:
        is_corrected = "YES"
    elif is_corrected is None and correction_abbr not in image_corrections:
        is_corrected = "NO"
    else:
        pass

    return is_corrected == "YES"


def convert_dicom_time(
        datetime_str: Optional[str] = None,
        date_str: Optional[str] = None,
        time_str: Optional[str] = None
) -> Optional[datetime.datetime]:
    """
    Converts DICOM date, time or datetime string to a datetime.datetime object to facilitate use in Python.

    Parameters
    ----------
    datetime_str: str, optional
        Datetime string extract from a DICOM tag.

    date_str: str, optional
        Date string extracted from a DICOM tag.

    time_str: str, optional
        Time string extracted from a DICOM tag.

    Returns
    -------
    datetime.datetime
    """

    if datetime_str is None and (date_str is None or time_str is None):
        # No reference time can be established
        ref_time = None

    elif datetime_str is not None:
        # Single datetime string provided
        year = int(datetime_str[0:4])
        month = int(datetime_str[4:6])
        day = int(datetime_str[6:8])
        hour = int(datetime_str[8:10])
        minute = int(datetime_str[10:12])
        second = int(datetime_str[12:14])
        if len(datetime_str) > 14:
            microsecond = int(round(float(datetime_str[14:]) * 1000))
        else:
            microsecond = 0

        ref_time = datetime.datetime(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            microsecond=microsecond
        )

    else:
        # Separate date and time strings provided
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        if len(time_str) > 6:
            microsecond = int(round(float(time_str[6:]) * 1000))
        else:
            microsecond = 0

        ref_time = datetime.datetime(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            microsecond=microsecond
        )

    return ref_time


def get_pydicom_meta_tag(
        dcm_seq: pydicom.Dataset,
        tag: Tuple[hex, hex],
        tag_type: Optional[str] = None,
        default: Optional[Any] = None,
        test_tag: bool = False
) -> Any:
    """
    Extract parameter from DICOM metadata.

    Parameters
    ----------
    dcm_seq: pydicom.Dataset
        DICOM sequence from which the parameter should be read.

    tag: tuple of hex
        Hexadecimal values representing the DICOM parameter tag.

    tag_type: str, optional
        Type to which the parameter should be converted.

    default: any, optional, default: None
        Default value to be used in absence of any value from the DICOM metadata.

    test_tag: bool, optional, default: False
        Determine whether a tag exists.

    Returns
    -------
    Any
        Value of the parameter read for from the DICOM metadata.
    """
    # Reads dicom tag

    # Initialise with default
    tag_value = default

    # Parse raw data
    try:
        tag_value = dcm_seq[tag].value
    except KeyError:
        if test_tag:
            return False
        else:
            pass

    if test_tag:
        return True

    if isinstance(tag_value, bytes):
        tag_value = tag_value.decode("ASCII")

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

        # Multiple floats
        elif tag_type == "mult_float":
            tag_value = [float(str_num) for str_num in tag_value]

        # Integer
        elif tag_type == "int":
            tag_value = int(tag_value)

        # Multiple floats
        elif tag_type == "mult_int":
            tag_value = [int(str_num) for str_num in tag_value]

        # Boolean
        elif tag_type == "bool":
            tag_value = bool(tag_value)

        elif tag_type == "mult_str":
            tag_value = list(tag_value)

        else:
            raise ValueError(f"The tag type was not recognised: {tag_type}")

    return tag_value


def has_pydicom_meta_tag(dcm_seq: Union[FileDataset, Dataset], tag):

    return get_pydicom_meta_tag(dcm_seq=dcm_seq, tag=tag, test_tag=True)


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
