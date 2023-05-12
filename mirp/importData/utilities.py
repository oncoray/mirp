from typing import Union, List
from os.path import split


def supported_image_modalities(modality: Union[None, str] = None) -> List[str]:
    if modality is None:
        return ["ct", "pt", "pet", "mr", "mri", "generic"]

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


def supported_mask_modalities(modality: Union[None, str] = None) -> List[str]:
    if modality is None:
        return ["rtsruct", "seg", "generic_mask"]

    elif modality == "rtstruct":
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
    if file_type is None:
        return [".dcm", ".nii", ".nii.gz", ".nrrd", ".npy", ".npz"]

    elif file_type == "dicom":
        return [".dcm"]

    elif file_type == "nifti":
        return [".nii", ".nii.gz"]

    elif file_type == "nrrd":
        return [".nrrd"]

    elif file_type == "numpy":
        return [".npy", ".npz"]

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


def path_to_parts(x: str) -> List[str]:
    """
    Split a path into its components.
    :param x: a string or path.
    :return: a list of path components.
    """

    path_parts = []
    while True:
        x_head, x_tail = split(x)
        path_parts += [x_tail]
        if x_head == "":
            break

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
