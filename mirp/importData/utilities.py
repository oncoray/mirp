from typing import Union


def supported_image_modalities(modality: Union[None, str] = None):
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
        raise ValueError(f"Encountered an unknown image modality as modality argument: {modality}")


def supported_mask_modalities(modality: Union[None, str] = None):
    if modality is None:
        return ["rtsruct", "seg", "generic_mask"]

    elif modality == "rtstruct":
        return ["rtstruct"]

    elif modality in ["seg"]:
        return ["seg"]

    elif modality == "generic_mask":
        return ["generic_mask"]

    else:
        raise ValueError(f"Encountered an unknown segmentation modality as modality argument: {modality}")


def supported_file_types(file_type: Union[None, str] = None):
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
        raise ValueError(f"Encountered an unknown type of image as file_type argument: {file_type}")


def flatten_list(unflattened_list):

    if len(unflattened_list) == 0:
        return unflattened_list

    if isinstance(unflattened_list[0], list):
        return flatten_list(unflattened_list[0]) + flatten_list(unflattened_list[1:])

    return unflattened_list[:1] + flatten_list(unflattened_list[1:])


def create_sample_name():
    ...