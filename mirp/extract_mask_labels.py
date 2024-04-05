from typing import Any
from pathlib import Path

import os
import pandas as pd

from mirp._data_import.generic_file import MaskFile


def extract_mask_labels(
        mask=None,
        sample_name: None | str | list[str] = None,
        mask_name: None | str | list[str] = None,
        mask_file_type: None | str = None,
        mask_modality: None | str | list[str] = None,
        mask_sub_folder: None | str = None,
        stack_masks: str = "auto",
        write_dir: None | str | Path = None
) -> pd.DataFrame | None:
    """
    Extract labels of regions of interest present in one or more mask files.

    Parameters
    ----------
    mask: Any
        A path to a mask file, a path to a directory containing mask files, a path to a config_data.xml
        file, a path to a csv file containing references to mask files, a pandas.DataFrame containing references to
        mask files, or a numpy.ndarray.

    sample_name: str or list of str, optional, default: None
        Name of expected sample names. This is used to select specific mask files. If None, no mask files are filtered
        based on the corresponding sample name (if known).

    mask_name: str, optional, default: None
        Pattern to match mask files against. The matches are exact. Use wildcard symbols ("*") to match varying
        structures. The sample name (if part of the file name) can also be specified using "#". For example,
        mask_name = '#_*_mask' would find John_Doe in John_Doe_CT_mask.nii or John_Doe_001_mask.nii. File extensions
        do not need to be specified. If None, file names are not used for filtering files and setting sample names.

    mask_file_type: {"dicom", "nifti", "nrrd", "numpy", "itk"}, optional, default: None
        The type of file that is expected. If None, the file type is not used for filtering files.
        "itk" comprises "nifti" and "nrrd" file types.

    mask_modality: {"rtstruct", "seg", "generic_mask"}, optional, default: None
        The type of modality that is expected. If None, modality is not used for filtering files.
        Note that only DICOM files contain metadata concerning modality. Masks from non-DICOM files are considered to
        be "generic_mask".

    mask_sub_folder: str, optional, default: None
        Fixed directory substructure where mask files are located. If None, the directory substructure is not used for
        filtering files.

    stack_masks: {"auto", "yes", "no"}, optional, default: "str"
        If mask files in the same directory cannot be assigned to different samples, and are 2D (slices) of the same
        size, they might belong to the same 3D mask stack. "auto" will stack 2D numpy arrays, but not other file
        types. "yes" will stack all files that contain 2D images, that have the same dimensions, orientation and
        spacing, except for DICOM files. "no" will not stack any files. DICOM files ignore this argument,
        because their stacking can be determined from metadata.

    write_dir: str, optional, default: None
        Directory to which a table with mask labels should be written, if any. Masks labels are exported to
        ``mask_labels.csv``.

    Returns
    -------
    pd.DataFrame | None
        The functions returns a table with labels extracted from mask files (``write_dir = None``) or nothing.

    """
    from mirp.data_import.import_mask import import_mask

    mask_list = import_mask(
        mask=mask,
        sample_name=sample_name,
        mask_name=mask_name,
        mask_file_type=mask_file_type,
        mask_modality=mask_modality,
        mask_sub_folder=mask_sub_folder,
        stack_masks=stack_masks
    )

    labels = [pd.DataFrame(_extract_mask_labels(ii, mask)) for ii, mask in enumerate(mask_list)]
    labels = pd.concat(labels)

    if write_dir is not None:
        write_dir = os.path.normpath(write_dir)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        labels.to_csv(
            path_or_buf=os.path.join(write_dir, "mask_labels.csv")
        )
    else:
        return labels


def _extract_mask_labels(index: int, mask: MaskFile) -> dict[str, Any]:

    labels = mask.export_roi_labels()
    labels.update({"mask_index": index})

    return labels
