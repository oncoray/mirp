import os
import pandas as pd
from typing import Union, Optional, List, Generator, Dict

from mirp.importData.imageGenericFile import ImageFile


def extract_image_parameters(
        image,
        sample_name: Union[None, str, List[str]] = None,
        image_name: Union[None, str, List[str]] = None,
        image_file_type: Union[None, str] = None,
        image_modality: Union[None, str, List[str]] = None,
        image_sub_folder: Union[None, str] = None,
        stack_images: str = "auto",
        write_file: bool = False,
        write_dir: Optional[str] = None
):
    """
    Extract parameters related to image acquisition and reconstruction from images. Not all metadata may
    be available.

    Parameters
    ----------
    image: Any
        A path to an image file, a path to a directory containing image files, a path to a config_data.xml
        file, a path to a csv file containing references to image files, a pandas.DataFrame containing references to
        image files, or a numpy.ndarray.

    sample_name: str or list of str, default: None
        Name of expected sample names. This is used to select specific image files. If None, no image files are
        filtered based on the corresponding sample name (if known).

    image_name: str, optional, default: None
        Pattern to match image files against. The matches are exact. Use wildcard symbols ("*") to
        match varying structures. The sample name (if part of the file name) can also be specified using "#". For
        example, image_name = '#_*_image' would find John_Doe in John_Doe_CT_image.nii or John_Doe_001_image.nii.
        File extensions do not need to be specified. If None, file names are not used for filtering files and
        setting sample names.

    image_file_type: {"dicom", "nifti", "nrrd", "numpy", "itk"}, optional, default: None
        The type of file that is expected. If None, the file type is not used for filtering files.
        "itk" comprises "nifti" and "nrrd" file types.

    image_modality: {"ct", "pet", "pt", "mri", "mr", "generic"}, optional, default: None
        The type of modality that is expected. If None, modality is not used for filtering files. Note that only
        DICOM files contain metadata concerning modality.

    image_sub_folder: str, optional, default: None
        Fixed directory substructure where image files are located. If None, the directory substructure is not used
        for filtering files.

    stack_images: {"auto", "yes", "no"}, optional, default: "str"
        If image files in the same directory cannot be assigned to different samples, and are 2D (slices) of the same
        size, they might belong to the same 3D image stack. "auto" will stack 2D numpy arrays, but not other file types.
        "yes" will stack all files that contain 2D images, that have the same dimensions, orientation and spacing,
        except for DICOM files. "no" will not stack any files. DICOM files ignore this argument, because their stacking
        can be determined from metadata.

    write_file: bool, optional, default: False
        Determines whether image acquisition and reconstruction metadata should be written to a table.

    write_dir: str, optional, default: None
        Folder to which the table with image acquisition and reconstruction metadata is written.

    Returns
    -------
    pd.DataFrame or None
        The functions returns a table with metadata.
    """

    from mirp.importData.importImage import import_image

    if not write_file:
        write_dir = None

    if write_file and write_dir is None:
        raise ValueError("write_dir argument should be provided for writing a table with image metadata.")

    image_list = import_image(
        image=image,
        sample_name=sample_name,
        image_name=image_name,
        image_file_type=image_file_type,
        image_modality=image_modality,
        image_sub_folder=image_sub_folder,
        stack_images=stack_images
    )

    metadata = [extract_image_parameters(ii, image) for ii, image in enumerate(image_list)]
    metadata = pd.DataFrame(metadata)

    if write_file:
        write_dir = os.path.normpath(write_dir)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        metadata.to_csv(
            path_or_buf=os.path.join(write_dir, "mask_labels.csv")
        )
    else:
        return metadata


def _extract_image_parameters(index: int, image: ImageFile) -> Generator[Dict[str, str], None, None]:

    metadata = image.export_metadata()
    metadata.update({"image_index": index})

    yield metadata