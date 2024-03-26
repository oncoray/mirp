import os.path
import shutil

import itk
import numpy as np
import pytest

from mirp.data_import.import_image import import_image
from mirp._data_import.itk_file import ImageITKFile
from mirp._data_import.dicom_file_stack import ImageDicomFileStack
from mirp._data_import.numpy_file import ImageNumpyFile
from mirp._data_import.numpy_file_stack import ImageNumpyFileStack

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _convert_to_numpy(as_slice=False):
    """
    Helper script for converting NIfTI files to numpy for testing numpy-based imports.

    :param as_slice:
    :return:
    """

    main_folders = ["STS_001", "STS_002", "STS_003"]
    sub_folders = ["CT", "MR_T1", "PET"]

    for sample_name in main_folders:
        for modality in sub_folders:

            # Read and parse image
            source_image_file = os.path.join(
                CURRENT_DIR, "data", "sts_images", sample_name, modality, "nifti", "image", "image.nii.gz")

            source_image = itk.imread(source_image_file)
            source_image = itk.GetArrayFromImage(source_image).astype(np.float32)

            if as_slice:
                folder_name = "numpy_slice"
                target_dir = os.path.join(
                    CURRENT_DIR, "data", "sts_images", sample_name, modality, folder_name, "image")

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                for ii in range(source_image.shape[0]):
                    target_image_file = os.path.join(
                        target_dir, "_".join([sample_name, "{:02d}".format(ii), "image.npy"]))

                    np.save(target_image_file, arr=source_image[ii, :, :].squeeze())

            else:
                folder_name = "numpy"
                target_dir = os.path.join(
                    CURRENT_DIR, "data", "sts_images", sample_name, modality, folder_name, "image")
                target_image_file = os.path.join(target_dir, "_".join([sample_name, "image.npy"]))

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                np.save(target_image_file, arr=source_image)

            # Read and parse mask
            source_mask_file = os.path.join(
                CURRENT_DIR, "data", "sts_images", sample_name, modality, "nifti", "mask", "mask.nii.gz")

            source_mask = itk.imread(source_mask_file)
            source_mask = itk.GetArrayFromImage(source_mask).astype(int)

            if as_slice:
                folder_name = "numpy_slice"
                target_dir = os.path.join(CURRENT_DIR, "data", "sts_images", sample_name, modality, folder_name, "mask")

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                for ii in range(source_mask.shape[0]):
                    target_mask_file = os.path.join(
                        target_dir, "_".join([sample_name, "{:02d}".format(ii), "mask.npy"]))

                    np.save(target_mask_file, arr=source_mask[ii, :, :].squeeze())

            else:
                folder_name = "numpy"
                target_dir = os.path.join(CURRENT_DIR, "data", "sts_images", sample_name, modality, folder_name, "mask")
                target_mask_file = os.path.join(target_dir, "_".join([sample_name, "mask.npy"]))

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                np.save(target_mask_file, arr=source_mask)


def _convert_to_flat_directory():
    """
    Helper script for converting soft-tissue sarcoma imaging files to a flat directory.
    :return:
    """

    sample_names = ["STS_001", "STS_002", "STS_003"]
    modalities = ["CT", "MR_T1", "PET"]
    file_types = ["dicom", "nifti", "numpy", "numpy_slice"]

    main_target_directory = os.path.join(CURRENT_DIR, "data", "sts_images_flat")

    for sample_name in sample_names:
        for modality in modalities:
            for file_type in file_types:
                for content_type in ["image", "mask"]:
                    source_directory = os.path.join(
                        CURRENT_DIR, "data", "sts_images", sample_name, modality, file_type, content_type)

                    target_directory = os.path.join(main_target_directory, file_type)
                    if not os.path.exists(target_directory):
                        os.makedirs(target_directory)

                    dir_contents = os.listdir(source_directory)
                    for current_file in dir_contents:
                        target_file_name = [modality, current_file]
                        if file_type in ["nifti", "dicom"]:
                            target_file_name = [sample_name] + target_file_name
                        target_file_name = "_".join(target_file_name)

                        shutil.copyfile(
                            os.path.join(source_directory, current_file),
                            os.path.join(target_directory, target_file_name)
                        )


def test_sample_name_parser():
    """
    This tests the isolate_sample_name function that is used to determine sample names from the name of file after a
    specific pattern.
    :return:
    """
    from mirp._data_import.utilities import isolate_sample_name

    # No sample name placeholder symbol.
    sample_name = isolate_sample_name(
        x="Sample_Name_CT_image.npy",
        pattern="*image",
        file_extenstion=".npy"
    )
    assert sample_name is None

    # No matching pattern.
    sample_name = isolate_sample_name(
        x="Sample_Name_CT_image.npy",
        pattern="#_PET*image",
        file_extenstion=".npy"
    )
    assert sample_name is None

    # Simple case.
    sample_name = isolate_sample_name(
        x="Sample_Name_CT_image.npy",
        pattern="#_CT_image",
        file_extenstion=".npy"
    )
    assert sample_name == "Sample_Name"

    # Generic case.
    sample_name = isolate_sample_name(
        x="Sample_Name_CT_image.npy",
        pattern="#_*_*",
        file_extenstion=".npy"
    )
    assert sample_name == "Sample_Name"

    # Case with preceding element.
    sample_name = isolate_sample_name(
        x="Sample_Name_CT_image.npy",
        pattern="Sample_#_*_*",
        file_extenstion=".npy"
    )
    assert sample_name == "Name"

    # Case with wrong preceding element.
    sample_name = isolate_sample_name(
        x="Sample_Name_CT_image.npy",
        pattern="Item_#_*_*",
        file_extenstion=".npy"
    )
    assert sample_name is None

    # Case with preceding and subsequent element.
    sample_name = isolate_sample_name(
        x="Sample_Name_CT_image.npy",
        pattern="Sample_#_CT_*",
        file_extenstion=".npy"
    )
    assert sample_name == "Name"

    # Case with wrong subsequent element.
    sample_name = isolate_sample_name(
        x="Sample_Name_CT_image.npy",
        pattern="Sample_#_PET_*",
        file_extenstion=".npy"
    )
    assert sample_name is None

    # Generic case.
    sample_name = isolate_sample_name(
        x="Sample_Name_CT_image.npy",
        pattern="*_#_*_*",
        file_extenstion=".npy"
    )
    assert sample_name == "Name"

    # Underspecified case.
    sample_name = isolate_sample_name(
        x="Sample_Name_CT_image.npy",
        pattern="*#*",
        file_extenstion=".npy"
    )
    assert sample_name == "Sample_Name_CT_image"

    # Only placeholder
    sample_name = isolate_sample_name(
        x="Sample_Name_CT_image.npy",
        pattern="#",
        file_extenstion=".npy"
    )
    assert sample_name == "Sample_Name_CT_image"

    # Generic case with varying separators.
    sample_name = isolate_sample_name(
        x="Sample-Name_CT.image.npy",
        pattern="*-#_*.*",
        file_extenstion=".npy"
    )
    assert sample_name == "Name"

    # Generic case with varying separators, where the remaining string is empty.
    sample_name = isolate_sample_name(
        x="Sample-Name_CT.image.npy",
        pattern="*-Name#_*.*",
        file_extenstion=".npy"
    )
    assert sample_name is None


def test_single_image_import():

    # Read a Nifti image directly.
    image_list = import_image(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"))
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageITKFile)

    # Read a DICOM image stack.
    image_list = import_image(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"))
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageDicomFileStack)
    assert image_list[0].sample_name == "STS_001"

    # Read a numpy image directly.
    image_list = import_image(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"))
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFile)

    # Read a numpy stack.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"))
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFileStack)

    # Read a Nifti image for a specific sample.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "nifti", "image"))
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageITKFile)
    assert image_list[0].sample_name == "STS_001"

    # Read a DICOM image stack for a specific sample.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "dicom", "image"))
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageDicomFileStack)
    assert image_list[0].sample_name == "STS_001"
    assert image_list[0].modality == "ct"

    # Read a numpy image for a specific sample.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "numpy", "image"))
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFile)
    assert image_list[0].sample_name == "STS_001"

    # Read a numpy image stack for a specific sample.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "numpy_slice", "image"))
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFileStack)
    assert image_list[0].sample_name == "STS_001"

    # Read a Nifti image by specifying the image name.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti"),
        image_name="image")
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageITKFile)

    # Read a numpy file by specifying the image name.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy"),
        image_name="*image")
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFile)

    # Read a numpy stack by specifying the image name.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice"),
        image_name="*image")
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFileStack)

    # Read a DICOM image stack by specifying the modality, the sample name and the file type.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_modality="CT",
        image_file_type="dicom")
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageDicomFileStack)
    assert image_list[0].sample_name == "STS_001"
    assert image_list[0].modality == "ct"


def test_multiple_image_import():
    # Read Nifti _images directly.
    image_list = import_image([
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "nifti", "image", "image.nii.gz"),
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "nifti", "image", "image.nii.gz")
    ])
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageITKFile) for image_object in image_list)

    # Read DICOM image stacks.
    image_list = import_image([
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "dicom", "image"),
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "dicom", "image")
    ])
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageDicomFileStack) for image_object in image_list)
    assert {image_list[0].sample_name, image_list[1].sample_name} == {"STS_002", "STS_003"}

    # Read a numpy image directly.
    image_list = import_image([
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "numpy", "image", "STS_002_image.npy"),
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "numpy", "image", "STS_003_image.npy")
    ])
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageNumpyFile) for image_object in image_list)

    # Read a numpy stack.
    image_list = import_image([
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_002", "CT", "numpy_slice", "image"),
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_003", "CT", "numpy_slice", "image")
    ])
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageNumpyFileStack) for image_object in image_list)

    # Read Nifti _images for specific samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        image_sub_folder=os.path.join("CT", "nifti", "image"))
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageITKFile) for image_object in image_list)
    assert {image_list[0].sample_name, image_list[1].sample_name} == {"STS_002", "STS_003"}

    # Read DICOM image stacks for a specific samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        image_sub_folder=os.path.join("CT", "dicom", "image"))
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageDicomFileStack) for image_object in image_list)
    assert {image_list[0].sample_name, image_list[1].sample_name} == {"STS_002", "STS_003"}
    assert all(image_object.modality == "ct" for image_object in image_list)

    # Read numpy _images for specific samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        image_sub_folder=os.path.join("CT", "numpy", "image"))
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageNumpyFile) for image_object in image_list)
    assert {image_list[0].sample_name, image_list[1].sample_name} == {"STS_002", "STS_003"}

    # Read numpy image stacks for specific samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        image_sub_folder=os.path.join("CT", "numpy_slice", "image"))
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageNumpyFileStack) for image_object in image_list)
    assert {image_list[0].sample_name, image_list[1].sample_name} == {"STS_002", "STS_003"}

    # Read Nifti _images for all samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "nifti", "image"))
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageITKFile) for image_object in image_list)

    # Read DICOM image stacks for all samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "dicom", "image"))
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageDicomFileStack) for image_object in image_list)
    assert {image_list[0].sample_name, image_list[1].sample_name, image_list[2].sample_name} == \
           {"STS_001", "STS_002", "STS_003"}
    assert all(image_object.modality == "ct" for image_object in image_list)

    # Read numpy _images for all samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "numpy", "image"))
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageNumpyFile) for image_object in image_list)

    # Read numpy image stacks for all samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        image_sub_folder=os.path.join("CT", "numpy_slice", "image"))
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageNumpyFileStack) for image_object in image_list)


def test_single_image_import_flat():
    # Read a Nifti image directly.
    image_list = import_image(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
        sample_name="STS_001",
        image_name="#_CT_image")
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageITKFile)
    assert image_list[0].sample_name == "STS_001"

    # Read a DICOM image stack.
    image_list = import_image(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "dicom"),
        sample_name="STS_001",
        image_modality="ct")
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageDicomFileStack)
    assert image_list[0].sample_name == "STS_001"
    assert image_list[0].modality == "ct"

    # Read a numpy image directly.
    image_list = import_image(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy"),
        sample_name="STS_001",
        image_name="CT_#_image")
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFile)
    assert image_list[0].sample_name == "STS_001"

    # Read a numpy stack.
    image_list = import_image(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy_slice"),
        sample_name="STS_001",
        image_name="CT_#_*_image"
    )
    assert len(image_list) == 1
    assert isinstance(image_list[0], ImageNumpyFileStack)
    assert image_list[0].sample_name == "STS_001"

    # Configurations that produce errors.
    with pytest.raises(ValueError):
        _ = import_image(
            os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
            sample_name="STS_001")


def test_multiple_image_import_flat():

    # Read Nifti _images for specific samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
        sample_name=["STS_002", "STS_003"],
        image_name="#_CT_image"
    )
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageITKFile) for image_object in image_list)
    assert set(image_object.sample_name for image_object in image_list) == {"STS_002", "STS_003"}

    # Read DICOM image stacks for a specific samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "dicom"),
        sample_name=["STS_002", "STS_003"],
        image_name="#_CT_*"
    )
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageDicomFileStack) for image_object in image_list)
    assert set(image_object.sample_name for image_object in image_list) == {"STS_002", "STS_003"}
    assert all(image_object.modality == "ct" for image_object in image_list)

    # Read numpy _images for specific samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy"),
        sample_name=["STS_002", "STS_003"],
        image_name="CT_#_image"
    )
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageNumpyFile) for image_object in image_list)
    assert set(image_object.sample_name for image_object in image_list) == {"STS_002", "STS_003"}

    # Read numpy image stacks for specific samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy_slice"),
        sample_name=["STS_002", "STS_003"],
        image_name="CT_#_*_image"
    )
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageNumpyFileStack) for image_object in image_list)
    assert set(image_object.sample_name for image_object in image_list) == {"STS_002", "STS_003"}

    # Read Nifti _images for all samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
        image_name="#_CT_image")
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageITKFile) for image_object in image_list)
    assert set(image_object.sample_name for image_object in image_list) == {"STS_001", "STS_002", "STS_003"}

    # Read DICOM image stacks for all samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "dicom"),
        image_name="#_CT_*"
    )
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageDicomFileStack) for image_object in image_list)
    assert set(image_object.sample_name for image_object in image_list) == {"STS_001", "STS_002", "STS_003"}
    assert all(image_object.modality == "ct" for image_object in image_list)

    # Read numpy _images for all samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy"),
        image_name="CT_#_image"
    )
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageNumpyFile) for image_object in image_list)
    assert set(image_object.sample_name for image_object in image_list) == {"STS_001", "STS_002", "STS_003"}

    # Read numpy image stacks for all samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy_slice"),
        image_name="CT_#_*_image"
    )
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageNumpyFileStack) for image_object in image_list)
    assert set(image_object.sample_name for image_object in image_list) == {"STS_001", "STS_002", "STS_003"}

    # Read Nifti _images for all samples without specifying the sample name in the image name.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti"),
        image_name="*CT_image")
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageITKFile) for image_object in image_list)

    # Read DICOM image stacks for all samples without specifying the image name.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "dicom"),
        image_modality="ct"
    )
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageDicomFileStack) for image_object in image_list)
    assert set(image_object.sample_name for image_object in image_list) == {"STS_001", "STS_002", "STS_003"}
    assert all(image_object.modality == "ct" for image_object in image_list)

    # Read numpy _images for all samples without specifying the sample name in the image name.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy"),
        image_name="CT_*_image"
    )
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageNumpyFile) for image_object in image_list)

    # Read numpy image stacks for all samples without specifying the sample name in the image name.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images_flat", "numpy_slice"),
        image_name="CT_*_image"
    )
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageNumpyFileStack) for image_object in image_list)


def test_image_import_flat_poor_naming():
    """
    Tests whether we can select files if their naming convention is poor, e.g. sample_1, sample_11, sample_111.
    :return:
    """
    # Test correctness when all names are provided.
    image_list = import_image(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti_poor_names"),
        sample_name=["STS_1", "STS_11", "STS_111"],
        image_name="#_*_image")
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageITKFile) for image_object in image_list)
    assert set(image_object.sample_name for image_object in image_list) == {"STS_1", "STS_11", "STS_111"}

    # Test correctness when no names are provided, but the naming structure is clear.
    image_list = import_image(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti_poor_names"),
        image_name="#_PET_image")
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageITKFile) for image_object in image_list)
    assert set(image_object.sample_name for image_object in image_list) == {"STS_1", "STS_11", "STS_111"}

    # Test correctness when no names are provided.
    image_list = import_image(
        os.path.join(CURRENT_DIR, "data", "sts_images_flat", "nifti_poor_names"),
        image_name="*image")
    assert len(image_list) == 3
    assert all(isinstance(image_object, ImageITKFile) for image_object in image_list)
