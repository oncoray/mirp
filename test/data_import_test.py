import os.path
import shutil

import itk
import numpy as np

from mirp.importData.importImage import import_image
from mirp.importData.importMask import import_mask
from mirp.importData.importImageAndMask import import_image_and_mask
from mirp.importData.imageITKFile import ImageITKFile
from mirp.importData.imageDicomFileStack import ImageDicomFileStack
from mirp.importData.imageNumpyFile import ImageNumpyFile
from mirp.importData.imageNumpyFileStack import ImageNumpyFileStack

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _convert_to_numpy(as_slice=False):
    """
    Helper script for converting NIfTI files to numpy for testing numpy-based imports.

    :param as_slice:
    :return:
    """

    main_folders = ["STS_001", "STS_002", "STS_003"]
    subfolders = ["CT", "MR_T1", "PET"]

    for sample_name in main_folders:
        for modality in subfolders:

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
                        if file_type in ["itk", "dicom"]:
                            target_file_name = [sample_name] + target_file_name
                        target_file_name = "_".join(target_file_name)

                        shutil.copyfile(
                            os.path.join(source_directory, current_file),
                            os.path.join(target_directory, target_file_name)
                        )


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
    # Read Nifti images directly.
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
    assert image_list[0].sample_name == "STS_002"
    assert image_list[1].sample_name == "STS_003"

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

    # Read Nifti images for specific samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        image_sub_folder=os.path.join("CT", "nifti", "image"))
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageITKFile) for image_object in image_list)
    assert image_list[0].sample_name == "STS_002"
    assert image_list[1].sample_name == "STS_003"

    # Read DICOM image stacks for a specific samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        image_sub_folder=os.path.join("CT", "dicom", "image"))
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageDicomFileStack) for image_object in image_list)
    assert image_list[0].sample_name == "STS_002"
    assert image_list[1].sample_name == "STS_003"
    assert all(image_object.modality == "ct" for image_object in image_list)

    # Read numpy images for specific samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        image_sub_folder=os.path.join("CT", "numpy", "image"))
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageNumpyFile) for image_object in image_list)
    assert image_list[0].sample_name == "STS_002"
    assert image_list[1].sample_name == "STS_003"

    # Read numpy image stacks for specific samples.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name=["STS_002", "STS_003"],
        image_sub_folder=os.path.join("CT", "numpy_slice", "image"))
    assert len(image_list) == 2
    assert all(isinstance(image_object, ImageNumpyFileStack) for image_object in image_list)
    assert image_list[0].sample_name == "STS_002"
    assert image_list[1].sample_name == "STS_003"

    # Read Nifti images for all samples.
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
    assert image_list[0].sample_name == "STS_001"
    assert image_list[1].sample_name == "STS_002"
    assert image_list[2].sample_name == "STS_003"
    assert all(image_object.modality == "ct" for image_object in image_list)

    # Read numpy images for all samples.
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


def test_single_image_mask_import():
    import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"))

    import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask"))

    import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder="CT/nifti/image",
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask_sub_folder="CT/nifti/mask")

    import_image_and_mask(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder="CT/dicom/image",
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask_sub_folder="CT/dicom/mask")
