import os.path

import itk
import numpy as np

from mirp.importData.importImage import import_image
from mirp.importData.importMask import import_mask
from mirp.importData.importImageAndMask import import_image_and_mask

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _convert_to_numpy(as_slice=False):
    """
    Helper script to convert NIfTI files to numpy for testing numpy-based imports.

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


def test_single_image_import():
    # Read a Nifti image directly.
    image_list = import_image(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"))

    # Read a DICOM image stack.
    image_list = import_image(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"))

    # Read a numpy image directly.
    image_list = import_image(
        os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy", "image", "STS_001_image.npy"))

    # Read a numpy stack.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice", "image"))

    # Read a Nifti image for a specific sample.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "nifti", "image"))

    # Read a DICOM image stack for a specific sample.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "dicom", "image"))

    # Read a numpy image for a specific sample.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "numpy", "image"))

    # Read a numpy image stack for a specific sample.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder=os.path.join("CT", "numpy_slice", "image"))

    # Read a Nifti image by specifying the image name.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti"),
        image_name="image")

    # Read a numpy file by specifying the image name.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy"),
        image_name="image")

    # Read a numpy stack by specifying the image name.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "numpy_slice"),
        image_name="image")

    # Read a DICOM image stack by specifying the modality, the sample name and the file type.
    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_modality="CT",
        image_file_type="dicom")

    # Read a
    1


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
