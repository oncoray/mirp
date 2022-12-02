import os.path
from mirp.importData.importImage import import_data

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_single_file_import():

    import_data(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "mask", "mask.nii.gz"))

    import_data(
        image=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "mask"))

    import_data(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder="CT/nifti/image",
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask_sub_folder="CT/nifti/mask")

    import_data(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder="CT/dicom/image",
        mask=os.path.join(CURRENT_DIR, "data", "sts_images"),
        mask_sub_folder="CT/dicom/mask")