import os.path
from mirp.importData.importImage import import_image
from mirp.importData.importMask import import_mask
from mirp.importData.importImageAndMask import import_image_and_mask

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_single_image_import():
    # # Read a Nifti image directly.
    # image_list = import_image(
    #     os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "nifti", "image", "image.nii.gz"))
    #
    # # Read a DICOM image stack.
    # image_list = import_image(
    #     os.path.join(CURRENT_DIR, "data", "sts_images", "STS_001", "CT", "dicom", "image"))
    #
    # # Read a Nifti image for a specific sample.
    # image_list = import_image(
    #     image=os.path.join(CURRENT_DIR, "data", "sts_images"),
    #     sample_name="STS_001",
    #     image_sub_folder="CT/nifti/image")

    image_list = import_image(
        image=os.path.join(CURRENT_DIR, "data", "sts_images"),
        sample_name="STS_001",
        image_sub_folder="CT/dicom/image")

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