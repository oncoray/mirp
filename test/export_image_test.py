import os.path
import numpy as np
import pytest

from mirp.data_import.import_image import import_image
from mirp._data_import.read_data import read_image

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.mark.ci
def test_single_image_import(tmp_path):

    # Read a Nifti image directly.
    image_list = import_image(
        os.path.join(CURRENT_DIR, "data", "sts_images_raw", "STS_001", "MR_T1_nifti", "image.nii.gz")
    )

    image_1 = read_image(image=image_list[0])
    image_1.write(dir_path=tmp_path, file_name="test_image", file_format="nifti")

    test_image_path = os.path.join(tmp_path, "test_image.nii.gz")
    assert os.path.exists(test_image_path)

    image_list = import_image(test_image_path)
    image_2 = read_image(image=image_list[0])

    assert np.allclose(image_1.image_orientation, image_2.image_orientation)

    # Clean up.
    os.remove(test_image_path)

