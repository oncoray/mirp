import numpy as np
import os

from mirp.extract_features_and_images import extract_features_and_images

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_basic_ct_feature_extraction():
    import shutil

    write_dir = os.path.join(CURRENT_DIR, "data", "temp_data")
    if os.path.exists(write_dir):
        shutil.rmtree(write_dir)

    image = np.zeros((128, 128), dtype=float)
    image[32:64, 32:64] = 1.0

    mask = np.zeros((128, 128), dtype=bool)
    mask[48:64, 48:64] = True

    data = extract_features_and_images(
        write_features=True,
        export_features=True,
        write_images=True,
        export_images=False,
        write_dir=write_dir,
        image=image,
        mask=mask,
        base_feature_families="statistics"
    )

    # Check that files exist.
    file_names = [file for file in os.listdir(write_dir) if os.path.isfile(os.path.join(write_dir, file))]

    assert len(file_names) == 3
    assert len([file for file in file_names if file.endswith(".csv")]) == 1
    assert len([file for file in file_names if file.endswith(".nii.gz")]) == 2
    assert len([file for file in file_names if "mask" in file]) == 1

    if os.path.exists(write_dir):
        shutil.rmtree(write_dir)
