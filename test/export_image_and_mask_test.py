import numpy as np
import os
import pytest

from mirp.extract_features_and_images import extract_features_and_images

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.ci
def test_basic_ct_feature_extraction(tmp_path):
    image = np.zeros((128, 128), dtype=float)
    image[32:64, 32:64] = 1.0

    mask = np.zeros((128, 128), dtype=bool)
    mask[48:64, 48:64] = True

    extract_features_and_images(
        write_features=True,
        export_features=False,
        write_images=True,
        export_images=False,
        write_dir=tmp_path,
        image=image,
        mask=mask,
        base_feature_families="statistics"
    )

    # Check that files exist.
    file_names = [file for file in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, file))]

    assert len(file_names) == 3
    assert len([file for file in file_names if file.endswith(".csv")]) == 1
    assert len([file for file in file_names if file.endswith(".nii.gz")]) == 2
    assert len([file for file in file_names if "mask" in file]) == 1
