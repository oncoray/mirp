"""
Fixed bin size and fixed bin number algorithms are already fully covered in ibsi_1_test.py and elsewhere.
"""

import os
from mirp import extract_features

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_pyradiomics_fbs():
    data = extract_features(
        write_features=False,
        export_features=True,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        ibsi_compliant=False,
        base_discretisation_method="fixed_bin_size_pyradiomics",
        base_discretisation_bin_width=15.0
    )

    feature_data = data[0]

    assert len(feature_data) == 1
