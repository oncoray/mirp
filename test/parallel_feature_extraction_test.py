import os
from mirp.extractFeaturesAndImages import extract_features_and_images

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_parallel_feature_extraction():
    sequential_data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        perturbation_translation_fraction=[0.0, 0.5],
        base_feature_families="statistics",
        resegmentation_method="range",
        resegmentation_intensity_range=[-1000.0, 250.0]
    )

    paralell_data = extract_features_and_images(
        num_cpus=2,
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        perturbation_translation_fraction=[0.0, 0.5],
        base_feature_families="statistics",
        resegmentation_method="range",
        resegmentation_intensity_range=[-1000.0, 250.0]
    )

    for ii in range(len(sequential_data)):
        assert sequential_data[ii].equals(paralell_data[ii])


