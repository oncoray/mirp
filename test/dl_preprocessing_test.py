import os.path

from mirp.settings.generic import SettingsClass
from mirp.deepLearningPreprocessing import deep_learning_preprocessing

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_extract_image_crop():
    # No cropping.
    data = deep_learning_preprocessing(
        output_slices=False,
        crop_size=None,
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1"
    )

    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert mask.shape == (60, 201, 204)

    # Split into splices (with mask present).
    data = deep_learning_preprocessing(
        output_slices=True,
        crop_size=None,
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1"
    )

    images = data[0][0]
    masks = data[0][1]

    assert len(images) == 60
    assert all(image.shape == (1, 201, 204) for image in images)
    assert len(masks) == 60
    assert all(mask.shape == (1, 201, 204) for mask in masks)

    # Crop to size.
    data = deep_learning_preprocessing(
        output_slices=False,
        crop_size=[20, 50, 50],
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1"
    )

    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (20, 50, 50)
    assert mask.shape == (20, 50, 50)

    # Crop to size in-plane.
    data = deep_learning_preprocessing(
        output_slices=False,
        crop_size=[50, 50],
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1"
    )

    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (26, 50, 50)
    assert mask.shape == (26, 50, 50)

    # Split into splices (with mask present) and crop
    data = deep_learning_preprocessing(
        output_slices=True,
        crop_size=[50, 50],
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1"
    )

    images = data[0][0]
    masks = data[0][1]

    assert len(images) == 26
    assert all(image.shape == (1, 50, 50) for image in images)
    assert len(masks) == 26
    assert all(mask.shape == (1, 50, 50) for mask in masks)

    # Crop to size with extrapolation.
    data = deep_learning_preprocessing(
        output_slices=False,
        crop_size=[20, 300, 300],
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1"
    )

    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (20, 300, 300)
    assert image[0, 0, 0] == -1000.0
    assert mask.shape == (20, 300, 300)
    assert not mask[0, 0, 0]

    # Split into splices (with mask present) and crop
    data = deep_learning_preprocessing(
        output_slices=True,
        crop_size=[300, 300],
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1"
    )

    images = data[0][0]
    masks = data[0][1]

    assert len(images) == 26
    assert all(image.shape == (1, 300, 300) for image in images)
    assert len(masks) == 26
    assert all(mask.shape == (1, 300, 300) for mask in masks)
    assert all(not mask[0, 0, 0] for mask in masks)


def test_normalisation_standardisation():
    import numpy as np

    # Intensity z-standardisation without saturation
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation="standardisation",
        tissue_mask_type="none"
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) > -1000.0
    assert np.max(image) < 500.0
    assert mask.shape == (60, 201, 204)

    # Intensity z-standardisation with saturation
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation="standardisation",
        intensity_normalisation_saturation=[-4.0, 4.0],
        tissue_mask_type="none"
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) >= -4.0
    assert np.max(image) <= 4.0
    assert mask.shape == (60, 201, 204)

    # Intensity z-standardisation with saturation and a rough tissue mask
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation="standardisation",
        intensity_normalisation_saturation=[-4.0, 4.0],
        tissue_mask_type="range",
        tissue_mask_range=[-950.0, np.nan]
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) >= -4.0
    assert np.max(image) <= 4.0
    assert mask.shape == (60, 201, 204)


def test_normalisation_range():
    import numpy as np

    # Intensity range-based normalisation without saturation and without range definition.
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation="range",
        tissue_mask_type="none"
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) == 0.0
    assert np.max(image) == 1.0
    assert mask.shape == (60, 201, 204)

    # Intensity range-based normalisation without saturation.
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation="range",
        intensity_normalisation_range=[-200.0, 200.0],
        tissue_mask_type="none"
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) < 0.0
    assert np.max(image) > 1.0
    assert mask.shape == (60, 201, 204)

    # Intensity range-based normalisation with saturation
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation="range",
        intensity_normalisation_range=[-200.0, 200.0],
        intensity_normalisation_saturation=[0.0, 1.0],
        tissue_mask_type="none"
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) == 0.0
    assert np.max(image) == 1.0
    assert mask.shape == (60, 201, 204)

    # Intensity range-based normalisation with saturation and a rough tissue mask
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation="range",
        intensity_normalisation_range=[-200.0, 200.0],
        intensity_normalisation_saturation=[0.0, 1.0],
        tissue_mask_type="relative_range",
        tissue_mask_range=[0.02, 1.00]
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) == 0.0
    assert np.max(image) == 1.0
    assert mask.shape == (60, 201, 204)


def test_normalisation_relative_range():
    import numpy as np

    # Relative intensity range-based normalisation without saturation
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation="relative_range",
        tissue_mask_type="none"
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) == 0.0
    assert np.max(image) == 1.0
    assert mask.shape == (60, 201, 204)

    # Relative intensity range-based normalisation with saturation
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation="relative_range",
        intensity_normalisation_range=[0.02, 0.98],
        tissue_mask_type="none"
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) < 0.0
    assert np.max(image) > 1.0
    assert mask.shape == (60, 201, 204)

    # Relative intensity range-based normalisation with saturation
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation="relative_range",
        intensity_normalisation_range=[0.02, 0.98],
        intensity_normalisation_saturation=[0.0, 1.0],
        tissue_mask_type="none"
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) == 0.0
    assert np.max(image) == 1.0
    assert mask.shape == (60, 201, 204)

    # Relative intensity range-based normalisation with saturation and a rough tissue mask
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation="relative_range",
        intensity_normalisation_saturation=[0.0, 1.0],
        tissue_mask_type="relative_range",
        tissue_mask_range=[0.02, 1.00]
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) == 0.0
    assert np.max(image) == 1.0
    assert mask.shape == (60, 201, 204)


def test_normalisation_quantile_range():
    import numpy as np

    # Quantile intensity range-based normalisation without saturation
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation="quantile_range",
        tissue_mask_type="none"
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) == 0.0
    assert np.max(image) > 1.0
    assert mask.shape == (60, 201, 204)

    # Quantile intensity range-based normalisation without saturation
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation_range=[0.02, 0.98],
        intensity_normalisation="quantile_range",
        tissue_mask_type="none"
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) == 0.0
    assert np.max(image) > 1.0
    assert mask.shape == (60, 201, 204)

    # Quantile intensity range-based normalisation with saturation
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation="quantile_range",
        intensity_normalisation_range=[0.02, 0.98],
        intensity_normalisation_saturation=[0.0, 1.0],
        tissue_mask_type="none"
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) == 0.0
    assert np.max(image) == 1.0
    assert mask.shape == (60, 201, 204)

    # Quantile intensity range-based normalisation with saturation and a rough tissue mask
    settings = SettingsClass(
        base_feature_families="none",
        response_map_feature_families="none",
        intensity_normalisation="quantile_range",
        intensity_normalisation_range=[0.02, 0.98],
        intensity_normalisation_saturation=[0.0, 1.0],
        tissue_mask_type="range",
        tissue_mask_range=[-950.0, np.nan]
    )

    data = process_data(settings)
    image = data[0][0][0]
    mask = data[0][1][0]

    assert image.shape == (60, 201, 204)
    assert np.min(image) == 0.0
    assert np.max(image) == 1.0
    assert mask.shape == (60, 201, 204)


def process_data(settings, output_slices=False, crop_size=None):
    return deep_learning_preprocessing(
        output_slices=output_slices,
        crop_size=crop_size,
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        settings=settings
    )
