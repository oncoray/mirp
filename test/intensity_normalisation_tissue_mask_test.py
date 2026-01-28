import os.path
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

GENERIC_KWARGS = dict([
    ("write_images", False),
    ("export_images", True),
    ("image_export_format", "native"),
    ("image", os.path.join(CURRENT_DIR, "data", "ct_images_seg", "CRLM-CT-1004", "image")),
    ("mask", os.path.join(CURRENT_DIR, "data", "ct_images_seg", "CRLM-CT-1004", "mask", "mask.dcm")),
    ("roi_name", ["Liver", "Tumor_1"])
])


def test_tissue_mask():
    from mirp import extract_images

    # Without tissue mask.
    data_no_tissue_mask = extract_images(
        tissue_mask_type="none",
        intensity_normalisation="standardisation",
        **GENERIC_KWARGS
    )

    # Using reference.
    data_ref = extract_images(
        tissue_mask_type="reference",
        tissue_mask_name="Liver",
        intensity_normalisation="standardisation",
        **GENERIC_KWARGS
    )

    image_no_mask = data_no_tissue_mask[0][0][0]
    image_ref = data_ref[0][0][0]

    # Liver is relatively homogeneous in CT. Thus we would expect that the maximum value after standardisation in the
    # image with the reference tissue mask is higher than that in the no-tissue-mask image.
    assert np.max(image_ref.get_voxel_grid()) > np.max(image_no_mask.get_voxel_grid())

    # Using range.
    data_range = extract_images(
        tissue_mask_type="range",
        tissue_mask_range=[-100.0, 100.0],
        intensity_normalisation="standardisation",
        **GENERIC_KWARGS
    )

    image_range = data_range[0][0][0]

    # The reference range is small relative to the entire HU range present in CT. Thus, we again expect a higher
    # maximum value.
    assert np.max(image_range.get_voxel_grid()) > np.max(image_no_mask.get_voxel_grid())

    # Using relative range
    data_rel_range = extract_images(
        tissue_mask_type="relative_range",
        tissue_mask_range=[0.25, 0.75],
        intensity_normalisation="standardisation",
        **GENERIC_KWARGS
    )

    image_rel_range = data_rel_range[0][0][0]

    # The reference relative range consists of 50% of the intensities present in the image. Thus, we again expect a
    # higher maximum value compared to standardisation based on the entire volume.
    assert np.max(image_rel_range.get_voxel_grid()) > np.max(image_no_mask.get_voxel_grid())

    # Fixed checks on values.
    assert 2.29 < np.max(image_no_mask.get_voxel_grid()) < 2.30
    assert -1.77 < np.min(image_no_mask.get_voxel_grid()) < -1.76
    assert np.std(image_no_mask.get_voxel_grid()) == 1.0

    assert 72.8 < np.max(image_ref.get_voxel_grid()) < 72.9
    assert -181.5 < np.min(image_ref.get_voxel_grid()) < -181.4
    assert 62.6 < np.std(image_ref.get_voxel_grid()) < 62.7

    assert 21.1 < np.max(image_range.get_voxel_grid()) < 21.2
    assert -47.2 < np.min(image_range.get_voxel_grid()) < -47.1
    assert 16.8 < np.std(image_range.get_voxel_grid()) < 16.9

    assert 4.16 < np.max(image_rel_range.get_voxel_grid()) < 4.17
    assert -5.08 < np.min(image_rel_range.get_voxel_grid()) < -5.07
    assert 2.27 < np.std(image_rel_range.get_voxel_grid()) < 2.28