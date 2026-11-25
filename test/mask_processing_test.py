import copy
import pytest

import numpy as np
from mirp import extract_images


@pytest.mark.ci
def test_mask_processing():

    # Default settings.
    image = np.ones(shape=(3, 10, 10), dtype=float)
    mask_1a = np.zeros(shape=(3, 10, 10), dtype=bool)
    mask_1b = np.zeros(shape=(3, 10, 10), dtype=bool)
    mask_2a = np.zeros(shape=(3, 10, 10), dtype=bool)
    mask_2b = np.zeros(shape=(3, 10, 10), dtype=bool)

    mask_1a[1, 0:3, 0:3] = True
    mask_1b[1, 7:10, 7:10] = True
    mask_1 = np.logical_or(mask_1a, mask_1b)

    mask_2a[0, 2:8, 2:8] = True
    mask_2b[1, 2:8, 2:8] = True
    mask_2 = np.logical_or(mask_2a, mask_2b)

    data = extract_images(
        image=image,
        mask=[copy.deepcopy(mask_1), copy.deepcopy(mask_2)],
        write_images=False,
        export_images=True
    )

    data = data[0]
    processed_masks = data[1]
    assert len(processed_masks) == 2
    assert np.array_equal(processed_masks[0]["mask"], mask_1)
    assert np.array_equal(processed_masks[1]["mask"], mask_2)

    # Masks merged.
    data = extract_images(
        image=image,
        mask=[copy.deepcopy(mask_1), copy.deepcopy(mask_2)],
        mask_merge=True,
        write_images=False,
        export_images=True
    )

    data = data[0]
    processed_masks = data[1]
    assert len(processed_masks) == 1
    assert np.array_equal(processed_masks[0]["mask"], np.logical_or(mask_1, mask_2))

    # Masks split.
    data = extract_images(
        image=image,
        mask=[copy.deepcopy(mask_1), copy.deepcopy(mask_2)],
        mask_split=True,
        write_images=False,
        export_images=True
    )

    data = data[0]
    processed_masks = data[1]
    assert len(processed_masks) == 3
    assert np.array_equal(processed_masks[0]["mask"], mask_1a)
    assert np.array_equal(processed_masks[1]["mask"], mask_1b)
    assert np.array_equal(processed_masks[2]["mask"], mask_2)

    # Masks merged and then split.
    data = extract_images(
        image=image,
        mask=[copy.deepcopy(mask_1), copy.deepcopy(mask_2)],
        mask_merge=True,
        write_images=False,
        export_images=True
    )

    data = data[0]
    processed_masks = data[1]
    assert len(processed_masks) == 1
    assert np.array_equal(processed_masks[0]["mask"], np.logical_or(mask_1, mask_2))

    # Select largest region.
    data = extract_images(
        image=image,
        mask=[copy.deepcopy(mask_1), copy.deepcopy(mask_2)],
        mask_select_largest_region=True,
        write_images=False,
        export_images=True
    )

    data = data[0]
    processed_masks = data[1]
    assert len(processed_masks) == 2
    assert np.array_equal(processed_masks[0]["mask"], mask_1a)
    assert np.array_equal(processed_masks[1]["mask"], mask_2)

    # Select largest slice.
    data = extract_images(
        image=image,
        mask=[copy.deepcopy(mask_1), copy.deepcopy(mask_2)],
        mask_select_largest_slice=True,
        write_images=False,
        export_images=True
    )

    data = data[0]
    processed_masks = data[1]
    assert len(processed_masks) == 2
    assert np.array_equal(processed_masks[0]["mask"], mask_1)
    assert np.array_equal(processed_masks[1]["mask"], mask_2a)


@pytest.mark.ci
def test_boundary_split():
    from skimage.draw import disk

    # Create image with circular mask.
    image = np.ones(shape=(1, 129, 129), dtype=float)
    mask = np.zeros(shape=(1, 129, 129), dtype=bool)
    ii, jj = disk(center=(64, 64), radius=32)
    mask[0, ii, jj] = True

    # Rim is part of the original mask.
    data = extract_images(
        image=image,
        mask=mask,
        write_images=False,
        export_images=True,
        by_slice=True,
        roi_create_rim=-8.0
    )

    masks = data[0][1]

    assert len(masks) == 3
    # Assert that the original mask [0] is correctly split into bulk [1] and rim [2].
    assert np.sum(masks[0]["mask"]) == np.sum(masks[1]["mask"]) + np.sum(masks[2]["mask"])
    # Assert that for the current settings, the bulk [1] is larger than the rim [2]
    assert np.sum(masks[1]["mask"]) > np.sum(masks[2]["mask"])

    # Rim is not part of the original mask.
    data = extract_images(
        image=image,
        mask=mask,
        write_images=False,
        export_images=True,
        by_slice=True,
        roi_create_rim=8.0
    )

    masks = data[0][1]

    assert len(masks) == 2
    # Assert that the rim [1] has no overlap with the original mask [0].
    assert np.sum(np.logical_and(masks[0]["mask"], (masks[1]["mask"]))) == 0
    # Assert that for the current settings, the area of the mask is between pi*(39.5^2 - 32.5^2) and
    # pi*(40.5^2 - 31.5^2)
    assert (39.5 ** 2.0 - 32.5 ** 2.0) < np.sum(masks[1]["mask"]) / np.pi < (40.5 ** 2.0 - 31.5 ** 2.0)
