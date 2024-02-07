import copy

import pytest

import numpy as np
from mirp import extract_images


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
