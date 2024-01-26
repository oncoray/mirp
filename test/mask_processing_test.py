import copy

import pytest

import numpy as np
from mirp import extract_images


def test_mask_processing():

    image = np.ones(shape=(3, 10, 10), dtype=float)
    mask_1 = np.zeros(shape=(3, 10, 10), dtype=bool)
    mask_2 = np.zeros(shape=(3, 10, 10), dtype=bool)

    mask_1[1, 0:3, 0:3] = True
    mask_1[1, 7:10, 7:10] = True
    mask_2[0:2, 2:8, 2:8] = True

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

    1