import numpy as np

from mirp._masks.base_mask import BaseMask


def split_masks(
        masks: None | BaseMask | list[BaseMask],
        boundary_sizes: None | list[float] = None,
        max_erosion: None | float = 0.8,
        by_slice: bool = False
):
    if boundary_sizes is None or len(boundary_sizes) == 0 or \
            all(boundary_size == 0.0 for boundary_size in boundary_sizes):
        return masks

    if masks is None:
        return None

    # Determine the return format.
    if not isinstance(masks, list):
        masks = [masks]

    masks: list[BaseMask] = masks
    new_masks = []

    # Iterate over _masks.
    for mask in masks:
        # Store original.
        new_masks += [mask]

        for boundary_size in boundary_sizes:
            if boundary_size == 0.0:
                continue

            bulk_mask = mask.copy()
            bulk_mask.roi_name += "_bulk_" + str(boundary_size)

            boundary_mask = mask.copy()
            boundary_mask.roi_name += "_boundary_" + str(boundary_size)

            bulk_mask.erode(
                by_slice=by_slice,
                distance=boundary_size,
                max_eroded_volume_fraction=max_erosion
            )

            boundary_mask.roi.set_voxel_grid(voxel_grid=np.logical_xor(
                mask.roi.get_voxel_grid(), bulk_mask.roi.get_voxel_grid()))

            if bulk_mask.roi.is_empty_mask() or boundary_mask.roi.is_empty_mask():
                continue

            new_masks += [bulk_mask, boundary_mask]

    return new_masks
