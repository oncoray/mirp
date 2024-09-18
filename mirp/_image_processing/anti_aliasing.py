import numpy as np


def gaussian_preprocess_filter(
        orig_vox,
        orig_spacing,
        sample_spacing=None,
        param_beta=0.98,
        mode="nearest",
        separate_slices=False
):

    from scipy.ndimage import gaussian_filter

    # If no sample spacing is provided, assume original spacing. Note that for most purposes sample
    # spacing should be provided.
    if sample_spacing is None:
        sample_spacing = orig_spacing

    # Set sample spacing and orig_spacing to float
    sample_spacing = sample_spacing.astype(float)
    orig_spacing = orig_spacing.astype(float)

    # Calculate the zoom factors
    map_spacing = sample_spacing / orig_spacing

    # Only apply to down-sampling (map_spacing > 1.0)
    # map_spacing[map_spacing<=1.0] = 0.0

    # Don't filter along slices if calculations are to occur within the slice only
    if separate_slices:
        map_spacing[0] = 0.0

    # Calculate sigma
    sigma = np.sqrt(-8 * np.power(map_spacing, 2.0) * np.log(param_beta))

    # Apply filter
    new_vox = gaussian_filter(
        input=orig_vox.astype(np.float32),
        sigma=sigma,
        order=0,
        mode=mode
    )

    return new_vox
