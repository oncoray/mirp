import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from mirp._featuresets.utilities import rep
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask


def get_local_intensity_features(image: GenericImage, mask: BaseMask) -> pd.DataFrame:
    """Calculate local intensity features"""

    # Determine the number of voxels in the mask
    if mask.roi_intensity is not None and not image.is_empty():
        n_voxels = np.sum(mask.roi_intensity.get_voxel_grid())

    elif mask.roi_intensity is None and mask.roi is not None and not image.is_empty():
        # Copy roi mask into the roi intensity mask
        mask.roi_intensity = mask.roi
        n_voxels = np.sum(mask.roi_intensity.get_voxel_grid())

    else:
        n_voxels = 0

    if n_voxels > 300:
        df_local_int = compute_local_mean_intensity_filter(image=image, mask=mask)
    elif n_voxels > 1:
        df_local_int = compute_local_mean_intensity_direct(image=image, mask=mask)
    else:
        df_local_int = None

    # Calculate features
    feature_data = compute_local_intensity_features(df_local_int)

    return feature_data


def compute_local_mean_intensity_filter(image: GenericImage, mask: BaseMask):
    """Use a filter to calculate the local mean intensity"""
    from mirp._images.pet_image import PETImage

    # Determine distance
    distance = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * 10.0

    # Get maximal extension in cubic space
    base_ext = np.floor(distance / np.array(image.image_spacing))

    # Create displacement map
    df_base = pd.DataFrame({
        "x": rep(
            x=np.arange(-base_ext[2], base_ext[2] + 1),
            each=(2 * base_ext[0] + 1) * (2 * base_ext[1] + 1),
            times=1),
        "y": rep(
            x=np.arange(-base_ext[1], base_ext[1] + 1),
            each=2 * base_ext[0] + 1,
            times=2 * base_ext[2] + 1),
        "z": rep(
            x=np.arange(-base_ext[0], base_ext[0] + 1),
            each=1,
            times=(2 * base_ext[1] + 1) * (2 * base_ext[2] + 1))
    })

    # Calculate distances for displacement map
    df_base["dist"] = np.sqrt(np.sum(np.multiply(
        df_base.loc[:, ("z", "y", "x")].values, image.image_spacing
    ) ** 2.0, axis=1))

    # Identify elements in range
    df_base["set_weight"] = df_base.dist <= distance

    # Set weights for filter
    df_base["weight"] = np.zeros(len(df_base))
    df_base.loc[df_base.set_weight == True, "weight"] = 1.0 / np.sum(df_base.set_weight)

    # Update coordinates to start at 0
    df_base.loc[:, ["x", "y", "z"]] -= df_base.loc[0, ["x", "y", "z"]]

    # Generate convolution filter
    conv_filter = np.zeros(shape=(
        np.max(df_base.z).astype(int) + 1,
        np.max(df_base.y).astype(int) + 1,
        np.max(df_base.x).astype(int) + 1)
    )
    conv_filter[df_base.z.astype(int), df_base.y.astype(int), df_base.x.astype(int)] = df_base.weight

    # Filter image using mean filter
    if isinstance(image, PETImage):
        # Use 0.0 constant for PET data
        img_avg = ndi.convolve(
            image.get_voxel_grid(),
            weights=conv_filter,
            mode="constant",
            cval=0.0)
    else:
        img_avg = ndi.convolve(
            image.get_voxel_grid(),
            weights=conv_filter,
            mode="nearest")

    # Construct data frame for comparison
    df_local = pd.DataFrame({
        "g": np.ravel(image.get_voxel_grid()),
        "g_loc": np.ravel(img_avg),
        "in_roi": np.ravel(mask.roi_intensity.get_voxel_grid())
    })

    return df_local


def compute_local_mean_intensity_direct(image: GenericImage, mask: BaseMask):
    """Calculate mean intensity directly from the voxels"""

    # Determine distance
    distance = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * 10.0

    # Construct data frame for comparison
    df_local = pd.DataFrame({
        "g": np.ravel(image.get_voxel_grid()),
        "g_loc": np.ravel(np.full(image.image_dimension, np.nan)),
        "in_roi": np.ravel(mask.roi_intensity.get_voxel_grid())
    })

    # Generate position matrix
    pos_mat = np.array(
        np.unravel_index(indices=np.arange(0, np.prod(image.image_dimension)), shape=image.image_dimension),
        dtype=np.float32).transpose()

    # Iterate over voxels in the roi
    if np.sum(df_local.in_roi) > 1:
        for i in np.array(np.where(df_local.in_roi == True)).squeeze():
            # Determine distance from currently selected voxel
            vox_dist = np.sqrt(np.sum(np.power(np.multiply(pos_mat - pos_mat[i, :], image.image_spacing), 2.0), axis=1))

            # Calculate mean grey level over all voxels within range
            df_local.loc[i, "g_loc"] = np.mean(df_local.g[vox_dist <= distance])

        return df_local

    elif np.sum(df_local.in_roi) == 1:
        i = np.where(df_local.in_roi == True)[0][0]

        # Determine distance from currently selected voxel
        vox_dist = np.sqrt(np.sum(np.power(np.multiply(pos_mat - pos_mat[i, :], image.image_spacing), 2.0), axis=1))

        # Calculate mean grey level over all voxels within range
        df_local.loc[i, "g_loc"] = np.mean(df_local.g[vox_dist <= distance])

        return df_local

    else:
        return None


def compute_local_intensity_features(df_local_int: pd.DataFrame):

    # Create feature table
    feat_names = ["loc_peak_loc", "loc_peak_glob"]
    df_feat = pd.DataFrame(np.full(shape=(1, len(feat_names)), fill_value=np.nan))
    df_feat.columns = feat_names

    # Check if there is a local intensity data frame
    if df_local_int is None:
        return df_feat

    # Shrink df_local to only contain roi voxels
    df_local_int = df_local_int.loc[df_local_int.in_roi == True, :]

    # Check if there are any voxels within the roi
    if len(df_local_int) == 0:
        return df_feat

    # Global grey level peak
    df_feat["loc_peak_glob"] = np.max(df_local_int.g_loc)

    # Local grey level peak
    df_feat["loc_peak_loc"] = np.max(df_local_int.loc[df_local_int.g == np.max(df_local_int.g), "g_loc"])

    return df_feat
