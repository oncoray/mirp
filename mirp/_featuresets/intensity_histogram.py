import copy

import numpy as np
import pandas as pd

from mirp._images.generic_image import GenericImage
from mirp._masks.baseMask import BaseMask


def get_intensity_histogram_features(image: GenericImage, mask: BaseMask) -> pd.DataFrame:
    """
    Extract intensity histogram features for the given mask.
    """

    # Convert image volume to table
    df_img = mask.as_pandas_dataframe(image=image, intensity_mask=True)

    # Extract features
    df_feat = compute_intensity_histogram_features(df_img=df_img, g_range=np.array(mask.intensity_range))

    return df_feat


def compute_intensity_histogram_features(df_img, g_range):
    """Definitions for intensity histogram features"""

    # Create feature table
    feat_names = [
        "ih_mean", "ih_var", "ih_skew", "ih_kurt", "ih_median",
        "ih_min", "ih_p10", "ih_p90", "ih_max", "ih_mode", "ih_iqr", "ih_range",
        "ih_mad", "ih_rmad", "ih_medad", "ih_cov", "ih_qcod", "ih_entropy", "ih_uniformity",
        "ih_max_grad", "ih_max_grad_g", "ih_min_grad", "ih_min_grad_g"
    ]
    df_feat = pd.DataFrame(np.full(shape=(1, len(feat_names)), fill_value=np.nan))
    df_feat.columns = feat_names

    # Skip processing if input image and/or roi are missing
    if df_img is None:
        return df_feat

    # Create working copy
    g_range_loc = copy.deepcopy(g_range)

    # Remove voxels outside ROI
    df_img = df_img[df_img.roi_int_mask == True]

    # Check if anything is left
    if len(df_img) == 0:
        return df_feat

    # Constant definitions
    n_v = len(df_img) * 1.0  # Number of voxels
    if np.isnan(g_range[0]):
        g_range_loc[0] = np.min(df_img.g) * 1.0
    if np.isnan(g_range[1]):
        g_range_loc[1] = np.max(df_img.g) * 1.0

    n_g = g_range_loc[1] - g_range_loc[0] + 1.0  # Number of grey levels

    # Define histogram
    df_his = df_img.groupby(by="g").size().reset_index(name="n")

    # Append empty grey levels to histogram
    levels = np.arange(start=0, stop=n_g) + 1
    miss_level = levels[np.logical_not(np.in1d(levels, df_his.g))]
    n_miss = len(miss_level)
    if n_miss > 0:
        df_his = pd.concat([df_his, pd.DataFrame({"g": miss_level, "n": np.zeros(n_miss)})],
                           ignore_index=True)

    del levels, miss_level, n_miss

    # Update histogram by sorting grey levels and adding bin probabilities
    df_his = df_his.sort_values(by="g")
    df_his["p"] = df_his.n / n_v

    ####################################################################################################################
    # Histogram features
    ####################################################################################################################

    # Intensity histogram mean
    mu = np.sum(df_his.g * df_his.p)
    df_feat["ih_mean"] = mu

    # Intensity histogram variance
    sigma = np.sqrt(np.sum((df_his.g - mu) ** 2.0 * df_his.p))
    df_feat["ih_var"] = sigma ** 2.0

    # Intensity histogram skewness
    if sigma == 0.0:
        df_feat["ih_skew"] = 0.0
    else:
        df_feat["ih_skew"] = np.sum((df_his.g - mu) ** 3.0 * df_his.p) / (sigma ** 3.0)

    # Intensity histogram kurtosis
    if sigma == 0.0:
        df_feat["ih_kurt"] = 0.0
    else:
        df_feat["ih_kurt"] = np.sum((df_his.g - mu) ** 4.0 * df_his.p) / (sigma ** 4.0) - 3.0

    # Intensity histogram median
    df_feat["ih_median"] = np.median(df_img.g)

    # Intensity histogram minimum grey level
    df_feat["ih_min"] = np.min(df_img.g)

    # Intensity histogram 10th percentile
    df_feat["ih_p10"] = np.percentile(df_img.g, q=10)

    # Intensity histogram 90th percentile
    df_feat["ih_p90"] = np.percentile(df_img.g, q=90)

    # Intensity histogram maximum grey level
    df_feat["ih_max"] = np.max(df_img.g)

    # Intensity histogram mode
    mode_g = df_his.loc[df_his.n == np.max(df_his.n)].g.values
    df_feat["ih_mode"] = mode_g[
        np.argmin(np.abs(mode_g - mu))]  # Resolves pathological cases where multiple modes are available

    # Intensity histogram interquartile range
    df_feat["ih_iqr"] = np.percentile(df_img.g, q=75) - np.percentile(df_img.g, q=25)

    # Intensity histogram grey level range
    df_feat["ih_range"] = np.max(df_img.g) - np.min(df_img.g)

    # Mean absolute deviation
    df_feat["ih_mad"] = np.mean(np.abs(df_img.g - mu))

    # Intensity histogram robust mean absolute deviation
    df_sel = df_img[(df_img.g >= np.percentile(df_img.g, q=10)) & (df_img.g <= np.percentile(df_img.g, q=90))]
    df_feat["ih_rmad"] = np.mean(np.abs(df_sel.g - np.mean(df_sel.g)))
    del df_sel

    # Intensity histogram median absolute deviation
    df_feat["ih_medad"] = np.mean(np.abs(df_img.g - np.median(df_img.g)))

    # Intensity histogram coefficient of variance
    if sigma == 0.0:
        df_feat["ih_cov"] = 0.0
    else:
        df_feat["ih_cov"] = sigma / mu

    # Intensity histogram quartile coefficient of dispersion
    df_feat["ih_qcod"] = (np.percentile(df_img.g, q=75) - np.percentile(df_img.g, q=25)) / (
            np.percentile(df_img.g, q=75) + np.percentile(df_img.g, q=25))

    # Intensity histogram entropy
    df_feat["ih_entropy"] = -np.sum(df_his.p[df_his.p > 0.0] * np.log2(df_his.p[df_his.p > 0.0]))

    # Intensity histogram uniformity
    df_feat["ih_uniformity"] = np.sum(df_his.p ** 2.0)

    ####################################################################################################################
    # Histogram gradient features
    ####################################################################################################################

    # Calculate gradient using a second order accurate central differences algorithm
    if len(df_his) > 1:
        df_his["grad"] = np.gradient(df_his.n)
    else:
        df_his["grad"] = 0.0

    # Maximum histogram gradient
    df_feat["ih_max_grad"] = np.max(df_his.grad)

    # Maximum histogram gradient grey level
    df_feat["ih_max_grad_g"] = df_his.g[df_his.grad.idxmax()]

    # Minimum histogram gradient
    df_feat["ih_min_grad"] = np.min(df_his.grad)

    # Minimum histogram gradient grey level
    df_feat["ih_min_grad_g"] = df_his.g[df_his.grad.idxmin()]

    return df_feat


def get_intensity_histogram_features_deprecated(img_obj, roi_obj):
    """
    Extract intensity histogram features for the given ROI
    :param img_obj: image object
    :param roi_obj: roi object with the requested ROI mask
    :return: pandas DataFrame with feature values
    """

    # Convert image volume to table
    df_img = roi_obj.as_pandas_dataframe(img_obj=img_obj, intensity_mask=True)

    # Extract features
    df_feat = compute_intensity_histogram_features(df_img=df_img, g_range=roi_obj.g_range)

    return df_feat
