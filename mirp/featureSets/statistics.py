import numpy as np
import pandas as pd


def get_intensity_statistics_features(img_obj, roi_obj):
    """
    Extract intensity statistics features for the given ROI
    :param img_obj: image object
    :param roi_obj: roi object with the requested ROI mask
    :return: pandas DataFrame with feature values
    """

    # Convert image volume to table
    df_img = roi_obj.as_pandas_dataframe(img_obj=img_obj, intensity_mask=True)

    # Extract features
    df_feat = compute_intensity_statistics_features(df_img=df_img)

    return df_feat


def compute_intensity_statistics_features(df_img):
    """
    Definitions of intensity-volume histogram features
    :param df_img: pandas DataFrame representation of the image
    :return: pandas DataFrame with feature values
    """

    # Import functions
    import scipy.stats as st

    # Create feature table
    feat_names = ["stat_mean", "stat_var", "stat_skew", "stat_kurt", "stat_median",
                  "stat_min", "stat_p10", "stat_p90", "stat_max", "stat_iqr", "stat_range",
                  "stat_mad", "stat_rmad", "stat_medad", "stat_cov", "stat_qcod", "stat_energy", "stat_rms"]
    df_feat = pd.DataFrame(np.full(shape=(1, len(feat_names)), fill_value=np.nan))
    df_feat.columns = feat_names

    # Skip calculation if df_img is None (e.g. because the input image and/or roi are missing)
    if df_img is None:
        return df_feat

    # Remove voxels outside ROI
    df_img = df_img[df_img.roi_int_mask == True]

    # Check if the roi contains any voxels
    if len(df_img) == 0:
        return df_feat

    # Constant definitions
    n_v = len(df_img)

    ####################################################################################################################
    # Statistical features
    ####################################################################################################################

    # Mean grey level
    df_feat["stat_mean"] = np.mean(df_img.g)

    # Variance
    df_feat["stat_var"] = np.var(df_img.g, ddof=0)

    # Skewness
    if np.var(df_img.g) == 0.0:
        df_feat["stat_skew"] = 0.0
    else:
        df_feat["stat_skew"] = st.skew(df_img.g, bias=True)

    # Kurtosis
    if np.var(df_img.g) == 0.0:
        df_feat["stat_kurt"] = 0.0
    else:
        df_feat["stat_kurt"] = st.kurtosis(df_img.g, bias=True)

    # Median grey level
    df_feat["stat_median"] = np.median(df_img.g)

    # Minimum grey level
    df_feat["stat_min"] = np.min(df_img.g)

    # 10th percentile
    df_feat["stat_p10"] = np.percentile(df_img.g, q=10)

    # 90th percentile
    df_feat["stat_p90"] = np.percentile(df_img.g, q=90)

    # Maximum grey level
    df_feat["stat_max"] = np.max(df_img.g)

    # Interquartile range
    df_feat["stat_iqr"] = np.percentile(df_img.g, q=75) - np.percentile(df_img.g, q=25)

    # Range
    df_feat["stat_range"] = np.max(df_img.g) - np.min(df_img.g)

    # Mean absolute deviation
    df_feat["stat_mad"] = np.mean(np.abs(df_img.g - np.mean(df_img.g)))

    # Robust mean absolute deviation
    df_sel = df_img[(df_img.g >= np.percentile(df_img.g, q=10)) & (df_img.g <= np.percentile(df_img.g, q=90))]
    df_feat["stat_rmad"] = np.mean(np.abs(df_sel.g - np.mean(df_sel.g)))
    del df_sel

    # Median absolute deviation
    df_feat["stat_medad"] = np.mean(np.abs(df_img.g - np.median(df_img.g)))

    # Coefficient of variance
    if np.var(df_img.g, ddof=0) == 0.0:
        df_feat["stat_cov"] = 0.0
    else:
        df_feat["stat_cov"] = np.sqrt(np.var(df_img.g, ddof=0)) / np.mean(df_img.g)

    # Quartile coefficient of dispersion
    denominator = np.percentile(df_img.g, q=75) + np.percentile(df_img.g, q=25)
    if denominator == 0.0:
        df_feat["stat_qcod"] = 1.0E6
    else:
        df_feat["stat_qcod"] = (np.percentile(df_img.g, q=75) - np.percentile(df_img.g, q=25)) / denominator

    # Energy
    df_feat["stat_energy"] = np.sum(df_img.g ** 2.0)

    # Root mean square
    df_feat["stat_rms"] = np.sqrt(np.sum(df_img.g ** 2.0) / n_v)

    return df_feat
