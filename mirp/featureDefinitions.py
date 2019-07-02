import numpy as np
import pandas as pd
import copy

# from textureMatrixDefinitions import rep
#
#
# def statisticalFeatures(df_img):
#     # Calculate statistical features
#
#     # Import functions
#     import scipy.stats as st
#
#     # Create feature table
#     feat_names = ["stat_mean", "stat_var", "stat_skew", "stat_kurt", "stat_median",
#                   "stat_min", "stat_p10", "stat_p90", "stat_max", "stat_iqr", "stat_range",
#                   "stat_mad", "stat_rmad", "stat_medad", "stat_cov", "stat_qcod", "stat_energy", "stat_rms"]
#     df_feat = pd.DataFrame(np.zeros([1, len(feat_names)]))
#     df_feat.columns = feat_names
#
#     # Create working copy
#     df_gl = copy.deepcopy(df_img)
#
#     # Remove voxels outside ROI
#     df_gl = df_gl[df_gl.in_roi == True]
#
#     # Constant definitions
#     n_v = len(df_gl)
#
#     ####################################################################################################################
#     # Statistical features
#     ####################################################################################################################
#
#     # Mean grey level
#     df_feat.ix[0, "stat_mean"] = np.mean(df_gl.g)
#
#     # Variance
#     df_feat.ix[0, "stat_var"] = np.var(df_gl.g, ddof=0)
#
#     # Skewness
#     if np.var(df_gl.g) == 0.0:
#         df_feat.ix[0, "stat_skew"] = 0.0
#     else:
#         df_feat.ix[0, "stat_skew"] = st.skew(df_gl.g, bias=True)
#
#     # Kurtosis
#     if np.var(df_gl.g) == 0.0:
#         df_feat.ix[0, "stat_kurt"] = 0.0
#     else:
#         df_feat.ix[0, "stat_kurt"] = st.kurtosis(df_gl.g, bias=True)
#
#     # Median grey level
#     df_feat.ix[0, "stat_median"] = np.median(df_gl.g)
#
#     # Minimum grey level
#     df_feat.ix[0, "stat_min"] = np.min(df_gl.g)
#
#     # 10th percentile
#     df_feat.ix[0, "stat_p10"] = np.percentile(df_gl.g, q=10)
#
#     # 90th percentile
#     df_feat.ix[0, "stat_p90"] = np.percentile(df_gl.g, q=90)
#
#     # Maximum grey level
#     df_feat.ix[0, "stat_max"] = np.max(df_gl.g)
#
#     # Interquartile range
#     df_feat.ix[0, "stat_iqr"] = np.percentile(df_gl.g, q=75) - np.percentile(df_gl.g, q=25)
#
#     # Range
#     df_feat.ix[0, "stat_range"] = np.max(df_gl.g) - np.min(df_gl.g)
#
#     # Mean absolute deviation
#     df_feat.ix[0, "stat_mad"] = np.mean(np.abs(df_gl.g - np.mean(df_gl.g)))
#
#     # Robust mean absolute deviation
#     df_sel = df_gl[(df_gl.g >= np.percentile(df_gl.g, q=10)) & (df_gl.g <= np.percentile(df_gl.g, q=90))]
#     df_feat.ix[0, "stat_rmad"] = np.mean(np.abs(df_sel.g - np.mean(df_sel.g)))
#     del df_sel
#
#     # Median absolute deviation
#     df_feat.ix[0, "stat_medad"] = np.mean(np.abs(df_gl.g - np.median(df_gl.g)))
#
#     # Coefficient of variance
#     if np.var(df_gl.g, ddof=0) == 0.0:
#         df_feat.ix[0, "stat_cov"] = 0.0
#     else:
#         df_feat.ix[0, "stat_cov"] = np.sqrt(np.var(df_gl.g, ddof=0)) / np.mean(df_gl.g)
#
#     # Quartile coefficient of dispersion
#     df_feat.ix[0, "stat_qcod"] = (np.percentile(df_gl.g, q=75) - np.percentile(df_gl.g, q=25)) / (np.percentile(df_gl.g, q=75) + np.percentile(df_gl.g, q=25))
#
#     # Energy
#     df_feat.ix[0, "stat_energy"] = np.sum(df_gl.g ** 2.0)
#
#     # Root mean square
#     df_feat.ix[0, "stat_rms"] = np.sqrt(np.sum(df_gl.g ** 2.0) / n_v)
#
#     return df_feat
#
#
# def intHistogramFeatures(df_img, g_range):
#     # Calculate intensity histogram features - Note: expects discretised grey level bins
#
#     # Create feature table
#     feat_names = ["ih_mean", "ih_var", "ih_skew", "ih_kurt", "ih_median",
#                   "ih_min", "ih_p10", "ih_p90", "ih_max", "ih_mode", "ih_iqr", "ih_range",
#                   "ih_mad", "ih_rmad", "ih_medad", "ih_cov", "ih_qcod", "ih_entropy", "ih_uniformity",
#                   "ih_max_grad", "ih_max_grad_g", "ih_min_grad", "ih_min_grad_g"]
#     df_feat = pd.DataFrame(np.zeros([1, len(feat_names)]))
#     df_feat.columns = feat_names
#
#     # Create working copy
#     df_gl = copy.deepcopy(df_img)
#     g_range_loc = copy.deepcopy(g_range)
#
#     # Remove voxels outside ROI
#     df_gl = df_gl[df_gl.in_roi == True]
#
#     # Constant definitions
#     n_v = len(df_gl) * 1.0  # Number of voxels
#     if np.isnan(g_range[0]): g_range_loc[0] = np.min(df_gl.g) * 1.0
#     if np.isnan(g_range[1]): g_range_loc[1] = np.max(df_gl.g) * 1.0
#     n_g = g_range_loc[1] - g_range_loc[0] + 1.0  # Number of grey levels
#
#     # Define histogram
#     df_his     = df_gl.groupby(by="g").size().reset_index(name="n")
#
#     # Append empty grey levels to histogram
#     levels     = np.arange(start=0, stop=n_g) + 1
#     miss_level = levels[np.logical_not(np.in1d(levels, df_his.g))]
#     n_miss     = len(miss_level)
#     if n_miss > 0:
#         df_his = df_his.append(
#             pd.DataFrame({"g": miss_level, "n": np.zeros(n_miss)}),
#             ignore_index=True)
#     del levels, miss_level, n_miss
#
#     # Update histogram by sorting grey levels and adding bin probabilities
#     df_his      = df_his.sort_values(by="g")
#     df_his["p"] = df_his.n / n_v
#
#     ####################################################################################################################
#     # Histogram features
#     ####################################################################################################################
#
#     # Intensity histogram mean
#     mu = np.sum(df_his.g * df_his.p)
#     df_feat.ix[0, "ih_mean"] = mu
#
#     # Intensity histogram variance
#     sigma = np.sqrt( np.sum( (df_his.g - mu) ** 2.0 * df_his.p) )
#     df_feat.ix[0, "ih_var"]  = sigma ** 2.0
#
#     # Intensity histogram skewness
#     if sigma == 0.0: df_feat.ix[0, "ih_skew"] = 0.0
#     else:            df_feat.ix[0, "ih_skew"] = np.sum( (df_his.g - mu) ** 3.0 * df_his.p) / (sigma ** 3.0)
#
#     # Intensity histogram kurtosis
#     if sigma == 0.0: df_feat.ix[0, "ih_kurt"] = 0.0
#     else:            df_feat.ix[0, "ih_kurt"] = np.sum( (df_his.g - mu) ** 4.0 * df_his.p) / (sigma ** 4.0) - 3.0
#
#     # Intensity histogram median
#     df_feat.ix[0, "ih_median"] = np.median(df_gl.g)
#
#     # Intensity histogram minimum grey level
#     df_feat.ix[0, "ih_min"] = np.min(df_gl.g)
#
#     # Intensity histogram 10th percentile
#     df_feat.ix[0, "ih_p10"] = np.percentile(df_gl.g, q=10)
#
#     # Intensity histogram 90th percentile
#     df_feat.ix[0, "ih_p90"] = np.percentile(df_gl.g, q=90)
#
#     # Intensity histogram maximum grey level
#     df_feat.ix[0, "ih_max"] = np.max(df_gl.g)
#
#     # Intensity histogram mode
#     mode_g = df_his.loc[df_his.n==np.max(df_his.n)].g.values
#     df_feat.ix[0, "ih_mode"] = mode_g[np.argmin(np.abs(mode_g - mu))] # Resolves pathological cases where multiple modes are available
#
#     # Intensity histogram interquartile range
#     df_feat.ix[0, "ih_iqr"] = np.percentile(df_gl.g, q=75) - np.percentile(df_gl.g, q=25)
#
#     # Intensity histogram grey level range
#     df_feat.ix[0, "ih_range"] = np.max(df_gl.g) - np.min(df_gl.g)
#
#     # Mean absolute deviation
#     df_feat.ix[0, "ih_mad"] = np.mean(np.abs(df_gl.g - mu))
#
#     # Intensity histogram robust mean absolute deviation
#     df_sel = df_gl[(df_gl.g >= np.percentile(df_gl.g, q=10)) & (df_gl.g <= np.percentile(df_gl.g, q=90))]
#     df_feat.ix[0, "ih_rmad"] = np.mean(np.abs(df_sel.g - np.mean(df_sel.g)))
#     del df_sel
#
#     # Intensity histogram median absolute deviation
#     df_feat.ix[0, "ih_medad"] = np.mean(np.abs(df_gl.g - np.median(df_gl.g)))
#
#     # Intensity histogram coefficient of variance
#     if sigma == 0.0:
#         df_feat.ix[0, "ih_cov"] = 0.0
#     else:
#         df_feat.ix[0, "ih_cov"] = sigma / mu
#
#     # Intensity histogram quartile coefficient of dispersion
#     df_feat.ix[0, "ih_qcod"] = (np.percentile(df_gl.g, q=75) - np.percentile(df_gl.g, q=25)) / (np.percentile(df_gl.g, q=75) + np.percentile(df_gl.g, q=25))
#
#     # Intensity histogram entropy
#     df_feat.ix[0, "ih_entropy"] = -np.sum(df_his.p[df_his.p>0.0] * np.log2(df_his.p[df_his.p>0.0]))
#
#     # Intensity histogram uniformity
#     df_feat.ix[0, "ih_uniformity"] = np.sum(df_his.p ** 2.0)
#
#     ####################################################################################################################
#     # Histogram gradient features
#     ####################################################################################################################
#
#     # Calculate gradient using a second order accurate central differences algorithm
#     df_his["grad"] = np.gradient(df_his.n)
#
#     # Maximum histogram gradient
#     df_feat.ix[0, "ih_max_grad"] = np.max(df_his.grad)
#
#     # Maximum histogram gradient grey level
#     df_feat.ix[0, "ih_max_grad_g"] = df_his.g[np.argmax(df_his.grad)]
#
#     # Minimum histogram gradient
#     df_feat.ix[0, "ih_min_grad"] = np.min(df_his.grad)
#
#     # Minimum histogram gradient grey level
#     df_feat.ix[0, "ih_min_grad_g"] = df_his.g[np.argmin(df_his.grad)]
#
#     return df_feat
#
#
# def intVolHistogramFeatures(df_img, modality, spat_transform, g_range=None, force_binning=False):
#     # Calculate intensity volume histogram features
#
#     # Create feature table
#     feat_names = ["ivh_v10", "ivh_v90", "ivh_i10", "ivh_i90", "ivh_diff_v10_v90", "ivh_diff_i10_i90", "ivh_auc"]
#     df_feat = pd.DataFrame(np.zeros([1, len(feat_names)]))
#     df_feat.columns = feat_names
#
#     # Create working copy
#     df_his = copy.deepcopy(df_img)
#
#     # Remove voxels outside ROI
#     df_his = df_his[df_his.in_roi == True]
#
#     # Constant definitions
#     n_v = len(df_his) * 1.0  # Number of voxels
#
#     if g_range is None:
#         g_range_loc = np.array([np.nan, np.nan])  # Grey level range within ROI
#     else:
#         g_range_loc = g_range
#
#     if force_binning is None:
#         force_binning = False
#
#     # Check if discretisation is required
#     if modality == "CT" and spat_transform == "base" and force_binning == False:
#
#         # Update grey level range when the range is not provided
#         if np.isnan(g_range_loc[0]):
#             g_range_loc[0] = np.min(df_his.g) * 1.0
#         if np.isnan(g_range_loc[1]):
#             g_range_loc[1] = np.max(df_his.g) * 1.0
#         n_bins = g_range_loc[1] - g_range_loc[0] + 1.0  # Number of grey levels
#
#         # Create histogram by grouping by intensity level and counting bin size
#         df_his = df_his.groupby(by="g").size().reset_index(name="n")
#
#         # Append empty grey levels to histogram
#         levels = np.arange(start=g_range_loc[0], stop=g_range_loc[1] + 1)
#         miss_level = levels[np.logical_not(np.in1d(levels, df_his.g))]
#         n_miss = len(miss_level)
#         if n_miss > 0:
#             df_his = df_his.append(
#                 pd.DataFrame({"g": miss_level, "n": np.zeros(n_miss)}),
#                 ignore_index=True)
#         del levels, miss_level, n_miss
#
#     else:
#         if modality == "PT" and spat_transform == "base" and force_binning == False:
#             g_range_loc[0] = 0.0
#             if np.isnan(g_range_loc[1]):
#                 g_range_loc[1] = np.max(df_his.g) * 1.0
#             bin_size       = 0.10
#             n_bins = np.ceil((g_range_loc[1] - g_range_loc[0]) / bin_size) + 1
#
#         else:
#             n_bins = 1000.0
#             g_range_loc[0] = np.min(df_his.g) * 1.0
#             g_range_loc[1] = np.max(df_his.g) * 1.0
#             bin_size       = (g_range_loc[1] - g_range_loc[0]) / n_bins
#
#         bins = g_range_loc[0] + np.arange(0, n_bins + 1) * bin_size
#         bin_c = bins[0:(len(bins)-1)] + np.diff(bins)/2.0
#
#         hist_g = np.histogram(df_his.g, bins=bins)[0]
#         df_his = pd.DataFrame({"g": bin_c, "n": hist_g})
#
#     # Order by increasing grey level
#     df_his      = df_his.sort_values(by="g")
#     df_his["gamma"] = (df_his.g - g_range_loc[0]) / (g_range_loc[1] - g_range_loc[0])
#     df_his["nu"]    = 1.0 - (np.cumsum(np.append([0], df_his.n))[0:np.int(n_bins)]) / n_v
#
#     ####################################################################################################################
#     # Intensity volume histogram features
#     ####################################################################################################################
#
#     # Volume fraction at 10% intensity
#     v10 = df_his.loc[df_his.gamma >= 0.10, :].nu.max()
#     df_feat.ix[0, "ivh_v10"] = v10
#
#     # Volume fraction at 90% intensity
#     v90 = df_his.loc[df_his.gamma >= 0.90, :].nu.max()
#     df_feat.ix[0, "ivh_v90"] = v90
#
#     # Intensity at 10% volume
#     i10 = df_his.loc[df_his.nu <= 0.10, :].g.min()
#     if np.isnan(i10): i10 = n_bins + 1.0
#     df_feat.ix[0, "ivh_i10"] = i10
#
#     # Intensity at 90% volume
#     i90 = df_his.loc[df_his.nu <= 0.90, :].g.min()
#     if np.isnan(i90): i90 = n_bins + 1.0
#     df_feat.ix[0, "ivh_i90"] = i90
#
#     # Difference in volume fraction between 10% and 90% intensity
#     df_feat.ix[0, "ivh_diff_v10_v90"] = v10 - v90
#
#     # Difference in intensity between 10% and 90% volume
#     df_feat.ix[0, "ivh_diff_i10_i90"] = i10 - i90
#
#     # Area under IVH curve
#     df_feat.ix[0, "ivh_auc"] = np.trapz(y=df_his.nu, x=df_his.gamma)
#
#     return df_feat
#
#
def morph_cont_features(roi_obj, spacing):
    # Contour features are calculated on the 2D slice containing the most roi voxels

    # Import functions
    import scipy.stats as st
    from morphologyUtilities import getPerimeter

    # Create feature table
    feat_names = ["morph_c_mean_rd", "morph_c_var_rd", "morph_c_skew_rd", "morph_c_kurt_rd", "morph_c_circularity",
                  "morph_c_roughness", "morph_c_compactness"]
    df_feat = pd.DataFrame(np.zeros([1, len(feat_names)]))
    df_feat.columns = feat_names

    # Find slice containing the largest number of voxels in the roi
    slice_sel = np.argmax(np.sum(np.sum(roi_obj, axis=0), axis=0))

    # Select slice
    roi_slice   = np.squeeze(roi_obj[slice_sel, :, :])

    # Get perimeter
    df_perim = getPerimeter(roi_slice=roi_slice, spacing=spacing[[1, 2]])

    # Radial distance mean
    mu =  np.mean(df_perim.r)
    df_feat.ix[0, "morph_c_mean_rd"] = mu

    # Radial distance variance
    df_feat.ix[0, "morph_c_var_rd"] = np.var(df_perim.r, ddof=0)

    # Radial distance skewness
    if np.var(df_perim.r) == 0.0:
        df_feat.ix[0, "morph_c_skew_rd"] = 0.0
    else:
        df_feat.ix[0, "morph_c_skew_rd"] = st.skew(df_perim.r, bias=True)

    # Radial distance circularity
    if np.var(df_perim.r) == 0.0:
        df_feat.ix[0, "morph_c_kurt_rd"] = 0.0
    else:
        df_feat.ix[0, "morph_c_kurt_rd"] = st.kurtosis(df_perim.r, bias=True)

    # Radial distance circularity (see Pohlman 1996)
    df_feat.ix[0, "morph_c_circularity"] = mu / np.sqrt( df_feat.ix[0, "morph_c_var_rd"])

    # Radial distance roughness
    df_feat.ix[0, "morph_c_roughness"] = ((np.mean( np.power(df_perim.r - mu, 4.0) )) ** 0.25 - (np.mean( np.power(df_perim.r - mu, 2.0) )) ** 0.5) / mu

    # Perimeter compactness
    x_pos = df_perim.x_pos.values
    x_pos = np.append(x_pos, x_pos[0])
    y_pos = df_perim.y_pos.values
    y_pos = np.append(y_pos, y_pos[0])

    # Divide squared perimeter length by roi area
    df_feat.ix[0, "morph_c_compactness"] = np.sum( np.sqrt( np.power(np.diff(x_pos), 2.0) + np.power(np.diff(y_pos), 2.0))) ** 2.0 / (np.sum(roi_slice) * np.prod(spacing))
    del x_pos, y_pos

    # Additional features (see Pohlman 1996)
#
#
# def morphologicalFeatures(map_obj, spacing, dims):
#     # Calculate regional morphological features
#
#     # Import functions
#     from scipy.spatial import ConvexHull
#     from scipy.spatial.distance import pdist
#     from morphologyDefinitions import ellipsoidArea, minOrientedBoundBox, minVolEnclosingEllipsoid, geospatial, mesh_voxels, mesh_volume
#     from skimage.measure import mesh_surface_area
#
#     # Create feature table
#     feat_names = ["morph_volume","morph_vol_approx", "morph_area_mesh", "morph_av", "morph_comp_1", "morph_comp_2",
#                   "morph_sph_dispr", "morph_sphericity", "morph_asphericity", "morph_com",
#                   "morph_diam", "morph_pca_maj_axis", "morph_pca_min_axis", "morph_pca_least_axis",
#                   "morph_pca_elongation", "morph_pca_flatness", "morph_vol_dens_aabb", "morph_area_dens_aabb",
#                   "morph_vol_dens_ombb", "morph_area_dens_ombb", "morph_vol_dens_aee", "morph_area_dens_aee",
#                   "morph_vol_dens_mvee", "morph_area_dens_mvee", "morph_vol_dens_conv_hull",
#                   "morph_area_dens_conv_hull", "morph_integ_int", "morph_moran_i", "morph_geary_c"]
#     df_feat = pd.DataFrame(np.zeros([1, len(feat_names)]))
#     df_feat.columns = feat_names
#
#     # Create working copy
#     df_morph = copy.deepcopy(map_obj)
#
#     # Remove voxels outside of ROI from table
#     df_morph = df_morph[df_morph.in_roi].reset_index()
#
#     # Constant definitions
#     # Note: voxel spacing is expected to follow the z,y,x order
#     n_v = len(df_morph)
#     #volume = n_v * spacing[2] * spacing[1] * spacing[0]
#
#     # Generate mesh
#     verts, faces, norms = mesh_voxels(map_obj=map_obj, spacing=spacing, dims=dims)
#
#     ####################################################################################################################
#     # Geometric features
#     ####################################################################################################################
#
#     # Surface area
#     #area = mesh_area(map_obj=map_obj, spacing=spacing, dims=dims)
#     area = mesh_surface_area(verts=verts, faces=faces)
#     df_feat.ix[0, "morph_area_mesh"] = area
#
#     # Volume
#     volume = mesh_volume(verts=verts, faces=faces)
#     df_feat.ix[0, "morph_volume"] = volume
#
#     # Approximate volume
#     df_feat.ix[0, "morph_vol_approx"] = n_v * np.prod(spacing)
#
#     # Surface to volume ratio
#     df_feat.ix[0, "morph_av"] = area / volume
#
#     # Compactness 1
#     sphereFeat = 36 * np.pi * volume ** 2.0 / area ** 3.0
#     df_feat.ix[0, "morph_comp_1"] = 1.0 / (6.0 * np.pi) * sphereFeat ** (1.0 / 2.0)
#
#     # Compactness 2
#     df_feat.ix[0, "morph_comp_2"] = sphereFeat
#
#     # Spherical disproportion
#     df_feat.ix[0, "morph_sph_dispr"] = sphereFeat ** (-1.0 / 3.0)
#
#     # Sphericity
#     df_feat.ix[0, "morph_sphericity"] = sphereFeat ** (1.0 / 3.0)
#
#     # Asphericity
#     df_feat.ix[0, "morph_asphericity"] = sphereFeat ** (-1.0 / 3.0) - 1.0
#     del sphereFeat
#
#     # Centre of mass shift
#     com_spatial = np.array([np.mean(df_morph.z), np.mean(df_morph.y), np.mean(df_morph.x)])
#     com_gl = np.array([np.sum(df_morph.g * df_morph.z), np.sum(df_morph.g * df_morph.y),
#                        np.sum(df_morph.g * df_morph.x)]) / np.sum(df_morph.g)
#     df_feat.ix[0, "morph_com"] = np.sqrt(np.sum(np.multiply((com_spatial - com_gl), spacing) ** 2.0))
#     del com_spatial, com_gl
#
#     # Integrated intensity
#     df_feat.ix[0, "morph_integ_int"] = volume * np.mean(df_morph.g)
#
#     ####################################################################################################################
#     # Convex hull - based features
#     ####################################################################################################################
#
#     # # Select only border voxels
#     # pos_mat = df_morph[df_morph.border == True].as_matrix(["z", "y", "x"])
#     # pos_mat = np.multiply(pos_mat, spacing)
#
#     # Determine smallest convex hull; edge points represent extremities; max 3D diameter is between extremities
#     # hull = ConvexHull(points=pos_mat)
#     hull = ConvexHull(verts)
#
#     # Maximum 3D diameter
#     # hull_mat = pos_mat[hull.vertices, :] - np.mean(pos_mat, axis=0)
#     # df_feat.ix[0, "morph_diam"] = np.max(pdist(hull_mat))
#     # del pos_mat
#
#     hull_verts = verts[hull.vertices,:] - np.mean(verts, axis=0)
#     df_feat.ix[0, "morph_diam"] = np.max(pdist(hull_verts))
#
#     # Volume density - convex hull
#     df_feat.ix[0, "morph_vol_dens_conv_hull"] = volume / hull.volume
#
#     # Area density - convex hull
#     df_feat.ix[0, "morph_area_dens_conv_hull"] = area / hull.area
#     del hull
#
#     ####################################################################################################################
#     # Bounding box - based features
#     ####################################################################################################################
#
#     # Volume density - axis-aligned bounding box
#     #aabb_dims = np.max(hull_mat, axis=0) - np.min(hull_mat, axis=0)
#     aabb_dims = np.max(hull_verts, axis=0) - np.min(hull_verts, axis=0)
#     df_feat.ix[0, "morph_vol_dens_aabb"] = volume / np.product(aabb_dims)
#
#     # Area density - axis-aligned bounding box
#     df_feat.ix[0, "morph_area_dens_aabb"] = area / (2.0 * aabb_dims[0] * aabb_dims[1] + 2.0 * aabb_dims[0] * aabb_dims[2] + 2.0 * aabb_dims[1] * aabb_dims[2])
#     del aabb_dims
#
#     # Volume density - oriented minimum bounding box
#     #ombb_dims = minOrientedBoundBox(pos_mat=hull_mat)
#     ombb_dims = minOrientedBoundBox(pos_mat=hull_verts)
#     df_feat.ix[0, "morph_vol_dens_ombb"] = volume / np.product(ombb_dims)
#
#     # Area density - oriented minimum bounding box
#     df_feat.ix[0, "morph_area_dens_ombb"] = area / (2.0 * ombb_dims[0] * ombb_dims[1] + 2.0 * ombb_dims[0] * ombb_dims[2] + 2.0 * ombb_dims[1] * ombb_dims[2])
#     del ombb_dims
#
#     ####################################################################################################################
#     # Minimum volume enclosing ellipsoid - based features
#     ####################################################################################################################
#
#     # Calculate semi_axes of minimum volume enclosing ellipsoid
#     # semi_axes = minVolEnclosingEllipsoid(pos_mat=hull_mat, tolerance=10E-4)
#     semi_axes = minVolEnclosingEllipsoid(pos_mat=hull_verts, tolerance=10E-4)
#
#     # Volume density - minimum volume enclosing ellipsoid
#     df_feat.ix[0, "morph_vol_dens_mvee"] = 3 * volume / (4 * np.pi * np.prod(semi_axes))
#
#     # Area density - minimum volume enclosing ellipsoid
#     df_feat.ix[0, "morph_area_dens_mvee"] = area / ellipsoidArea(semi_axes, n_degree=20)
#     del semi_axes, hull_verts
#
#     ####################################################################################################################
#     # Principal component analysis - based features
#     ####################################################################################################################
#
#     # Get position matrix
#     pos_mat_pca = df_morph.as_matrix(["z", "y", "x"])
#
#     # Subtract mean
#     pos_mat_pca = np.multiply((pos_mat_pca - np.mean(pos_mat_pca, axis=0)), spacing)
#
#     # Get eigenvalues and vectors
#     eigen_val, eigen_vec = np.linalg.eigh(np.cov(pos_mat_pca, rowvar=False))
#     semi_axes = 2.0 * np.sqrt(np.sort(eigen_val))
#
#     # Major axis length
#     df_feat.ix[0, "morph_pca_maj_axis"] = semi_axes[2] * 2.0
#
#     # Minor axis length
#     df_feat.ix[0, "morph_pca_min_axis"] = semi_axes[1] * 2.0
#
#     # Least axis length
#     df_feat.ix[0, "morph_pca_least_axis"] = semi_axes[0] * 2.0
#
#     # Elongation
#     df_feat.ix[0, "morph_pca_elongation"] = semi_axes[1] / semi_axes[2]
#
#     # Flatness
#     df_feat.ix[0, "morph_pca_flatness"] = semi_axes[0] / semi_axes[2]
#
#     # Volume density - approximate enclosing ellipsoid
#     df_feat.ix[0, "morph_vol_dens_aee"] = 3 * volume / (4 * np.pi * np.prod(semi_axes))
#
#     # Area density - approximate enclosing ellipsoid
#     df_feat.ix[0, "morph_area_dens_aee"] = area / ellipsoidArea(semi_axes, n_degree=20)
#     del semi_axes, pos_mat_pca
#
#     ####################################################################################################################
#     # Geospatial analysis - based features
#     ####################################################################################################################
#
#     if len(df_morph) < 1000:
#         # Calculate geospatial features using a brute force approach
#         moran_i, geary_c = geospatial(df_morph=df_morph, spacing=spacing)
#
#         df_feat.ix[0, "morph_moran_i"] = moran_i
#         df_feat.ix[0, "morph_geary_c"] = geary_c
#
#     else:
#         # Use monte carlo approach to estimate geospatial features
#
#         # Create lists for storing feature values
#         moran_list, geary_list = [], []
#
#         # Initiate iterations
#         iter    = 1
#         tol_aim = 0.002
#         tol_sem = 1.000
#
#         # Iterate until the sample error of the mean drops below the target tol_aim
#         while tol_sem > tol_aim:
#
#             # Select a small random subset of 100 points in the volume
#             curr_points = np.random.choice(len(df_morph), size=100, replace=False)
#
#             # Calculate Moran's I and Geary's C for the point subset
#             moran_i, geary_c = geospatial(df_morph=df_morph.loc[curr_points, :], spacing=spacing)
#
#             # Append values to the lists
#             moran_list.append(moran_i)
#             geary_list.append(geary_c)
#
#             # From the tenth iteration, estimate the sample error of the mean
#             if iter > 10:
#                 tol_sem = np.max( [np.std(moran_list), np.std(geary_list)] ) / np.sqrt(iter)
#
#             # Update counter
#             iter += 1
#
#             del curr_points, moran_i, geary_c
#
#         # Calculate approximate Moran's I and Geary's C
#         df_feat.ix[0, "morph_moran_i"] = np.mean(moran_list)
#         df_feat.ix[0, "morph_geary_c"] = np.mean(geary_list)
#
#         del iter
#
#     return df_feat
#
#
# def locIntFeatures(img_obj, roi_obj, spacing, cval):
#     # Calculate local intensity features
#
#     import scipy.ndimage.filters
#
#     # Create feature table
#     feat_names = ["loc_peak_loc", "loc_peak_glob"]
#     df_feat = pd.DataFrame(np.zeros([1, len(feat_names)]))
#     df_feat.columns = feat_names
#
#     if np.sum(roi_obj) > 300:
#         ################################################################################################################
#         # Calculate spheroid mean using a custom convolution filter
#         ################################################################################################################
#
#         # Determine distance
#         dist = (3.0/(4.0*np.pi))**(1.0/3.0) * 10.0
#
#         # Get maximal extension in cubic space
#         base_ext = np.floor(dist / spacing)
#
#         # Create displacement map
#         df_base = pd.DataFrame({"x": rep(x=np.arange(-base_ext[2], base_ext[2] + 1),
#                                          each=(2 * base_ext[0] + 1) * (2 * base_ext[1] + 1), times=1),
#                                 "y": rep(x=np.arange(-base_ext[1], base_ext[1] + 1), each=2 * base_ext[0] + 1,
#                                          times=2 * base_ext[2] + 1),
#                                 "z": rep(x=np.arange(-base_ext[0], base_ext[0] + 1), each=1,
#                                          times=(2 * base_ext[1] + 1) * (2 * base_ext[2] + 1))})
#
#         # Calculate distances for displacement map
#         df_base["dist"] = np.sqrt(np.sum(np.multiply(df_base.loc[:, ("z", "y", "x")].values, spacing) ** 2.0, axis=1))
#
#         # Identify elements in range
#         df_base["set_weight"] = df_base.dist <= dist
#
#         # Set weights for filter
#         df_base["weight"] = np.zeros(len(df_base))
#         df_base.loc[df_base.set_weight==True, "weight"] = 1.0 / np.sum(df_base.set_weight)
#
#         # Update coordinates to start at 0
#         df_base.loc[:, ["x", "y", "z"]] -= df_base.loc[0, ["x", "y", "z"]]
#
#         # Generate convolution filter
#         conv_filter = np.zeros(shape=(np.max(df_base.z).astype(np.int)+1, np.max(df_base.y).astype(np.int)+1, np.max(df_base.x).astype(np.int)+1))
#         conv_filter[df_base.z.astype(np.int), df_base.y.astype(np.int), df_base.x.astype(np.int)] = df_base.weight
#
#         # Filter image using mean filter
#         img_avg = scipy.ndimage.filters.convolve(img_obj, weights=conv_filter, mode="constant", cval=cval)
#
#         # Construct data frame for comparison
#         df_local = pd.DataFrame({"g":      np.ravel(img_obj),
#                                  "g_loc":  np.ravel(img_avg),
#                                  "in_roi": np.ravel(roi_obj)==1})
#
#     else:
#         ################################################################################################################
#         # Calculate mean iteratively for small volumes
#         ################################################################################################################
#
#         # Determine distance
#         dist = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * 10.0
#
#         # Construct data frame for comparison
#         df_local = pd.DataFrame({"g":      np.ravel(img_obj),
#                                  "g_loc":  np.full(img_obj.size, np.nan),
#                                  "in_roi": np.ravel(roi_obj)==1})
#
#         # Generate position matrix
#         pos_mat = np.array(np.unravel_index(indices=np.arange(start=0, stop=np.prod(np.shape(img_obj))), dims=np.shape(img_obj)), dtype=np.float64).transpose()
#
#         # Iterate over voxels in the roi
#         for i in np.array(np.where(df_local.in_roi==True)).squeeze():
#
#             # Determine distance from currently selected voxel
#             vox_dist = np.sqrt(np.sum(np.power(np.multiply(pos_mat - pos_mat[i,:], spacing), 2.0), axis=1))
#
#             # Calculate mean grey level over all voxels within range
#             df_local.loc[i, "g_loc"] = np.mean(df_local.g[vox_dist <= dist])
#
#     ####################################################################################################################
#     # Grey level peak features
#     ####################################################################################################################
#
#     # Shrink df_local to only contain roi voxels
#     df_local = df_local.loc[df_local.in_roi==True,:]
#
#     # Global grey level peak
#     df_feat.ix[0, "loc_peak_glob"] = np.max(df_local.g_loc)
#
#     # Local grey level peak
#     df_feat.ix[0, "loc_peak_loc"] = np.max(df_local.loc[df_local.g==np.max(df_local.g), "g_loc"])
#
#     return df_feat
#
#
# def glcmFeatures(list_glcm, g_range):
#     # Calculate grey level co-occurrence features
#
#     # Create feature table
#     feat_names = ["cm_joint_max", "cm_joint_avg", "cm_joint_var", "cm_joint_entr",
#                   "cm_diff_avg", "cm_diff_var", "cm_diff_entr",
#                   "cm_sum_avg", "cm_sum_var", "cm_sum_entr",
#                   "cm_energy", "cm_contrast", "cm_dissimilarity",
#                   "cm_inv_diff", "cm_inv_diff_norm", "cm_inv_diff_mom", "cm_inv_diff_mom_norm",
#                   "cm_inv_var", "cm_corr", "cm_auto_corr",
#                   "cm_clust_tend", "cm_clust_shade", "cm_clust_prom", "cm_info_corr1", "cm_info_corr2"]
#     df_feat = pd.DataFrame(np.zeros([len(list_glcm), len(feat_names)]))
#     df_feat.columns = feat_names
#
#     # Iterate over contents of GLCM list
#     for m in np.arange(start=0, stop=len(list_glcm)):
#
#         # Occurrence data frames
#         df_pij = copy.deepcopy(list_glcm[m])
#         df_pij["pij"] = df_pij.n / sum(df_pij.n)
#         df_pi = df_pij.groupby(by="i")["pij"].agg({"pi": np.sum}).reset_index()
#         df_pj = df_pij.groupby(by="j")["pij"].agg({"pj": np.sum}).reset_index()
#
#         # Diagonal probilities p(i-j)
#         df_pimj = copy.deepcopy(df_pij)
#         df_pimj["k"] = np.abs(df_pimj.i - df_pimj.j)
#         df_pimj = df_pimj.groupby(by="k")["pij"].agg({"pimj": np.sum}).reset_index()
#
#         # Cross-diagonal probabilities p(i+j)
#         df_pipj = copy.deepcopy(df_pij)
#         df_pipj["k"] = df_pipj.i + df_pipj.j
#         df_pipj = df_pipj.groupby(by="k")["pij"].agg({"pipj": np.sum}).reset_index()
#
#         # Merger of df.p_ij, df.p_i and df.p_j
#         df_pij = pd.merge(df_pij, df_pi, on="i")
#         df_pij = pd.merge(df_pij, df_pj, on="j")
#
#         # Constant definitions
#         g_range_loc = copy.deepcopy(g_range)
#         if np.isnan(g_range[0]): g_range_loc[0] = np.min(df_pi.i) * 1.0
#         if np.isnan(g_range[1]): g_range_loc[1] = np.max(df_pi.i) * 1.0
#         n_g = g_range_loc[1] - g_range_loc[0] + 1.0  # Number of grey levels
#
#         ###############################################
#         # GLCM features
#         ###############################################
#
#         # Joint maximum
#         df_feat.ix[m, "cm_joint_max"] = np.max(df_pij.pij)
#
#         # Joint average
#         df_feat.ix[m, "cm_joint_avg"] = np.sum(df_pij.i * df_pij.pij)
#
#         # Joint variance
#         mu = np.sum(df_pij.i * df_pij.pij)
#         df_feat.ix[m, "cm_joint_var"] = np.sum((df_pij.i - mu) ** 2.0 * df_pij.pij)
#
#         # Joint entropy
#         df_feat.ix[m, "cm_joint_entr"] = -np.sum(df_pij.pij * np.log2(df_pij.pij))
#
#         # Difference average
#         df_feat.ix[m, "cm_diff_avg"] = np.sum(df_pimj.k * df_pimj.pimj)
#
#         # Difference variance
#         mu = np.sum(df_pimj.k * df_pimj.pimj)
#         df_feat.ix[m, "cm_diff_var"] = np.sum((df_pimj.k - mu) ** 2.0 * df_pimj.pimj)
#
#         # Difference entropy
#         df_feat.ix[m, "cm_diff_entr"] = -np.sum(df_pimj.pimj * np.log2(df_pimj.pimj))
#
#         # Sum average
#         df_feat.ix[m, "cm_sum_avg"] = np.sum(df_pipj.k * df_pipj.pipj)
#
#         # Sum variance
#         mu = np.sum(df_pipj.k * df_pipj.pipj)
#         df_feat.ix[m, "cm_sum_var"] = np.sum((df_pipj.k - mu) ** 2.0 * df_pipj.pipj)
#
#         # Sum entropy
#         df_feat.ix[m, "cm_sum_entr"] = -np.sum(df_pipj.pipj * np.log2(df_pipj.pipj))
#
#         # Angular second moment
#         df_feat.ix[m, "cm_energy"] = np.sum(df_pij.pij ** 2.0)
#
#         # Contrast
#         df_feat.ix[m, "cm_contrast"] = np.sum((df_pij.i - df_pij.j) ** 2.0 * df_pij.pij)
#
#         # Dissimilarity
#         df_feat.ix[m, "cm_dissimilarity"] = np.sum(np.abs(df_pij.i - df_pij.j) * df_pij.pij)
#
#         # Inverse difference
#         df_feat.ix[m, "cm_inv_diff"] = np.sum(df_pij.pij / (1.0 + np.abs(df_pij.i - df_pij.j)))
#
#         # Inverse difference normalised
#         df_feat.ix[m, "cm_inv_diff_norm"] = np.sum(df_pij.pij / (1.0 + np.abs(df_pij.i - df_pij.j) / n_g))
#
#         # Inverse difference moment
#         df_feat.ix[m, "cm_inv_diff_mom"] = np.sum(df_pij.pij / (1.0 + (df_pij.i - df_pij.j) ** 2.0))
#
#         # Inverse difference moment normalised
#         df_feat.ix[m, "cm_inv_diff_mom_norm"] = np.sum(df_pij.pij / (1.0 + (df_pij.i - df_pij.j) ** 2.0 / n_g ** 2.0))
#
#         # Inverse variance
#         df_sel = df_pij[df_pij.i != df_pij.j]
#         df_feat.ix[m, "cm_inv_var"] = np.sum(df_sel.pij / (df_sel.i - df_sel.j) ** 2.0)
#         del df_sel
#
#         # Correlation
#         mu_marg = np.sum(df_pi.i * df_pi.pi)
#         var_marg = np.sum((df_pi.i - mu_marg) ** 2.0 * df_pi.pi)
#         df_feat.ix[m, "cm_corr"] = 1.0 / var_marg * (np.sum(df_pij.i * df_pij.j * df_pij.pij) - mu_marg ** 2.0)
#
#         del mu_marg, var_marg
#
#         # Autocorrelation
#         df_feat.ix[m, "cm_auto_corr"] = np.sum(df_pij.i * df_pij.j * df_pij.pij)
#
#         # Information correlation 1
#         hxy = - np.sum(df_pij.pij * np.log2(df_pij.pij))
#         hxy_1 = - np.sum(df_pij.pij * np.log2(df_pij.pi * df_pij.pj))
#         hx = - np.sum(df_pi.pi * np.log2(df_pi.pi))
#         df_feat.ix[m, "cm_info_corr1"] = (hxy - hxy_1) / hx
#         del hxy, hxy_1, hx
#
#         # Information correlation 2 - Note: iteration over combinations of i and j
#         hxy = - np.sum(df_pij.pij * np.log2(df_pij.pij))
#         hxy_2 = - np.sum(np.tile(df_pi.pi, len(df_pj)) * np.repeat(df_pj.pj, len(df_pi)) * np.log2(
#             np.tile(df_pi.pi, len(df_pj)) * np.repeat(df_pj.pj, len(df_pi))))
#         #        hxy_2 = - np.sum(df_pij.pi  * df_pij.pj * np.log2(df_pij.pi * df_pij.pj))
#         if hxy_2 < hxy:
#             df_feat.ix[m, "cm_info_corr2"] = 0
#         else:
#             df_feat.ix[m, "cm_info_corr2"] = np.sqrt(1 - np.exp(-2.0 * (hxy_2 - hxy)))
#         del hxy, hxy_2
#
#         # Cluster tendency
#         mu = np.sum(df_pi.i * df_pi.pi)
#         df_feat.ix[m, "cm_clust_tend"] = np.sum((df_pij.i + df_pij.j - 2 * mu) ** 2.0 * df_pij.pij)
#         del mu
#
#         # Cluster shade
#         mu = np.sum(df_pi.i * df_pi.pi)
#         df_feat.ix[m, "cm_clust_shade"] = np.sum((df_pij.i + df_pij.j - 2 * mu) ** 3.0 * df_pij.pij)
#         del mu
#
#         # Cluster prominence
#         mu = np.sum(df_pi.i * df_pi.pi)
#         df_feat.ix[m, "cm_clust_prom"] = np.sum((df_pij.i + df_pij.j - 2 * mu) ** 4.0 * df_pij.pij)
#
#         del df_pi, df_pj, df_pij, df_pimj, df_pipj, n_g
#
#     # Average over directions; maintain as dataframe
#     df_feat = df_feat.mean(axis=0).to_frame().transpose()
#
#     return df_feat
#
#
# def glrlmFeatures(list_glrlm, rlm_n_v):
#     # Calculate grey level run length features
#
#     # Create feature table
#     feat_names = ["rlm_sre", "rlm_lre", "rlm_lgre", "rlm_hgre", "rlm_srlge", "rlm_srhge", "rlm_lrlge", "rlm_lrhge",
#                   "rlm_glnu", "rlm_glnu_norm", "rlm_rlnu", "rlm_rlnu_norm", "rlm_r_perc",
#                   "rlm_gl_var", "rlm_rl_var", "rlm_rl_entr"]
#     df_feat = pd.DataFrame(np.zeros([len(list_glrlm), len(feat_names)]))
#     df_feat.columns = feat_names
#
#     # Iterate over contents of GLRLM list
#     for m in np.arange(start=0, stop=len(list_glrlm)):
#         # Occurrence dataframes
#         df_rij = copy.deepcopy(list_glrlm[m])
#         df_rij.columns = ["i", "j", "rij"]
#
#         # Sum over grey levels
#         df_ri = df_rij.groupby(by="i")["rij"].agg({"ri": np.sum}).reset_index()
#
#         # Sum over run lengths
#         df_rj = df_rij.groupby(by="j")["rij"].agg({"rj": np.sum}).reset_index()
#
#         # Constant definitions
#         n_s = np.sum(df_rij.rij) * 1.0  # Number of runs
#         n_v = rlm_n_v[m] * 1.0  # Number of voxels
#
#         ###############################################
#         # GLRLM features
#         ###############################################
#
#         # Short runs emphasis
#         df_feat.ix[m, "rlm_sre"] = np.sum(df_rj.rj / df_rj.j ** 2.0) / n_s
#
#         # Long runs emphasis
#         df_feat.ix[m, "rlm_lre"] = np.sum(df_rj.rj * df_rj.j ** 2.0) / n_s
#
#         # Grey level non-uniformity
#         df_feat.ix[m, "rlm_glnu"] = np.sum(df_ri.ri ** 2.0) / n_s
#
#         # Grey level non-uniformity, normalised
#         df_feat.ix[m, "rlm_glnu_norm"] = np.sum(df_ri.ri ** 2.0) / n_s ** 2.0
#
#         # Run length non-uniformity
#         df_feat.ix[m, "rlm_rlnu"] = np.sum(df_rj.rj ** 2.0) / n_s
#
#         # Run length non-uniformity
#         df_feat.ix[m, "rlm_rlnu_norm"] = np.sum(df_rj.rj ** 2.0) / n_s ** 2.0
#
#         # Run percentage
#         df_feat.ix[m, "rlm_r_perc"] = n_s / n_v
#
#         # Low grey level run emphasis
#         df_feat.ix[m, "rlm_lgre"] = np.sum(df_ri.ri / df_ri.i ** 2.0) / n_s
#
#         # High grey level run emphasis
#         df_feat.ix[m, "rlm_hgre"] = np.sum(df_ri.ri * df_ri.i ** 2.0) / n_s
#
#         # Short run low grey level emphasis
#         df_feat.ix[m, "rlm_srlge"] = np.sum(df_rij.rij / (df_rij.i * df_rij.j) ** 2.0) / n_s
#
#         # Short run high grey level emphasis
#         df_feat.ix[m, "rlm_srhge"] = np.sum(df_rij.rij * df_rij.i ** 2.0 / df_rij.j ** 2.0) / n_s
#
#         # Long run low grey level emphasis
#         df_feat.ix[m, "rlm_lrlge"] = np.sum(df_rij.rij * df_rij.j ** 2.0 / df_rij.i ** 2.0) / n_s
#
#         # Long run high grey level emphasis
#         df_feat.ix[m, "rlm_lrhge"] = np.sum(df_rij.rij * df_rij.i ** 2.0 * df_rij.j ** 2.0) / n_s
#
#         # Grey level variance
#         mu = np.sum(df_rij.rij * df_rij.i) / n_s
#         df_feat.ix[m, "rlm_gl_var"] = np.sum((df_rij.i - mu) ** 2.0 * df_rij.rij) / n_s
#
#         # Run length variance
#         mu = np.sum(df_rij.rij * df_rij.j) / n_s
#         df_feat.ix[m, "rlm_rl_var"] = np.sum((df_rij.j - mu) ** 2.0 * df_rij.rij) / n_s
#
#         # Zone size entropy
#         df_feat.ix[m, "rlm_rl_entr"] = - np.sum(df_rij.rij * np.log2(df_rij.rij / n_s)) / n_s
#
#         del mu, n_s, n_v, df_ri, df_rj, df_rij
#
#         # Average over directions
#     df_feat = df_feat.mean(axis=0).to_frame().transpose()
#
#     return df_feat
#
#
# def glszmFeatures(list_glszm, szm_n_v):
#     # Calculate grey level size zone features
#
#     # Create feature table
#     feat_names = ["szm_sze", "szm_lze", "szm_lgze", "szm_hgze", "szm_szlge", "szm_szhge", "szm_lzlge", "szm_lzhge",
#                   "szm_glnu", "szm_glnu_norm", "szm_zsnu", "szm_zsnu_norm", "szm_z_perc",
#                   "szm_gl_var", "szm_zs_var", "szm_zs_entr"]
#     df_feat = pd.DataFrame(np.zeros([len(list_glszm), len(feat_names)]))
#     df_feat.columns = feat_names
#
#     # Iterate over size zone matrices in list_glszm
#     for m in np.arange(start=0, stop=len(list_glszm)):
#         # Occurrence dataframe
#         df_sij = copy.deepcopy(list_glszm[m])
#         df_sij.columns = ("i", "j", "sij")
#
#         # Sum over grey levels
#         df_si = df_sij.groupby(by="i")["sij"].agg({"si": np.sum}).reset_index()
#
#         # Sum over zone sizes
#         df_sj = df_sij.groupby(by="j")["sij"].agg({"sj": np.sum}).reset_index()
#
#         # Constant definitions
#         n_s = np.sum(df_sij.sij) * 1.0  # Number of size zones
#         n_v = szm_n_v[m]  # Number of voxels
#
#         ###############################################
#         # GLSZM features
#         ###############################################
#
#         # Small zone emphasis
#         df_feat.ix[m, "szm_sze"] = np.sum(df_sj.sj / df_sj.j ** 2.0) / n_s
#
#         # Large zone emphasis
#         df_feat.ix[m, "szm_lze"] = np.sum(df_sj.sj * df_sj.j ** 2.0) / n_s
#
#         # Grey level non-uniformity
#         df_feat.ix[m, "szm_glnu"] = np.sum(df_si.si ** 2.0) / n_s
#
#         # Grey level non-uniformity, normalised
#         df_feat.ix[m, "szm_glnu_norm"] = np.sum(df_si.si ** 2.0) / n_s ** 2.0
#
#         # Zone size non-uniformity
#         df_feat.ix[m, "szm_zsnu"] = np.sum(df_sj.sj ** 2.0) / n_s
#
#         # Zone size non-uniformity
#         df_feat.ix[m, "szm_zsnu_norm"] = np.sum(df_sj.sj ** 2.0) / n_s ** 2.0
#
#         # Zone percentage
#         df_feat.ix[m, "szm_z_perc"] = n_s / n_v
#
#         # Low grey level emphasis
#         df_feat.ix[m, "szm_lgze"] = np.sum(df_si.si / df_si.i ** 2.0) / n_s
#
#         # High grey level emphasis
#         df_feat.ix[m, "szm_hgze"] = np.sum(df_si.si * df_si.i ** 2.0) / n_s
#
#         # Small zone low grey level emphasis
#         df_feat.ix[m, "szm_szlge"] = np.sum(df_sij.sij / (df_sij.i * df_sij.j) ** 2.0) / n_s
#
#         # Small zone high grey level emphasis
#         df_feat.ix[m, "szm_szhge"] = np.sum(df_sij.sij * df_sij.i ** 2.0 / df_sij.j ** 2.0) / n_s
#
#         # Large zone low grey level emphasis
#         df_feat.ix[m, "szm_lzlge"] = np.sum(df_sij.sij * df_sij.j ** 2.0 / df_sij.i ** 2.0) / n_s
#
#         # Large zone high grey level emphasis
#         df_feat.ix[m, "szm_lzhge"] = np.sum(df_sij.sij * df_sij.i ** 2.0 * df_sij.j ** 2.0) / n_s
#
#         # Grey level variance
#         mu = np.sum(df_sij.sij * df_sij.i) / n_s
#         df_feat.ix[m, "szm_gl_var"] = np.sum((df_sij.i - mu) ** 2.0 * df_sij.sij) / n_s
#         del mu
#
#         # Zone size variance
#         mu = np.sum(df_sij.sij * df_sij.j) / n_s
#         df_feat.ix[m, "szm_zs_var"] = np.sum((df_sij.j - mu) ** 2.0 * df_sij.sij) / n_s
#         del mu
#
#         # Zone size entropy
#         df_feat.ix[m, "szm_zs_entr"] = - np.sum(df_sij.sij * np.log2(df_sij.sij / n_s)) / n_s
#
#     # Average over directions
#     df_feat = df_feat.mean(axis=0).to_frame().transpose()
#
#     return df_feat
#
#
# def gldzmFeatures(list_gldzm, dzm_n_v):
#     # Calculate grey level distance zone matrix features
#
#     # Create feature table
#     feat_names = ["dzm_sde", "dzm_lde", "dzm_lgze", "dzm_hgze", "dzm_sdlge", "dzm_sdhge", "dzm_ldlge", "dzm_ldhge",
#                   "dzm_glnu", "dzm_glnu_norm", "dzm_zdnu", "dzm_zdnu_norm", "dzm_z_perc",
#                   "dzm_gl_var", "dzm_zd_var", "dzm_zd_entr"]
#     df_feat = pd.DataFrame(np.zeros([len(list_gldzm), len(feat_names)]))
#     df_feat.columns = feat_names
#
#     # Iterate over size zone matrices in list_gldzm
#     for m in np.arange(start=0, stop=len(list_gldzm)):
#         # Occurrence dataframe
#         df_dij = copy.deepcopy(list_gldzm[m])
#         df_dij.columns = ("i", "j", "dij")
#
#         # Sum over grey levels
#         df_di = df_dij.groupby(by="i")["dij"].agg({"di": np.sum}).reset_index()
#
#         # Sum over zone distances
#         df_dj = df_dij.groupby(by="j")["dij"].agg({"dj": np.sum}).reset_index()
#
#         # Constant definitions
#         n_s = np.sum(df_dij.dij) * 1.0  # Number of size zones
#         n_v = dzm_n_v[m]  # Number of voxels
#
#         ###############################################
#         # GLDZM features
#         ###############################################
#
#         # Small distance emphasis
#         df_feat.ix[m, "dzm_sde"] = np.sum(df_dj.dj / df_dj.j ** 2.0) / n_s
#
#         # Large distance emphasis
#         df_feat.ix[m, "dzm_lde"] = np.sum(df_dj.dj * df_dj.j ** 2.0) / n_s
#
#         # Grey level non-uniformity
#         df_feat.ix[m, "dzm_glnu"] = np.sum(df_di.di ** 2.0) / n_s
#
#         # Grey level non-uniformity, normalised
#         df_feat.ix[m, "dzm_glnu_norm"] = np.sum(df_di.di ** 2.0) / n_s ** 2.0
#
#         # Zone distance non-uniformity
#         df_feat.ix[m, "dzm_zdnu"] = np.sum(df_dj.dj ** 2.0) / n_s
#
#         # Zone distance non-uniformity
#         df_feat.ix[m, "dzm_zdnu_norm"] = np.sum(df_dj.dj ** 2.0) / n_s ** 2.0
#
#         # Zone percentage
#         df_feat.ix[m, "dzm_z_perc"] = n_s / n_v
#
#         # Low grey level emphasis
#         df_feat.ix[m, "dzm_lgze"] = np.sum(df_di.di / df_di.i ** 2.0) / n_s
#
#         # High grey level emphasis
#         df_feat.ix[m, "dzm_hgze"] = np.sum(df_di.di * df_di.i ** 2.0) / n_s
#
#         # Small distance low grey level emphasis
#         df_feat.ix[m, "dzm_sdlge"] = np.sum(df_dij.dij / (df_dij.i * df_dij.j) ** 2.0) / n_s
#
#         # Small distance high grey level emphasis
#         df_feat.ix[m, "dzm_sdhge"] = np.sum(df_dij.dij * df_dij.i ** 2.0 / df_dij.j ** 2.0) / n_s
#
#         # Large distance low grey level emphasis
#         df_feat.ix[m, "dzm_ldlge"] = np.sum(df_dij.dij * df_dij.j ** 2.0 / df_dij.i ** 2.0) / n_s
#
#         # Large distance high grey level emphasis
#         df_feat.ix[m, "dzm_ldhge"] = np.sum(df_dij.dij * df_dij.i ** 2.0 * df_dij.j ** 2.0) / n_s
#
#         # Grey level variance
#         mu = np.sum(df_dij.dij * df_dij.i) / n_s
#         df_feat.ix[m, "dzm_gl_var"] = np.sum((df_dij.i - mu) ** 2.0 * df_dij.dij) / n_s
#         del mu
#
#         # Zone distance variance
#         mu = np.sum(df_dij.dij * df_dij.j) / n_s
#         df_feat.ix[m, "dzm_zd_var"] = np.sum((df_dij.j - mu) ** 2.0 * df_dij.dij) / n_s
#         del mu
#
#         # Zone distance entropy
#         df_feat.ix[m, "dzm_zd_entr"] = - np.sum(df_dij.dij * np.log2(df_dij.dij / n_s)) / n_s
#
#     # Average over directions
#     df_feat = df_feat.mean(axis=0).to_frame().transpose()
#
#     return df_feat
#
#
# def ngtdmFeatures(list_ngtdm, ngt_n_v, g_range):
#     # Calculate neighbourhood grey tone difference features
#
#     # Create feature table
#     feat_names = ["ngt_coarseness", "ngt_contrast", "ngt_busyness", "ngt_complexity", "ngt_strength"]
#     df_feat = pd.DataFrame(np.zeros([len(list_ngtdm), len(feat_names)]))
#     df_feat.columns = feat_names
#
#     for m in np.arange(start=0, stop=len(list_ngtdm)):
#
#         # Occurrence dataframe
#         df_pi = copy.deepcopy(list_ngtdm[m])
#         df_pi["pi"] = df_pi.n / np.sum(df_pi.n)
#
#         # Constant definitions
#         n_v = ngt_n_v[m] * 1.0  # Number of voxels
#         n_p = len(df_pi) * 1.0  # Number of valid grey levels
#
#         g_range_loc = copy.deepcopy(g_range)
#         if np.isnan(g_range[0]): g_range_loc[0] = np.min(df_pi.i) * 1.0
#         if np.isnan(g_range[1]): g_range_loc[1] = np.max(df_pi.i) * 1.0
#         n_g = g_range_loc[1] - g_range_loc[0] + 1.0  # Number of grey levels
#
#         # Append empty grey levels to
#         levels = np.arange(start=0, stop=n_g) + 1.0
#         miss_level = levels[np.logical_not(np.in1d(levels, df_pi.i))]
#         n_miss = len(miss_level)
#         if n_miss > 0:
#             df_pi = df_pi.append(
#                 pd.DataFrame({"i": miss_level, "s": np.zeros(n_miss), "n": np.zeros(n_miss), "pi": np.zeros(n_miss)}),
#                 ignore_index=True)
#         del levels, miss_level, n_miss
#
#         # Compose occurrence correspondence table
#         df_pij = copy.deepcopy(df_pi)
#         df_pij = df_pij.rename(columns={"s": "si"})
#         df_pij = df_pij.iloc[rep(np.arange(start=0, stop=n_g), each=n_g).astype(np.int), :]
#         df_pij["j"] = rep(df_pi.i, each=1, times=n_g)
#         df_pij["pj"] = rep(df_pi.pi, each=1, times=n_g)
#         df_pij["sj"] = rep(df_pi.s, each=1, times=n_g)
#         df_pij = df_pij.loc[(df_pij.pi > 0) & (df_pij.pj > 0), :].reset_index()
#
#         ###############################################
#         # NGTDM features
#         ###############################################
#
#         # Coarseness
#         if np.sum(df_pi.pi * df_pi.s) < 1E-6:
#             df_feat.ix[m, "ngt_coarseness"] = 1.0 / 1E-6
#         else:
#             df_feat.ix[m, "ngt_coarseness"] = 1.0 / np.sum(df_pi.pi * df_pi.s)
#
#         # Contrast
#         if n_p > 1.0:
#             df_feat.ix[m, "ngt_contrast"] = np.sum(df_pij.pi * df_pij.pj * (df_pij.i - df_pij.j) ** 2.0) / (
#             n_p * (n_p - 1.0)) * np.sum(df_pi.s) / n_v
#         else:
#             df_feat.ix[m, "ngt_contrast"] = 0.0
#
#         # Busyness
#         if n_p > 1.0:
#             df_feat.ix[m, "ngt_busyness"] = np.sum(df_pi.pi * df_pi.s) / (
#             np.sum(np.abs(df_pij.i * df_pij.pi - df_pij.j * df_pij.pj)))
#         else:
#             df_feat.ix[m, "ngt_busyness"] = 0.0
#
#         # Complexity
#         df_feat.ix[m, "ngt_complexity"] = np.sum(
#             np.abs(df_pij.i - df_pij.j) * (df_pij.pi * df_pij.si + df_pij.pj * df_pij.sj) / (
#             df_pij.pi + df_pij.pj)) / n_v
#
#         # Strength
#         if np.sum(df_pi.s) > 0.0:
#             df_feat.ix[m, "ngt_strength"] = np.sum((df_pij.pi + df_pij.pj) * (df_pij.i - df_pij.j) ** 2.0) / np.sum(
#                 df_pi.s)
#         else:
#             df_feat.ix[m, "ngt_strength"] = 0.0
#
#     # Average over directions
#     df_feat = df_feat.mean(axis=0).to_frame().transpose()
#
#     return df_feat
#
#
# def ngldmFeatures(list_ngldm, ngl_n_v):
#     # Calculate neighbouring grey level dependence matrix features
#
#     # Create feature table
#     feat_names = ["ngl_lde", "ngl_hde", "ngl_lgce", "ngl_hgce", "ngl_ldlge", "ngl_ldhge", "ngl_hdlge", "ngl_hdhge",
#                   "ngl_glnu", "ngl_glnu_norm", "ngl_dcnu", "ngl_dcnu_norm", "ngl_dc_perc",
#                   "ngl_gl_var", "ngl_dc_var", "ngl_dc_entr", "ngl_dc_energy"]
#     df_feat = pd.DataFrame(np.zeros([len(list_ngldm), len(feat_names)]))
#     df_feat.columns = feat_names
#
#     # Iterate over neighbouring grey level dependence matrices in list_ngldm
#     for m in np.arange(start=0, stop=len(list_ngldm)):
#         # Dependence count dataframe
#         df_sij = copy.deepcopy(list_ngldm[m])
#         df_sij.columns = ("i", "j", "sij")
#
#         # Sum over grey levels
#         df_si = df_sij.groupby(by="i")["sij"].agg({"si": np.sum}).reset_index()
#
#         # Sum over dependence counts
#         df_sj = df_sij.groupby(by="j")["sij"].agg({"sj": np.sum}).reset_index()
#
#         # Constant definitions
#         n_s = np.sum(df_sij.sij) * 1.0  # Number of neighbourhoods considered
#         n_v = ngl_n_v[m]  # Number of voxels
#
#         ###############################################
#         # NGLDM features
#         ###############################################
#
#         # Low dependence emphasis
#         df_feat.ix[m, "ngl_lde"] = np.sum(df_sj.sj / df_sj.j ** 2.0) / n_s
#
#         # High dependence emphasis
#         df_feat.ix[m, "ngl_hde"] = np.sum(df_sj.sj * df_sj.j ** 2.0) / n_s
#
#         # Grey level non-uniformity
#         df_feat.ix[m, "ngl_glnu"] = np.sum(df_si.si ** 2.0) / n_s
#
#         # Grey level non-uniformity, normalised
#         df_feat.ix[m, "ngl_glnu_norm"] = np.sum(df_si.si ** 2.0) / n_s ** 2.0
#
#         # Dependence count non-uniformity
#         df_feat.ix[m, "ngl_dcnu"] = np.sum(df_sj.sj ** 2.0) / n_s
#
#         # Dependence count non-uniformity, normalised
#         df_feat.ix[m, "ngl_dcnu_norm"] = np.sum(df_sj.sj ** 2.0) / n_s ** 2.0
#
#         # Dependence count percentage
#         df_feat.ix[m, "ngl_dc_perc"] = n_s / n_v
#
#         # Low grey level count emphasis
#         df_feat.ix[m, "ngl_lgce"] = np.sum(df_si.si / df_si.i ** 2.0) / n_s
#
#         # High grey level count emphasis
#         df_feat.ix[m, "ngl_hgce"] = np.sum(df_si.si * df_si.i ** 2.0) / n_s
#
#         # Low dependence low grey level emphasis
#         df_feat.ix[m, "ngl_ldlge"] = np.sum(df_sij.sij / (df_sij.i * df_sij.j) ** 2.0) / n_s
#
#         # Low dependence high grey level emphasis
#         df_feat.ix[m, "ngl_ldhge"] = np.sum(df_sij.sij * df_sij.i ** 2.0 / df_sij.j ** 2.0) / n_s
#
#         # High dependence low grey level emphasis
#         df_feat.ix[m, "ngl_hdlge"] = np.sum(df_sij.sij * df_sij.j ** 2.0 / df_sij.i ** 2.0) / n_s
#
#         # High dependence high grey level emphasis
#         df_feat.ix[m, "ngl_hdhge"] = np.sum(df_sij.sij * df_sij.i ** 2.0 * df_sij.j ** 2.0) / n_s
#
#         # Grey level variance
#         mu = np.sum(df_sij.sij * df_sij.i) / n_s
#         df_feat.ix[m, "ngl_gl_var"] = np.sum((df_sij.i - mu) ** 2.0 * df_sij.sij) / n_s
#         del mu
#
#         # Dependence count variance
#         mu = np.sum(df_sij.sij * df_sij.j) / n_s
#         df_feat.ix[m, "ngl_dc_var"] = np.sum((df_sij.j - mu) ** 2.0 * df_sij.sij) / n_s
#         del mu
#
#         # Dependence count entropy
#         df_feat.ix[m, "ngl_dc_entr"] = - np.sum(df_sij.sij * np.log2(df_sij.sij / n_s)) / n_s
#
#         # Dependence count energy
#         df_feat.ix[m, "ngl_dc_energy"] = np.sum(df_sij.sij ** 2.0) / (n_s ** 2.0)
#
#     # Average over directions
#     df_feat = df_feat.mean(axis=0).to_frame().transpose()
#
#     return df_feat
