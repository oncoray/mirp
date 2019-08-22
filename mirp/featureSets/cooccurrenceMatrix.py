import numpy as np
import pandas as pd
from mirp.featureSets.utilities import get_neighbour_directions, is_list_all_none, coord2Index, get_intensity_value
import copy


def get_cm_features(img_obj, roi_obj, settings):
    """Extract cooccurrence matrix-based features from the intensity roi"""

    # Get a table of the roi intensity mask
    df_img = roi_obj.as_pandas_dataframe(img_obj=img_obj, intensity_mask=True)

    if df_img is None:
        # In case the input image or ROI are missing.
        n_slices = 1
        img_dims = None
    else:
        # Default case with input image and ROI available
        n_slices = img_obj.size[0]
        img_dims = np.array(img_obj.size)

    # Generate an empty feature list
    feat_list = []

    # Iterate over spatial arrangements
    for ii_spatial in settings.feature_extr.glcm_spatial_method:

        # Iterate over distances
        for ii_dist in settings.feature_extr.glcm_dist:

            # Initiate list of glcm objects
            glcm_list = []

            # Perform 2D analysis
            if ii_spatial.lower() in ["2d", "2.5d"]:

                # Get neighbour directions
                nbrs = get_neighbour_directions(d=1, distance="chebyshev", centre=False, complete=False, dim3=False) * np.int(ii_dist)

                # Iterate over slices
                for ii_slice in np.arange(0, n_slices):

                    # Iterate over neighbours
                    for ii_direction in np.arange(0, np.shape(nbrs)[1]):

                        # Add glcm matrices to list
                        glcm_list += [CooccurrenceMatrix(distance=np.int(ii_dist), direction=nbrs[:, ii_direction], direction_id=ii_direction,
                                                         spatial_method=ii_spatial.lower(), slice_id=ii_slice)]
            # Perform 3D analysis
            if ii_spatial.lower() == "3d":

                # Get neighbour direction and iterate over neighbours
                nbrs = get_neighbour_directions(d=1, distance="chebyshev", centre=False, complete=False, dim3=True) * np.int(ii_dist)
                for ii_direction in np.arange(0, np.shape(nbrs)[1]):

                    # Add glcm matrices to list
                    glcm_list += [CooccurrenceMatrix(distance=np.int(ii_dist), direction=nbrs[:, ii_direction], direction_id=ii_direction,
                                                     spatial_method=ii_spatial.lower())]

            # Calculate glcm matrices
            for glcm in glcm_list: glcm.calculate_matrix(df_img=df_img, img_dims=img_dims)

            # Merge matrices according to the given method
            for merge_method in settings.feature_extr.glcm_merge_method:
                upd_list = combine_matrices(glcm_list=glcm_list, merge_method=merge_method, spatial_method=ii_spatial.lower())

                # Skip if no matrices are available (due to illegal combinations of merge and spatial methods
                if upd_list is None:
                    continue

                # Calculate features
                feat_run_list = []
                for glcm in upd_list:
                    feat_run_list += [glcm.compute_features(g_range=roi_obj.g_range)]

                # Average feature values
                feat_list += [pd.concat(feat_run_list, axis=0).mean(axis=0, skipna=True).to_frame().transpose()]

    # Merge feature tables into a single table
    df_feat = pd.concat(feat_list, axis=1)

    return df_feat


def combine_matrices(glcm_list, merge_method, spatial_method):
    """Function to combine glcm matrices prior to feature calculation."""

    # Initiate empty list
    use_list = []

    if merge_method == "average" and spatial_method in ["2d", "3d"]:
        # For average features over direction, maintain original glcms

        # Make copy of glcm_list
        for glcm in glcm_list: use_list += [glcm.copy()]

        # Set merge method to average
        for glcm in use_list: glcm.merge_method = "average"

    elif merge_method == "slice_merge" and spatial_method == "2d":
        # Merge glcms by slice

        # Find slice_ids
        slice_id = []
        for glcm in glcm_list: slice_id += [glcm.slice]

        # Iterate over unique slice_ids
        for ii_slice in np.unique(slice_id):
            slice_glcm_id = np.squeeze(np.where(slice_id == ii_slice))

            # Select all matrices within the slice
            sel_matrix_list = []
            for glcm_id in slice_glcm_id: sel_matrix_list += [glcm_list[glcm_id].matrix]

            # Check if any matrix has been created for the currently selected slice
            if is_list_all_none(sel_matrix_list):
                # No matrix was created
                use_list += [CooccurrenceMatrix(distance=glcm_list[slice_glcm_id[0]].distance, direction=None, direction_id=None,
                                                spatial_method=spatial_method, slice_id=ii_slice, merge_method=merge_method,
                                                matrix=None, n_v=0.0)]
            else:
                # Merge matrices within the slice
                merge_cm = pd.concat(sel_matrix_list, axis=0)
                merge_cm = merge_cm.groupby(by=["i", "j"]).sum().reset_index()

                # Update the number of voxels within the merged slice
                merge_n_v = 0.0
                for glcm_id in slice_glcm_id: merge_n_v += glcm_list[glcm_id].n_v

                # Create new cooccurrence matrix
                use_list += [CooccurrenceMatrix(distance=glcm_list[slice_glcm_id[0]].distance, direction=None, direction_id=None,
                                                spatial_method=spatial_method, slice_id=ii_slice, merge_method=merge_method,
                                                matrix=merge_cm, n_v=merge_n_v)]

    elif merge_method == "dir_merge" and spatial_method == "2.5d":
        # Merge glcms by direction

        # Find slice_ids
        dir_id = []
        for glcm in glcm_list: dir_id += [glcm.direction_id]

        # Iterate over unique directions
        for ii_dir in np.unique(dir_id):
            dir_glcm_id = np.squeeze(np.where(dir_id == ii_dir))

            # Select all matrices with the same direction
            sel_matrix_list = []
            for glcm_id in dir_glcm_id: sel_matrix_list += [glcm_list[glcm_id].matrix]

            # Check if any matrix has been created for the currently selected direction
            if is_list_all_none(sel_matrix_list):
                # No matrix was created
                use_list += [CooccurrenceMatrix(distance=glcm_list[dir_glcm_id[0]].distance, direction=glcm_list[dir_glcm_id[0]].direction, direction_id=ii_dir,
                                                spatial_method=spatial_method, slice_id=None, merge_method=merge_method,
                                                matrix=None, n_v=0.0)]
            else:
                # Merge matrices with the same direction
                merge_cm = pd.concat(sel_matrix_list, axis=0)
                merge_cm = merge_cm.groupby(by=["i", "j"]).sum().reset_index()

                # Update the number of voxels for the merged matrices with the same direction
                merge_n_v = 0.0
                for glcm_id in dir_glcm_id: merge_n_v += glcm_list[glcm_id].n_v

                # Create new cooccurrence matrix
                use_list += [CooccurrenceMatrix(distance=glcm_list[dir_glcm_id[0]].distance, direction=glcm_list[dir_glcm_id[0]].direction, direction_id=ii_dir,
                                                spatial_method=spatial_method, slice_id=None, merge_method=merge_method,
                                                matrix=merge_cm, n_v=merge_n_v)]

    elif merge_method == "vol_merge" and spatial_method in ["2.5d", "3d"]:
        # Merge all glcms into a single representation

        # Select all matrices within the slice
        sel_matrix_list = []
        for glcm_id in np.arange(len(glcm_list)): sel_matrix_list += [glcm_list[glcm_id].matrix]

        # Check if any matrix has been created
        if is_list_all_none(sel_matrix_list):
            # No matrix was created
            use_list += [CooccurrenceMatrix(distance=glcm_list[0].distance, direction=None, direction_id=None,
                                            spatial_method=spatial_method, slice_id=None, merge_method=merge_method,
                                            matrix=None, n_v=0.0)]
        else:
            # Merge cooccurrence matrices
            merge_cm = pd.concat(sel_matrix_list, axis=0)
            merge_cm = merge_cm.groupby(by=["i", "j"]).sum().reset_index()

            # Update the number of voxels
            merge_n_v = 0.0
            for glcm_id in np.arange(len(glcm_list)): merge_n_v += glcm_list[glcm_id].n_v

            # Create new cooccurrence matrix
            use_list += [CooccurrenceMatrix(distance=glcm_list[0].distance, direction=None, direction_id=None,
                                            spatial_method=spatial_method, slice_id=None, merge_method=merge_method,
                                            matrix=merge_cm, n_v=merge_n_v)]
    else:
        use_list = None

    return use_list


class CooccurrenceMatrix:

    def __init__(self, distance, direction, direction_id, spatial_method, slice_id=None, merge_method=None, matrix=None, n_v=None):

        # Distance used
        self.distance = distance

        # Direction and slice for which the current matrix is extracted
        self.direction = direction
        self.direction_id = direction_id
        self.slice = slice_id

        # Spatial analysis method (2d, 2.5d, 3d) and merge method (average, slice_merge, dir_merge, vol_merge)
        self.spatial_method = spatial_method
        self.merge_method = merge_method

        # Place holders
        self.matrix = matrix
        self.n_v = n_v

    def copy(self):
        return copy.deepcopy(self)

    def set_empty(self):
        self.n_v = 0
        self.matrix = None

    def calculate_matrix(self, df_img, img_dims):

        # Check if the df_img actually exists
        if df_img is None:
            self.set_empty()
            return

        # Check if the roi contains any masked voxels. If this is not the case, don't construct the GLCM.
        if not np.any(df_img.roi_int_mask):
            self.set_empty()
            return

        # Create local copies of the image table
        if self.spatial_method == "3d":
            df_cm = copy.deepcopy(df_img)
        elif self.spatial_method in ["2d", "2.5d"]:
            df_cm = copy.deepcopy(df_img[df_img.z == self.slice])
            df_cm["index_id"] = np.arange(0, len(df_cm))
            df_cm["z"] = 0
            df_cm = df_cm.reset_index(drop=True)
        else:
            raise ValueError("The spatial method for grey level co-occurrence matrices should be one of \"2d\", \"2.5d\" or \"3d\".")

        # Set grey level of voxels outside ROI to NaN
        df_cm.loc[df_cm.roi_int_mask == False, "g"] = np.nan

        # Determine potential transitions
        df_cm["to_index"] = coord2Index(x=df_cm.x.values + self.direction[2],
                                             y=df_cm.y.values + self.direction[1],
                                             z=df_cm.z.values + self.direction[0],
                                             dims=img_dims)

        # Get grey levels from transitions
        df_cm["to_g"] = get_intensity_value(x=df_cm.g.values, index=df_cm.to_index.values)

        # Check if any transitions exist.
        if np.all(np.isnan(df_cm[["to_g"]])):
            self.set_empty()
            return

        # Count occurrences of grey level transitions
        df_cm = df_cm.groupby(by=["g", "to_g"]).size().reset_index(name="n")

        # Append grey level transitions in opposite direction
        df_cm_inv = pd.DataFrame({"g": df_cm.to_g, "to_g": df_cm.g, "n": df_cm.n})
        df_cm = df_cm.append(df_cm_inv, ignore_index=True)

        # Sum occurrences of grey level transitions
        df_cm = df_cm.groupby(by=["g", "to_g"]).sum().reset_index()

        # Rename columns
        df_cm.columns = ["i", "j", "n"]

        # Set the number of voxels
        self.n_v = np.sum(df_cm.n)

        # Add matrix and number of voxels to object
        self.matrix = df_cm

    def compute_features(self, g_range):

        # Create feature table
        feat_names = ["cm_joint_max", "cm_joint_avg", "cm_joint_var", "cm_joint_entr",
                      "cm_diff_avg", "cm_diff_var", "cm_diff_entr",
                      "cm_sum_avg", "cm_sum_var", "cm_sum_entr",
                      "cm_energy", "cm_contrast", "cm_dissimilarity",
                      "cm_inv_diff", "cm_inv_diff_norm", "cm_inv_diff_mom", "cm_inv_diff_mom_norm",
                      "cm_inv_var", "cm_corr", "cm_auto_corr",
                      "cm_clust_tend", "cm_clust_shade", "cm_clust_prom", "cm_info_corr1", "cm_info_corr2"]
        df_feat = pd.DataFrame(np.full(shape=(1, len(feat_names)), fill_value=np.nan))
        df_feat.columns = feat_names

        # Don't return data for empty slices or slices without a good matrix
        if self.matrix is None:
            # Update names
            df_feat.columns += self.parse_feature_names()
            return df_feat
        elif len(self.matrix) == 0:
            # Update names
            df_feat.columns += self.parse_feature_names()
            return df_feat

        # Occurrence data frames
        df_pij = copy.deepcopy(self.matrix)
        df_pij["pij"] = df_pij.n / sum(df_pij.n)
        df_pi = df_pij.groupby(by="i")["pij"].agg(np.sum).reset_index().rename(columns={"pij": "pi"})
        df_pj = df_pij.groupby(by="j")["pij"].agg(np.sum).reset_index().rename(columns={"pij": "pj"})

        # Diagonal probilities p(i-j)
        df_pimj = copy.deepcopy(df_pij)
        df_pimj["k"] = np.abs(df_pimj.i - df_pimj.j)
        df_pimj = df_pimj.groupby(by="k")["pij"].agg(np.sum).reset_index().rename(columns={"pij": "pimj"})

        # Cross-diagonal probabilities p(i+j)
        df_pipj = copy.deepcopy(df_pij)
        df_pipj["k"] = df_pipj.i + df_pipj.j
        df_pipj = df_pipj.groupby(by="k")["pij"].agg(np.sum).reset_index().rename(columns={"pij": "pipj"})

        # Merger of df.p_ij, df.p_i and df.p_j
        df_pij = pd.merge(df_pij, df_pi, on="i")
        df_pij = pd.merge(df_pij, df_pj, on="j")

        # Constant definitions
        g_range_loc = copy.deepcopy(g_range)
        if np.isnan(g_range[0]): g_range_loc[0] = np.min(df_pi.i) * 1.0
        if np.isnan(g_range[1]): g_range_loc[1] = np.max(df_pi.i) * 1.0
        n_g = g_range_loc[1] - g_range_loc[0] + 1.0  # Number of grey levels

        ###############################################
        # GLCM features
        ###############################################

        # Joint maximum
        df_feat["cm_joint_max"] = np.max(df_pij.pij)

        # Joint average
        df_feat["cm_joint_avg"] = np.sum(df_pij.i * df_pij.pij)

        # Joint variance
        mu = np.sum(df_pij.i * df_pij.pij)
        df_feat["cm_joint_var"] = np.sum((df_pij.i - mu) ** 2.0 * df_pij.pij)

        # Joint entropy
        df_feat["cm_joint_entr"] = -np.sum(df_pij.pij * np.log2(df_pij.pij))

        # Difference average
        df_feat["cm_diff_avg"] = np.sum(df_pimj.k * df_pimj.pimj)

        # Difference variance
        mu = np.sum(df_pimj.k * df_pimj.pimj)
        df_feat["cm_diff_var"] = np.sum((df_pimj.k - mu) ** 2.0 * df_pimj.pimj)

        # Difference entropy
        df_feat["cm_diff_entr"] = -np.sum(df_pimj.pimj * np.log2(df_pimj.pimj))

        # Sum average
        df_feat["cm_sum_avg"] = np.sum(df_pipj.k * df_pipj.pipj)

        # Sum variance
        mu = np.sum(df_pipj.k * df_pipj.pipj)
        df_feat["cm_sum_var"] = np.sum((df_pipj.k - mu) ** 2.0 * df_pipj.pipj)

        # Sum entropy
        df_feat["cm_sum_entr"] = -np.sum(df_pipj.pipj * np.log2(df_pipj.pipj))

        # Angular second moment
        df_feat["cm_energy"] = np.sum(df_pij.pij ** 2.0)

        # Contrast
        df_feat["cm_contrast"] = np.sum((df_pij.i - df_pij.j) ** 2.0 * df_pij.pij)

        # Dissimilarity
        df_feat["cm_dissimilarity"] = np.sum(np.abs(df_pij.i - df_pij.j) * df_pij.pij)

        # Inverse difference
        df_feat["cm_inv_diff"] = np.sum(df_pij.pij / (1.0 + np.abs(df_pij.i - df_pij.j)))

        # Inverse difference normalised
        df_feat["cm_inv_diff_norm"] = np.sum(df_pij.pij / (1.0 + np.abs(df_pij.i - df_pij.j) / n_g))

        # Inverse difference moment
        df_feat["cm_inv_diff_mom"] = np.sum(df_pij.pij / (1.0 + (df_pij.i - df_pij.j) ** 2.0))

        # Inverse difference moment normalised
        df_feat["cm_inv_diff_mom_norm"] = np.sum(
            df_pij.pij / (1.0 + (df_pij.i - df_pij.j) ** 2.0 / n_g ** 2.0))

        # Inverse variance
        df_sel = df_pij[df_pij.i != df_pij.j]
        df_feat["cm_inv_var"] = np.sum(df_sel.pij / (df_sel.i - df_sel.j) ** 2.0)
        del df_sel

        # Correlation
        mu_marg = np.sum(df_pi.i * df_pi.pi)
        var_marg = np.sum((df_pi.i - mu_marg) ** 2.0 * df_pi.pi)

        if var_marg == 0.0:
            df_feat["cm_corr"] = 1.0
        else:
            df_feat["cm_corr"] = 1.0 / var_marg * (np.sum(df_pij.i * df_pij.j * df_pij.pij) - mu_marg ** 2.0)

        del mu_marg, var_marg

        # Autocorrelation
        df_feat["cm_auto_corr"] = np.sum(df_pij.i * df_pij.j * df_pij.pij)

        # Information correlation 1
        hxy = - np.sum(df_pij.pij * np.log2(df_pij.pij))
        hxy_1 = - np.sum(df_pij.pij * np.log2(df_pij.pi * df_pij.pj))
        hx = - np.sum(df_pi.pi * np.log2(df_pi.pi))
        if len(df_pij) == 1 or hx == 0.0:
            df_feat["cm_info_corr1"] = 1.0
        else:
            df_feat["cm_info_corr1"] = (hxy - hxy_1) / hx
        del hxy, hxy_1, hx

        # Information correlation 2 - Note: iteration over combinations of i and j
        hxy = - np.sum(df_pij.pij * np.log2(df_pij.pij))
        hxy_2 = - np.sum(np.tile(df_pi.pi, len(df_pj)) * np.repeat(df_pj.pj, len(df_pi)) * np.log2(
                         np.tile(df_pi.pi, len(df_pj)) * np.repeat(df_pj.pj, len(df_pi))))
        #        hxy_2 = - np.sum(df_pij.pi  * df_pij.pj * np.log2(df_pij.pi * df_pij.pj))
        if hxy_2 < hxy:
            df_feat["cm_info_corr2"] = 0
        else:
            df_feat["cm_info_corr2"] = np.sqrt(1 - np.exp(-2.0 * (hxy_2 - hxy)))
        del hxy, hxy_2

        # Cluster tendency
        mu = np.sum(df_pi.i * df_pi.pi)
        df_feat["cm_clust_tend"] = np.sum((df_pij.i + df_pij.j - 2 * mu) ** 2.0 * df_pij.pij)
        del mu

        # Cluster shade
        mu = np.sum(df_pi.i * df_pi.pi)
        df_feat["cm_clust_shade"] = np.sum((df_pij.i + df_pij.j - 2 * mu) ** 3.0 * df_pij.pij)
        del mu

        # Cluster prominence
        mu = np.sum(df_pi.i * df_pi.pi)
        df_feat["cm_clust_prom"] = np.sum((df_pij.i + df_pij.j - 2 * mu) ** 4.0 * df_pij.pij)

        del df_pi, df_pj, df_pij, df_pimj, df_pipj, n_g

        # Update names
        df_feat.columns += self.parse_feature_names()

        return df_feat

    def parse_feature_names(self):
        """"Used for parsing names to feature names"""
        parse_str = [""]

        # Add distance
        parse_str += ["d" + str(np.round(self.distance, 1))]

        # Add spatial method
        if self.spatial_method is not None:
            parse_str += [self.spatial_method]

        # Add merge method
        if self.merge_method is not None:
            if self.merge_method == "average":     parse_str += ["avg"]
            if self.merge_method == "slice_merge": parse_str += ["s_mrg"]
            if self.merge_method == "dir_merge":   parse_str += ["d_mrg"]
            if self.merge_method == "vol_merge":   parse_str += ["v_mrg"]

        return "_".join(parse_str).rstrip("_")
