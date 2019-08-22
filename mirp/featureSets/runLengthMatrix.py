import copy

import numpy as np
import pandas as pd

from mirp.featureSets.utilities import get_neighbour_directions, is_list_all_none, coord2Index


def get_rlm_features(img_obj, roi_obj, settings):
    """Extract run length matrix-based features from the intensity roi"""

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
    for ii_spatial in settings.feature_extr.glrlm_spatial_method:

        # Initiate list of rlm objects
        rlm_list = []

        # Perform 2D analysis
        if ii_spatial.lower() in ["2d", "2.5d"]:

            # Iterate over slices
            for ii_slice in np.arange(0, n_slices):

                # Get neighbour direction and iterate over neighbours
                nbrs = get_neighbour_directions(d=1, distance="chebyshev", centre=False, complete=False, dim3=False)
                for ii_direction in np.arange(0, np.shape(nbrs)[1]):

                    # Add rlm matrices to list
                    rlm_list += [RunLengthMatrix(direction=nbrs[:, ii_direction], direction_id=ii_direction, spatial_method=ii_spatial.lower(), slice_id=ii_slice)]

        # Perform 3D analysis
        if ii_spatial.lower() == "3d":

            # Get neighbour direction and iterate over neighbours
            nbrs = get_neighbour_directions(d=1, distance="chebyshev", centre=False, complete=False, dim3=True)
            for ii_direction in np.arange(0, np.shape(nbrs)[1]):

                # Add rlm matrices to list
                rlm_list += [RunLengthMatrix(direction=nbrs[:, ii_direction], direction_id=ii_direction, spatial_method=ii_spatial.lower())]

        # Calculate run length matrices
        for rlm in rlm_list: rlm.calculate_matrix(df_img=df_img, img_dims=img_dims)

        # Merge matrices according to the given method
        for merge_method in settings.feature_extr.glrlm_merge_method:
            upd_list = combine_matrices(rlm_list=rlm_list, merge_method=merge_method, spatial_method=ii_spatial.lower())

            # Skip if no matrices are available (due to illegal combinations of merge and spatial methods
            if upd_list is None: continue

            # Calculate features
            feat_run_list = []
            for rlm in upd_list: feat_run_list += [rlm.compute_features(g_range=roi_obj.g_range)]

            # Average feature values
            feat_list += [pd.concat(feat_run_list, axis=0).mean(axis=0, skipna=True).to_frame().transpose()]

    # Merge feature tables into a single table
    df_feat = pd.concat(feat_list, axis=1)

    return df_feat


def combine_matrices(rlm_list, merge_method, spatial_method):
    """Function to combine rlm matrices prior to feature calculation."""

    # Initiate empty list
    use_list = []

    if merge_method == "average" and spatial_method in ["2d", "3d"]:
        # For average features over direction, maintain original run length matrices

        # Make copy of rlm_list
        for rlm in rlm_list: use_list += [rlm.copy()]

        # Set merge method to average
        for rlm in use_list: rlm.merge_method = "average"

    elif merge_method == "slice_merge" and spatial_method == "2d":
        # Merge rlms within each slice

        # Find slice_ids
        slice_id = []
        for rlm in rlm_list: slice_id += [rlm.slice]

        # Iterate over unique slice_ids
        for ii_slice in np.unique(slice_id):
            slice_rlm_id = np.squeeze(np.where(slice_id == ii_slice))

            # Select all matrices within the slice
            sel_matrix_list = []
            for rlm_id in slice_rlm_id: sel_matrix_list += [rlm_list[rlm_id].matrix]

            # Check if any matrix has been created for the currently selected slice
            if is_list_all_none(sel_matrix_list):
                # No matrix was created
                use_list += [RunLengthMatrix(direction=None, direction_id=None, spatial_method=spatial_method, slice_id=ii_slice,
                                             merge_method=merge_method, matrix=None, n_v=0.0)]
            else:
                # Merge matrices within the slice
                merge_rlm = pd.concat(sel_matrix_list, axis=0)
                merge_rlm = merge_rlm.groupby(by=["i", "r"]).sum().reset_index()

                # Update the number of voxels within the merged slice
                merge_n_v = 0.0
                for rlm_id in slice_rlm_id: merge_n_v += rlm_list[rlm_id].n_v

                # Create new run length matrix
                use_list += [RunLengthMatrix(direction=None, direction_id=None, spatial_method=spatial_method, slice_id=ii_slice,
                                             merge_method=merge_method, matrix=merge_rlm, n_v=merge_n_v)]

    elif merge_method == "dir_merge" and spatial_method == "2.5d":
        # Merge rlms within each slice

        # Find direction ids
        dir_id = []
        for rlm in rlm_list: dir_id += [rlm.direction_id]

        # Iterate over unique dir_ids
        for ii_dir in np.unique(dir_id):
            dir_rlm_id = np.squeeze(np.where(dir_id == ii_dir))

            # Select all matrices with the same direction
            sel_matrix_list = []
            for rlm_id in dir_rlm_id: sel_matrix_list += [rlm_list[rlm_id].matrix]

            # Check if any matrix has been created for the currently selected direction
            if is_list_all_none(sel_matrix_list):
                # No matrix was created
                use_list += [RunLengthMatrix(direction=rlm_list[dir_rlm_id[0]].direction, direction_id=ii_dir, spatial_method=spatial_method, slice_id=None,
                                             merge_method=merge_method, matrix=None, n_v=0.0)]
            else:
                # Merge matrices with the same direction
                merge_rlm = pd.concat(sel_matrix_list, axis=0)
                merge_rlm = merge_rlm.groupby(by=["i", "r"]).sum().reset_index()

                # Update the number of voxels within the merged slice
                merge_n_v = 0.0
                for rlm_id in dir_rlm_id: merge_n_v += rlm_list[rlm_id].n_v

                # Create new run length matrix
                use_list += [RunLengthMatrix(direction=rlm_list[dir_rlm_id[0]].direction, direction_id=ii_dir, spatial_method=spatial_method, slice_id=None,
                                             merge_method=merge_method, matrix=merge_rlm, n_v=merge_n_v)]

    elif merge_method == "vol_merge" and spatial_method in ["2.5d", "3d"]:
        # Merge all rlms into a single representation

        # Select all matrices within the slice
        sel_matrix_list = []
        for rlm_id in np.arange(len(rlm_list)): sel_matrix_list += [rlm_list[rlm_id].matrix]

        # Check if any matrix has been created
        if is_list_all_none(sel_matrix_list):
            # No matrix was created
            use_list += [RunLengthMatrix(direction=None, direction_id=None, spatial_method=spatial_method, slice_id=None,
                                         merge_method=merge_method, matrix=None, n_v=0.0)]
        else:
            # Merge run length matrices
            merge_rlm = pd.concat(sel_matrix_list, axis=0)
            merge_rlm = merge_rlm.groupby(by=["i", "r"]).sum().reset_index()

            # Update the number of voxels
            merge_n_v = 0.0
            for rlm_id in np.arange(len(rlm_list)): merge_n_v += rlm_list[rlm_id].n_v

            # Create new run length matrix
            use_list += [RunLengthMatrix(direction=None, direction_id=None, spatial_method=spatial_method, slice_id=None,
                                         merge_method=merge_method, matrix=merge_rlm, n_v=merge_n_v)]

    else:
        use_list = None

    # Return to new rlm list to calling function
    return use_list


class RunLengthMatrix:

    def __init__(self, direction, direction_id, spatial_method, slice_id=None, merge_method=None, matrix=None, n_v=None):

        # Direction and slice for which the current matrix is extracted
        self.direction = direction
        self.direction_id = direction_id
        self.slice = slice_id

        # Spatial analysis method (2d, 2.5d, 3d) and merge method (average, slice_merge, dir_merge, vol_merge)
        self.spatial_method = spatial_method

        # Place holders
        self.merge_method = merge_method
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

        # Check if the roi contains any masked voxels. If this is not the case, don't construct the GLRLM.
        if not np.any(df_img.roi_int_mask):
            self.set_empty()
            return

        # Create local copies of the image table
        if self.spatial_method == "3d":
            df_rlm = copy.deepcopy(df_img)
        elif self.spatial_method in ["2d", "2.5d"]:
            df_rlm = copy.deepcopy(df_img[df_img.z == self.slice])
            df_rlm["index_id"] = np.arange(0, len(df_rlm))
            df_rlm["z"] = 0
            df_rlm = df_rlm.reset_index(drop=True)
        else:
            raise ValueError("The spatial method for grey level run length matrices should be one of \"2d\", \"2.5d\" or \"3d\".")

        # Set grey level of voxels outside ROI to NaN
        df_rlm.loc[df_rlm.roi_int_mask == False, "g"] = np.nan

        # Set the number of voxels
        self.n_v = np.sum(df_rlm.roi_int_mask.values)

        # Determine update index number for direction
        if (self.direction[2] + self.direction[1] * img_dims[2] + self.direction[0] * img_dims[2] * img_dims[1]) >= 0:
            curr_dir = self.direction
        else:
            curr_dir = - self.direction

        # Step size
        ind_update = curr_dir[2] + curr_dir[1] * img_dims[2] + curr_dir[0] * img_dims[2] * img_dims[1]

        # Generate information concerning segments
        n_seg = ind_update  # Number of segments

        # Check if the number of segments is greater than one
        if n_seg == 0:
            self.set_empty()
            return

        seg_len = (len(df_rlm) - 1) // ind_update + 1  # Nominal segment length
        trans_seg_len = np.tile([seg_len - 1], reps=n_seg)  # Initial segment length for transitions (nominal length - 1)
        full_len_trans = n_seg - n_seg * seg_len + len(df_rlm)  # Number of full segments
        trans_seg_len[0:full_len_trans] += 1  # Update full segments

        # Create transition vector
        trans_vec = np.tile(np.arange(start=0, stop=len(df_rlm), step=ind_update), reps=ind_update) + np.repeat(np.arange(start=0, stop=n_seg), repeats=seg_len)
        trans_vec = trans_vec[trans_vec < len(df_rlm)]

        # Determine valid transitions
        to_index = coord2Index(x=df_rlm.x.values + curr_dir[2],
                                    y=df_rlm.y.values + curr_dir[1],
                                    z=df_rlm.z.values + curr_dir[0],
                                    dims=img_dims)

        # Determine which transitions are valid
        end_ind = np.nonzero(to_index[trans_vec] < 0)[0]  # Find transitions that form an endpoints

        # Get an interspersed array of intensities. Runs are broken up by np.nan
        intensities = np.insert(df_rlm.g.values[trans_vec], end_ind + 1, np.nan)

        # Determine run length start and end indices
        rle_end = np.array(np.append(np.where(intensities[1:] != intensities[:-1]), len(intensities) - 1))
        rle_start = np.cumsum(np.append(0, np.diff(np.append(-1, rle_end))))[:-1]

        # Generate dataframe
        df_rltable = pd.DataFrame({"i": intensities[rle_start],
                                   "r": rle_end - rle_start + 1})
        df_rltable = df_rltable.loc[~np.isnan(df_rltable.i), :]
        df_rltable = df_rltable.groupby(by=["i", "r"]).size().reset_index(name="n")

        # Add matrix to object
        self.matrix = df_rltable

    def compute_features(self, g_range):

        # Create feature table
        feat_names = ["rlm_sre", "rlm_lre", "rlm_lgre", "rlm_hgre", "rlm_srlge", "rlm_srhge", "rlm_lrlge", "rlm_lrhge",
                      "rlm_glnu", "rlm_glnu_norm", "rlm_rlnu", "rlm_rlnu_norm", "rlm_r_perc",
                      "rlm_gl_var", "rlm_rl_var", "rlm_rl_entr"]
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

        # Create local copy of the run length matrix and set column names
        df_rij = copy.deepcopy(self.matrix)
        df_rij.columns = ["i", "j", "rij"]

        # Sum over grey levels
        df_ri = df_rij.groupby(by="i")["rij"].agg(np.sum).reset_index().rename(columns={"rij": "ri"})

        # Sum over run lengths
        df_rj = df_rij.groupby(by="j")["rij"].agg(np.sum).reset_index().rename(columns={"rij": "rj"})

        # Constant definitions
        n_s = np.sum(df_rij.rij) * 1.0  # Number of runs
        n_v = self.n_v * 1.0  # Number of voxels

        ###############################################
        # GLRLM features
        ###############################################

        # Short runs emphasis
        df_feat["rlm_sre"] = np.sum(df_rj.rj / df_rj.j ** 2.0) / n_s

        # Long runs emphasis
        df_feat["rlm_lre"] = np.sum(df_rj.rj * df_rj.j ** 2.0) / n_s

        # Grey level non-uniformity
        df_feat["rlm_glnu"] = np.sum(df_ri.ri ** 2.0) / n_s

        # Grey level non-uniformity, normalised
        df_feat["rlm_glnu_norm"] = np.sum(df_ri.ri ** 2.0) / n_s ** 2.0

        # Run length non-uniformity
        df_feat["rlm_rlnu"] = np.sum(df_rj.rj ** 2.0) / n_s

        # Run length non-uniformity
        df_feat["rlm_rlnu_norm"] = np.sum(df_rj.rj ** 2.0) / n_s ** 2.0

        # Run percentage
        df_feat["rlm_r_perc"] = n_s / n_v

        # Low grey level run emphasis
        df_feat["rlm_lgre"] = np.sum(df_ri.ri / df_ri.i ** 2.0) / n_s

        # High grey level run emphasis
        df_feat["rlm_hgre"] = np.sum(df_ri.ri * df_ri.i ** 2.0) / n_s

        # Short run low grey level emphasis
        df_feat["rlm_srlge"] = np.sum(df_rij.rij / (df_rij.i * df_rij.j) ** 2.0) / n_s

        # Short run high grey level emphasis
        df_feat["rlm_srhge"] = np.sum(df_rij.rij * df_rij.i ** 2.0 / df_rij.j ** 2.0) / n_s

        # Long run low grey level emphasis
        df_feat["rlm_lrlge"] = np.sum(df_rij.rij * df_rij.j ** 2.0 / df_rij.i ** 2.0) / n_s

        # Long run high grey level emphasis
        df_feat["rlm_lrhge"] = np.sum(df_rij.rij * df_rij.i ** 2.0 * df_rij.j ** 2.0) / n_s

        # Grey level variance
        mu = np.sum(df_rij.rij * df_rij.i) / n_s
        df_feat["rlm_gl_var"] = np.sum((df_rij.i - mu) ** 2.0 * df_rij.rij) / n_s

        # Run length variance
        mu = np.sum(df_rij.rij * df_rij.j) / n_s
        df_feat["rlm_rl_var"] = np.sum((df_rij.j - mu) ** 2.0 * df_rij.rij) / n_s

        # Zone size entropy
        df_feat["rlm_rl_entr"] = - np.sum(df_rij.rij * np.log2(df_rij.rij / n_s)) / n_s

        # Update names
        df_feat.columns += self.parse_feature_names()

        return df_feat

    def parse_feature_names(self):
        """"Used for parsing names to feature names"""
        parse_str = [""]

        # Add spatial method
        if self.spatial_method is not None:
            parse_str += [self.spatial_method]

        # Add merge method
        if self.merge_method is not None:
            if self.merge_method == "average": parse_str += ["avg"]
            if self.merge_method == "slice_merge": parse_str += ["s_mrg"]
            if self.merge_method == "dir_merge":   parse_str += ["d_mrg"]
            if self.merge_method == "vol_merge":   parse_str += ["v_mrg"]

        return "_".join(parse_str).rstrip("_")
