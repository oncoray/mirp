import copy

import numpy as np
import pandas as pd

from mirp._featuresets.utilities import get_neighbour_directions, is_list_all_none, coord2Index, get_intensity_value, rep
from mirp._images.generic_image import GenericImage
from mirp._masks.baseMask import BaseMask
from mirp.settings.settingsFeatureExtraction import FeatureExtractionSettingsClass


def get_ngtdm_features(
        image: GenericImage,
        mask: BaseMask,
        settings: FeatureExtractionSettingsClass
) -> pd.DataFrame:
    """Extract neighbourhood grey tone difference-based features from the intensity mask"""

    # Get a table of the roi intensity mask
    df_img = mask.as_pandas_dataframe(image=image, intensity_mask=True)

    if df_img is None:
        # In case the input image or ROI are missing.
        n_slices = 1
        img_dims = None
    else:
        # Default case with input image and ROI available
        n_slices = image.image_dimension[0]
        img_dims = np.array(image.image_dimension)

    # Generate an empty feature list
    feat_list = []

    # Iterate over spatial arrangements
    for ii_spatial in settings.ngtdm_spatial_method:

        # Initiate list of neighbourhood grey tone difference matrix objects
        ngtdm_list = []

        # Perform 2D analysis
        if ii_spatial.lower() in ["2d", "2.5d"]:

            # Iterate over slices
            for ii_slice in np.arange(0, n_slices):

                # Perform analysis per slice
                ngtdm_list += [GreyToneDifferenceMatrix(
                    spatial_method=ii_spatial.lower(),
                    slice_id=ii_slice)]

        # Perform 3D analysis
        elif ii_spatial.lower() == "3d":
            # Perform analysis on the entire volume
            ngtdm_list += [GreyToneDifferenceMatrix(
                spatial_method=ii_spatial.lower(),
                slice_id=None)]

        else:
            raise ValueError("Spatial methods for NGTDM should be \"2d\", \"2.5d\" or \"3d\".")

        # Calculate size zone matrices
        for ngtdm in ngtdm_list:
            ngtdm.calculate_matrix(
                df_img=df_img,
                img_dims=img_dims)

        # Merge matrices according to the given method
        upd_list = combine_matrices(
            ngtdm_list=ngtdm_list,
            spatial_method=ii_spatial.lower())

        # Calculate features
        feat_run_list = []
        for ngtdm in upd_list:
            feat_run_list += [ngtdm.compute_features(g_range=np.array(mask.intensity_range))]

        # Average feature values
        feat_list += [pd.concat(feat_run_list, axis=0).mean(axis=0, skipna=True).to_frame().transpose()]

    # Merge feature tables into a single table
    feature_data = pd.concat(feat_list, axis=1)

    return feature_data


def combine_matrices(ngtdm_list, spatial_method):
    """Function to combine neighbourhood grey tone difference matrices prior to feature calculation."""

    # Initiate empty list
    use_list = []

    if spatial_method == "2d":
        # Average features over slices, maintain original neighbourhood grey tone difference matrices

        # Make copy of ngtdm_list
        use_list = []
        for ngtdm in ngtdm_list:
            use_list += [ngtdm.copy()]

        return use_list

    elif spatial_method in ["2.5d", "3d"]:
        # Merge all neighbourhood grey tone difference matrices into a single representation

        # Select all matrices within the slice
        sel_matrix_list = []
        for ngtdm_id in np.arange(len(ngtdm_list)):
            sel_matrix_list += [ngtdm_list[ngtdm_id].matrix]

        # Check if any matrix has been created
        if is_list_all_none(sel_matrix_list):
            # No matrix was created
            use_list += [GreyToneDifferenceMatrix(
                spatial_method=spatial_method,
                slice_id=None,
                matrix=None,
                n_v=0.0)]
        else:
            # Merge neighbourhood grey tone difference matrices
            merge_ngtdm = pd.concat(sel_matrix_list, axis=0)
            merge_ngtdm = merge_ngtdm.groupby(by=["i"]).agg({"s": np.sum, "n": np.sum}).reset_index()

            # Update the number of voxels
            merge_n_v = 0.0
            for ngtdm_id in np.arange(len(ngtdm_list)):
                merge_n_v += ngtdm_list[ngtdm_id].n_v

            # Create new neighbourhood grey tone difference matrix
            use_list += [GreyToneDifferenceMatrix(
                spatial_method=spatial_method,
                slice_id=None,
                matrix=merge_ngtdm,
                n_v=merge_n_v)]

    else:
        use_list = None

    # Return to new ngtdm list to calling function
    return use_list


class GreyToneDifferenceMatrix:

    def __init__(self, spatial_method, slice_id=None, matrix=None, n_v=None):

        # Slice id
        self.slice = slice_id

        # Spatial analysis method (2d, 2.5d, 3d)
        self.spatial_method = spatial_method

        # Placeholders
        self.matrix = matrix
        self.n_v = n_v

    def copy(self):
        return copy.deepcopy(self)

    def set_empty(self):
        self.n_v = 0
        self.matrix = None

    def calculate_matrix(self, df_img, img_dims):

        # Check if the input image and roi exist
        if df_img is None:
            self.set_empty()
            return

        # Check if the roi contains any masked voxels. If this is not the case, don't construct the NGTDM.
        if not np.any(df_img.roi_int_mask):
            self.set_empty()
            return

        # Define neighbour directions and update the df_img table
        if self.spatial_method == "3d":
            nbrs = get_neighbour_directions(d=1, distance="chebyshev", centre=False, complete=True, dim3=True)
            df_ngtdm = copy.deepcopy(df_img)

        elif self.spatial_method in ["2d", "2.5d"]:
            nbrs = get_neighbour_directions(d=1, distance="chebyshev", centre=False, complete=True, dim3=False)
            df_ngtdm = copy.deepcopy(df_img[df_img.z == self.slice])
            df_ngtdm["index_id"] = np.arange(0, len(df_ngtdm))
            df_ngtdm["z"] = 0
            df_ngtdm = df_ngtdm.reset_index(drop=True)

        else:
            raise ValueError(
                "The spatial method for neighbourhood grey tone difference matrices should be "
                "one of \"2d\", \"2.5d\" or \"3d\".")

        # Set grey level of voxels outside ROI to NaN
        df_ngtdm.loc[df_ngtdm.roi_int_mask == False, "g"] = np.nan

        # Initialise sum of grey levels and number of neighbours
        df_ngtdm["g_sum"] = 0.0
        df_ngtdm["n_nbrs"] = 0.0

        for k in np.arange(0, np.shape(nbrs)[1]):
            # Determine potential transitions from valid voxels
            df_ngtdm["to_index"] = coord2Index(
                x=df_ngtdm.x.values + nbrs[2, k],
                y=df_ngtdm.y.values + nbrs[1, k],
                z=df_ngtdm.z.values + nbrs[0, k],
                dims=img_dims)

            # Get grey level value from transitions
            df_ngtdm["to_g"] = get_intensity_value(x=df_ngtdm.g.values, index=df_ngtdm.to_index.values)

            # Determine which voxels have valid neighbours
            sel_index = np.isfinite(df_ngtdm.to_g)

            # Sum grey level and increase neighbour counter
            df_ngtdm.loc[sel_index, "g_sum"] += df_ngtdm.loc[sel_index, "to_g"]
            df_ngtdm.loc[sel_index, "n_nbrs"] += 1.0

        # Calculate average neighbourhood grey level
        df_ngtdm["g_nbr_avg"] = df_ngtdm.g_sum / df_ngtdm.n_nbrs

        # Work with voxels without a missing grey value and with a valid neighbourhood
        df_ngtdm = df_ngtdm[np.logical_and(np.isfinite(df_ngtdm.g_nbr_avg), df_ngtdm.roi_int_mask)]

        # Determine contribution to s per voxel
        df_ngtdm["s_sub"] = np.abs(df_ngtdm.g - df_ngtdm.g_nbr_avg)

        # Drop superfluous columns
        df_ngtdm = df_ngtdm.drop(
            labels=["index_id", "x", "y", "z", "g_sum", "n_nbrs", "to_index", "to_g", "g_nbr_avg", "roi_int_mask"],
            axis=1)

        # Sum s over voxels
        df_ngtdm = df_ngtdm.groupby(by="g")
        df_ngtdm = df_ngtdm.sum().join(pd.DataFrame(df_ngtdm.size(), columns=["n"])).reset_index()

        # Rename columns
        df_ngtdm.columns = ["i", "s", "n"]

        # Update number of voxels for current iteration
        self.n_v = np.sum(df_ngtdm.n)

        # Store matrix
        self.matrix = df_ngtdm

    def compute_features(self, g_range):

        # Create feature table
        feat_names = ["ngt_coarseness", "ngt_contrast", "ngt_busyness", "ngt_complexity", "ngt_strength"]
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

        # Occurrence dataframe
        df_pi = copy.deepcopy(self.matrix)
        df_pi["pi"] = df_pi.n / np.sum(df_pi.n)

        # Constant definitions
        n_v = self.n_v * 1.0  # Number of voxels
        n_p = len(df_pi) * 1.0  # Number of valid grey levels

        # Set range of grey levels
        g_range_loc = copy.deepcopy(g_range)
        if np.isnan(g_range[0]):
            g_range_loc[0] = np.min(df_pi.i) * 1.0
        if np.isnan(g_range[1]):
            g_range_loc[1] = np.max(df_pi.i) * 1.0

        n_g = g_range_loc[1] - g_range_loc[0] + 1.0  # Number of grey levels

        # Append empty grey levels to table
        levels = np.arange(start=0, stop=n_g) + 1.0
        miss_level = levels[np.logical_not(np.in1d(levels, df_pi.i))]
        n_miss = len(miss_level)
        if n_miss > 0:
            df_pi = pd.concat([
                df_pi,
                pd.DataFrame({
                    "i": miss_level,
                    "s": np.zeros(n_miss),
                    "n": np.zeros(n_miss),
                    "pi": np.zeros(n_miss)})
            ], ignore_index=True)

        del levels, miss_level, n_miss

        # Compose occurrence correspondence table
        df_pij = copy.deepcopy(df_pi)
        df_pij = df_pij.rename(columns={"s": "si"})
        df_pij = df_pij.iloc[rep(np.arange(start=0, stop=n_g), each=n_g).astype(int), :]
        df_pij["j"] = rep(df_pi.i, each=1, times=n_g)
        df_pij["pj"] = rep(df_pi.pi, each=1, times=n_g)
        df_pij["sj"] = rep(df_pi.s, each=1, times=n_g)
        df_pij = df_pij.loc[(df_pij.pi > 0) & (df_pij.pj > 0), :].reset_index()

        ###############################################
        # NGTDM features
        ###############################################

        # Coarseness
        if np.sum(df_pi.pi * df_pi.s) < 1E-6:
            df_feat["ngt_coarseness"] = 1.0 / 1E-6
        else:
            df_feat["ngt_coarseness"] = 1.0 / np.sum(df_pi.pi * df_pi.s)

        # Contrast
        if n_p > 1.0:
            df_feat["ngt_contrast"] = np.sum(df_pij.pi * df_pij.pj * (df_pij.i - df_pij.j) ** 2.0) / (
                n_p * (n_p - 1.0)) * np.sum(df_pi.s) / n_v
        else:
            df_feat["ngt_contrast"] = 0.0

        # Busyness
        if n_p > 1.0 and np.sum(np.abs(df_pij.i * df_pij.pi - df_pij.j * df_pij.pj)) > 0.0:
            df_feat["ngt_busyness"] = np.sum(df_pi.pi * df_pi.s) / (
                np.sum(np.abs(df_pij.i * df_pij.pi - df_pij.j * df_pij.pj)))
        else:
            df_feat["ngt_busyness"] = 0.0

        # Complexity
        df_feat["ngt_complexity"] = np.sum(
            np.abs(df_pij.i - df_pij.j) * (df_pij.pi * df_pij.si + df_pij.pj * df_pij.sj) / (
                df_pij.pi + df_pij.pj)) / n_v

        # Strength
        if np.sum(df_pi.s) > 0.0:
            df_feat["ngt_strength"] = np.sum((df_pij.pi + df_pij.pj) * (df_pij.i - df_pij.j) ** 2.0) / np.sum(
                df_pi.s)
        else:
            df_feat["ngt_strength"] = 0.0

        # Update names
        df_feat.columns += self.parse_feature_names()

        return df_feat

    def parse_feature_names(self):
        """"Used for parsing names to feature names"""
        parse_str = [""]

        # Add spatial method
        if self.spatial_method is not None:
            parse_str += [self.spatial_method]

        return "_".join(parse_str).rstrip("_")
