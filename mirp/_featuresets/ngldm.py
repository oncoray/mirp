import copy

import numpy as np
import pandas as pd

from mirp._featuresets.utilities import is_list_all_none, get_neighbour_directions, coord2Index, get_intensity_value
from mirp._images.genericImage import GenericImage
from mirp._masks.baseMask import BaseMask
from mirp.settings.settingsFeatureExtraction import FeatureExtractionSettingsClass


def get_ngldm_features(
        image: GenericImage,
        mask: BaseMask,
        settings: FeatureExtractionSettingsClass
) -> pd.DataFrame:
    """Extract neighbouring grey level difference matrix-based features from the intensity roi"""

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
    for ii_spatial in settings.ngldm_spatial_method:

        # Iterate over difference levels
        for ii_diff_lvl in settings.ngldm_diff_lvl:

            # Iterate over distances
            for ii_dist in settings.ngldm_dist:

                # Initiate list of ngldm objects
                ngldm_list = []

                if ii_spatial.lower() in ["2d", "2.5d"]:
                    # Perform 2D analysis
                    for ii_slice in np.arange(0, n_slices):

                        # Add ngldm matrices to list
                        ngldm_list += [GreyLevelDependenceMatrix(
                            distance=int(ii_dist),
                            diff_lvl=ii_diff_lvl,
                            spatial_method=ii_spatial.lower(),
                            slice_id=ii_slice)]

                elif ii_spatial.lower() == "3d":
                    # Perform 3D analysis
                    ngldm_list += [GreyLevelDependenceMatrix(
                        distance=int(ii_dist),
                        diff_lvl=ii_diff_lvl,
                        spatial_method=ii_spatial.lower(),
                        slice_id=None)]

                else:
                    raise ValueError("Spatial methods for NGLDM should be \"2d\", \"2.5d\" or \"3d\".")

                # Calculate ngldm matrices
                for ngldm in ngldm_list:
                    ngldm.calculate_matrix(
                        df_img=df_img,
                        img_dims=img_dims)

                # Merge matrices according to the given method
                upd_list = combine_matrices(
                    ngldm_list=ngldm_list,
                    spatial_method=ii_spatial.lower())

                # Calculate features
                feat_run_list = []
                for ngldm in upd_list:
                    feat_run_list += [ngldm.compute_features(g_range=np.array(mask.intensity_range))]

                # Average feature values
                feat_list += [pd.concat(feat_run_list, axis=0).mean(axis=0, skipna=True).to_frame().transpose()]

    # Merge feature tables into a single table
    feature_data = pd.concat(feat_list, axis=1)

    return feature_data


def combine_matrices(ngldm_list, spatial_method):
    """Function to combine ngldm matrices prior to feature calculation."""

    # Initiate empty list
    use_list = []

    if spatial_method == "2d":
        # Average features over slice: maintain original ngldms

        # Make copy of ngldm_list
        use_list = []
        for ngldm in ngldm_list:
            use_list += [ngldm.copy()]

    elif spatial_method in ["2.5d", "3d"]:
        # Merge all ngldms into a single representation

        # Select all matrices within the slice
        sel_matrix_list = []
        for ngldm_id in np.arange(len(ngldm_list)):
            sel_matrix_list += [ngldm_list[ngldm_id].matrix]

        # Check if any matrix has been created
        if is_list_all_none(sel_matrix_list):
            # No matrix was created
            use_list += [GreyLevelDependenceMatrix(
                distance=ngldm_list[0].distance,
                diff_lvl=ngldm_list[0].diff_lvl,
                spatial_method=spatial_method,
                slice_id=None,
                matrix=None,
                n_v=0.0)]

        else:
            # Merge neighbouring grey level difference matrices
            merge_ngldm = pd.concat(sel_matrix_list, axis=0)
            merge_ngldm = merge_ngldm.groupby(by=["i", "j"]).sum().reset_index()

            # Update the number of voxels
            merge_n_v = 0.0
            for ngldm_id in np.arange(len(ngldm_list)):
                merge_n_v += ngldm_list[ngldm_id].n_v

            # Create new neighbouring grey level difference matrix
            use_list += [GreyLevelDependenceMatrix(
                distance=ngldm_list[0].distance,
                diff_lvl=ngldm_list[0].diff_lvl,
                spatial_method=spatial_method,
                slice_id=None,
                matrix=merge_ngldm,
                n_v=merge_n_v)]

    else:
        use_list = None

    # Return to new ngldm list to calling function
    return use_list


class GreyLevelDependenceMatrix:

    def __init__(
            self,
            distance,
            diff_lvl,
            spatial_method,
            slice_id=None,
            matrix=None,
            n_v=None):

        # Distance used
        self.distance = distance
        self.diff_lvl = diff_lvl

        # Slice for which the current matrix is extracted
        self.slice = slice_id

        # Spatial analysis method (2d, 2.5d, 3d)
        self.spatial_method = spatial_method

        # Place holders
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

        # Check if the roi contains any masked voxels. If this is not the case, don't construct the NGLDM.
        if not np.any(df_img.roi_int_mask):
            self.set_empty()
            return

        if self.spatial_method == "3d":
            # Set up neighbour vectors
            nbrs = get_neighbour_directions(
                d=self.distance,
                distance="chebyshev",
                centre=False,
                complete=True,
                dim3=True)

            # Set up work copy
            df_ngldm = copy.deepcopy(df_img)
        elif self.spatial_method in ["2d", "2.5d"]:
            # Set up neighbour vectors
            nbrs = get_neighbour_directions(
                d=self.distance,
                distance="chebyshev",
                centre=False,
                complete=True,
                dim3=False)

            # Set up work copy
            df_ngldm = copy.deepcopy(df_img[df_img.z == self.slice])
            df_ngldm["index_id"] = np.arange(0, len(df_ngldm))
            df_ngldm["z"] = 0
            df_ngldm = df_ngldm.reset_index(drop=True)
        else:
            raise ValueError("The spatial method for neighbouring grey level dependence matrices should be one of \"2d\", \"2.5d\" or \"3d\".")

        # Set grey level of voxels outside ROI to NaN
        df_ngldm.loc[df_ngldm.roi_int_mask == False, "g"] = np.nan

        # Update number of voxels for current iteration
        self.n_v = np.sum(df_ngldm.roi_int_mask.values)

        # Initialise sum of grey levels and number of neighbours
        df_ngldm["occur"] = 0.0
        df_ngldm["n_nbrs"] = 0.0

        for k in range(0, np.shape(nbrs)[1]):
            # Determine potential transitions from valid voxels
            df_ngldm["to_index"] = coord2Index(
                x=df_ngldm.x.values + nbrs[2, k],
                y=df_ngldm.y.values + nbrs[1, k],
                z=df_ngldm.z.values + nbrs[0, k],
                dims=img_dims)

            # Get grey level value from transitions
            df_ngldm["to_g"] = get_intensity_value(x=df_ngldm.g.values, index=df_ngldm.to_index.values)

            # Determine which voxels have valid neighbours
            sel_index = np.isfinite(df_ngldm.to_g)

            # Determine co-occurrence within diff_lvl
            df_ngldm.loc[sel_index, "occur"] += ((np.abs(df_ngldm.to_g - df_ngldm.g)[sel_index]) <= self.diff_lvl) * 1

        # Work with voxels within the intensity roi
        df_ngldm = df_ngldm[df_ngldm.roi_int_mask]

        # Drop superfluous columns
        df_ngldm = df_ngldm.drop(labels=["index_id", "x", "y", "z", "to_index", "to_g", "roi_int_mask"], axis=1)

        # Sum s over voxels
        df_ngldm = df_ngldm.groupby(by=["g", "occur"]).size().reset_index(name="n")

        # Rename columns
        df_ngldm.columns = ["i", "j", "s"]

        # Add one to dependency count as features are not defined for k=0
        df_ngldm.j += 1.0

        # Add matrix to object
        self.matrix = df_ngldm

    def compute_features(self, g_range):

        # Create feature table
        feat_names = [
            "ngl_lde", "ngl_hde", "ngl_lgce", "ngl_hgce", "ngl_ldlge", "ngl_ldhge", "ngl_hdlge", "ngl_hdhge",
            "ngl_glnu", "ngl_glnu_norm", "ngl_dcnu", "ngl_dcnu_norm", "ngl_dc_perc",
            "ngl_gl_var", "ngl_dc_var", "ngl_dc_entr", "ngl_dc_energy"
        ]
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

        # Dependence count dataframe
        df_sij = copy.deepcopy(self.matrix)
        df_sij.columns = ("i", "j", "sij")

        # Sum over grey levels
        df_si = df_sij.groupby(by="i")["sij"].agg(np.sum).reset_index().rename(columns={"sij": "si"})

        # Sum over dependence counts
        df_sj = df_sij.groupby(by="j")["sij"].agg(np.sum).reset_index().rename(columns={"sij": "sj"})

        # Constant definitions
        n_s = np.sum(df_sij.sij) * 1.0  # Number of neighbourhoods considered
        n_v = self.n_v  # Number of voxels

        ###############################################
        # NGLDM features
        ###############################################

        # Low dependence emphasis
        df_feat["ngl_lde"] = np.sum(df_sj.sj / df_sj.j ** 2.0) / n_s

        # High dependence emphasis
        df_feat["ngl_hde"] = np.sum(df_sj.sj * df_sj.j ** 2.0) / n_s

        # Grey level non-uniformity
        df_feat["ngl_glnu"] = np.sum(df_si.si ** 2.0) / n_s

        # Grey level non-uniformity, normalised
        df_feat["ngl_glnu_norm"] = np.sum(df_si.si ** 2.0) / n_s ** 2.0

        # Dependence count non-uniformity
        df_feat["ngl_dcnu"] = np.sum(df_sj.sj ** 2.0) / n_s

        # Dependence count non-uniformity, normalised
        df_feat["ngl_dcnu_norm"] = np.sum(df_sj.sj ** 2.0) / n_s ** 2.0

        # Dependence count percentage
        df_feat["ngl_dc_perc"] = n_s / n_v

        # Low grey level count emphasis
        df_feat["ngl_lgce"] = np.sum(df_si.si / df_si.i ** 2.0) / n_s

        # High grey level count emphasis
        df_feat["ngl_hgce"] = np.sum(df_si.si * df_si.i ** 2.0) / n_s

        # Low dependence low grey level emphasis
        df_feat["ngl_ldlge"] = np.sum(df_sij.sij / (df_sij.i * df_sij.j) ** 2.0) / n_s

        # Low dependence high grey level emphasis
        df_feat["ngl_ldhge"] = np.sum(df_sij.sij * df_sij.i ** 2.0 / df_sij.j ** 2.0) / n_s

        # High dependence low grey level emphasis
        df_feat["ngl_hdlge"] = np.sum(df_sij.sij * df_sij.j ** 2.0 / df_sij.i ** 2.0) / n_s

        # High dependence high grey level emphasis
        df_feat["ngl_hdhge"] = np.sum(df_sij.sij * df_sij.i ** 2.0 * df_sij.j ** 2.0) / n_s

        # Grey level variance
        mu = np.sum(df_sij.sij * df_sij.i) / n_s
        df_feat["ngl_gl_var"] = np.sum((df_sij.i - mu) ** 2.0 * df_sij.sij) / n_s
        del mu

        # Dependence count variance
        mu = np.sum(df_sij.sij * df_sij.j) / n_s
        df_feat["ngl_dc_var"] = np.sum((df_sij.j - mu) ** 2.0 * df_sij.sij) / n_s
        del mu

        # Dependence count entropy
        df_feat["ngl_dc_entr"] = - np.sum(df_sij.sij * np.log2(df_sij.sij / n_s)) / n_s

        # Dependence count energy
        df_feat["ngl_dc_energy"] = np.sum(df_sij.sij ** 2.0) / (n_s ** 2.0)

        # Update names
        df_feat.columns += self.parse_feature_names()

        return df_feat

    def parse_feature_names(self):
        """"Used for parsing names to feature names"""
        parse_str = [""]

        # Add distance
        parse_str += ["d" + str(np.round(self.distance, 1))]

        # Add difference level
        parse_str += ["a" + str(np.round(self.diff_lvl, 0))]

        # Add spatial method
        if self.spatial_method is not None:
            parse_str += [self.spatial_method]

        return "_".join(parse_str).rstrip("_")
