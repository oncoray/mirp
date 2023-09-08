import copy

import numpy as np
import pandas as pd

from mirp.featureSets.utilities import is_list_all_none
from mirp.imageClass import ImageClass
from mirp.roiClass import RoiClass
from mirp.images.genericImage import GenericImage
from mirp.masks.baseMask import BaseMask
from mirp.settings.settingsFeatureExtraction import FeatureExtractionSettingsClass
from mirp.utilities import real_ndim


def get_szm_features(
        image: GenericImage,
        mask: BaseMask,
        settings: FeatureExtractionSettingsClass
) -> pd.DataFrame:
    """Extract size zone matrix-based features from the intensity roi"""

    # Generate an empty feature list
    feat_list = []

    if image.is_empty() or mask.roi_intensity is None:
        # In case the input image or ROI are missing.
        n_slices = 1
    else:
        # Default case with input image and ROI available
        n_slices = image.image_dimension[0]

    # Iterate over spatial arrangements
    for ii_spatial in settings.glszm_spatial_method:

        # Initiate list of size zone matrix objects
        szm_list = []

        # Perform 2D analysis
        if ii_spatial.lower() in ["2d", "2.5d"]:

            # Iterate over slices
            for ii_slice in np.arange(0, n_slices):

                # Perform analysis per slice
                szm_list += [SizeZoneMatrix(
                    spatial_method=ii_spatial.lower(),
                    slice_id=ii_slice)]

        # Perform 3D analysis
        if ii_spatial.lower() == "3d":
            # Perform analysis on the entire volume
            szm_list += [SizeZoneMatrix(
                spatial_method=ii_spatial.lower(),
                slice_id=None)]

        # Calculate size zone matrices
        for szm in szm_list:
            szm.calculate_matrix(
                image=image,
                mask=mask)

        # Merge matrices according to the given method
        upd_list = combine_matrices(
            szm_list=szm_list,
            spatial_method=ii_spatial.lower())

        # Calculate features
        feat_run_list = []
        for szm in upd_list:
            feat_run_list += [szm.compute_features()]

        # Average feature values
        feat_list += [pd.concat(feat_run_list, axis=0).mean(axis=0, skipna=True).to_frame().transpose()]

    # Merge feature tables into a single table
    feature_data = pd.concat(feat_list, axis=1)

    return feature_data


def combine_matrices(szm_list, spatial_method):
    """Function to combine szm matrices prior to feature calculation."""

    # Make copy of szm_list
    use_list = []

    if spatial_method == "2d":
        # Average features over slices: maintain original size zone matrices
        for szm in szm_list:
            use_list += [szm.copy()]

    elif spatial_method in ["2.5d", "3d"]:
        # Merge all szms into a single representation

        # Select all matrices within the slice
        sel_matrix_list = []
        for szm_id in np.arange(len(szm_list)):
            sel_matrix_list += [szm_list[szm_id].matrix]

        # Check if any matrix has been created
        if is_list_all_none(sel_matrix_list):
            # No matrix was created
            use_list += [SizeZoneMatrix(spatial_method=spatial_method,
                                        slice_id=None,
                                        matrix=None,
                                        n_v=0.0)]
        else:
            # Merge size zone matrices
            merge_szm = pd.concat(sel_matrix_list, axis=0)
            merge_szm = merge_szm.groupby(by=["i", "s"]).sum().reset_index()

            # Update the number of voxels
            merge_n_v = 0.0
            for szm_id in np.arange(len(szm_list)):
                merge_n_v += szm_list[szm_id].n_v

            # Create new size zone matrix
            use_list += [SizeZoneMatrix(spatial_method=spatial_method,
                                        slice_id=None,
                                        matrix=merge_szm,
                                        n_v=merge_n_v)]

    else:
        use_list = None

    return use_list


class SizeZoneMatrix:

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

    def calculate_matrix(
            self,
            image: GenericImage,
            mask: BaseMask):

        # Check if the input image and roi exist
        if image.is_empty() or mask.roi_intensity is None:
            self.set_empty()
            return

        # Check if the roi contains any masked voxels. If this is not the case, don't construct the GLSZM.
        if not np.any(mask.roi_intensity.get_voxel_grid()):
            self.set_empty()
            return

        from skimage.measure import label

        # Define neighbour directions
        if self.spatial_method == "3d":
            connectivity = 3
            img_vol = copy.deepcopy(image.get_voxel_grid())
            roi_vol = copy.deepcopy(mask.roi_intensity.get_voxel_grid())

        elif self.spatial_method in ["2d", "2.5d"]:
            connectivity = 2
            img_vol = image.get_voxel_grid()[self.slice, :, :]
            roi_vol = mask.roi_intensity.get_voxel_grid()[self.slice, :, :]
        else:
            raise ValueError(
                "The spatial method for grey level size zone matrices should be one of \"2d\", \"2.5d\" or \"3d\".")

        # Check dimensionality and update connectivity if necessary.
        connectivity = min([connectivity, real_ndim(img_vol)])

        # Set voxels outside roi to 0.0
        img_vol[~roi_vol] = 0.0

        # Count the number of voxels within the roi
        self.n_v = np.sum(roi_vol)

        # Label all connected voxels with the same label.
        img_label = label(img_vol, background=0, connectivity=connectivity)

        # Generate data frame
        df_szm = pd.DataFrame({
            "g": np.ravel(img_vol),
            "vol_id": np.ravel(img_label),
            "in_roi": np.ravel(roi_vol)
        })

        # Remove all non-roi entries and count occurrence of combinations of volume id and grey level
        df_szm = df_szm[df_szm.in_roi].groupby(by=["g", "vol_id"]).size().reset_index(name="zone_size")

        # Count the number of co-occurring sizes and grey values
        df_szm = df_szm.groupby(by=["g", "zone_size"]).size().reset_index(name="n")

        # Rename columns
        df_szm.columns = ["i", "s", "n"]

        # Add matrix to object
        self.matrix = df_szm

    def calculate_matrix_deprecated(self, img_obj, roi_obj):

        # Check if the input image and roi exist
        if img_obj.is_missing or roi_obj.roi_intensity is None:
            self.set_empty()
            return

        # Check if the roi contains any masked voxels. If this is not the case, don't construct the GLSZM.
        if not np.any(roi_obj.roi_intensity.get_voxel_grid()):
            self.set_empty()
            return

        from skimage.measure import label

        # Define neighbour directions
        if self.spatial_method == "3d":
            connectivity = 3
            img_vol = copy.deepcopy(img_obj.get_voxel_grid())
            roi_vol = copy.deepcopy(roi_obj.roi_intensity.get_voxel_grid())
        elif self.spatial_method in ["2d", "2.5d"]:
            connectivity = 2
            img_vol = img_obj.get_voxel_grid()[self.slice, :, :]
            roi_vol = roi_obj.roi_intensity.get_voxel_grid()[self.slice, :, :]
        else:
            raise ValueError("The spatial method for grey level size zone matrices should be one of \"2d\", \"2.5d\" or \"3d\".")

        # Check dimensionality and update connectivity if necessary.
        connectivity = min([connectivity, real_ndim(img_vol)])

        # Set voxels outside roi to 0.0
        img_vol[~roi_vol] = 0.0

        # Count the number of voxels within the roi
        self.n_v = np.sum(roi_vol)

        # Label all connected voxels with the same label.
        img_label = label(img_vol, background=0, connectivity=connectivity)

        # Generate data frame
        df_szm = pd.DataFrame({"g":      np.ravel(img_vol),
                               "vol_id": np.ravel(img_label),
                               "in_roi": np.ravel(roi_vol)})

        # Remove all non-roi entries and count occurrence of combinations of volume id and grey level
        df_szm = df_szm[df_szm.in_roi].groupby(by=["g", "vol_id"]).size().reset_index(name="zone_size")

        # Count the number of co-occurring sizes and grey values
        df_szm = df_szm.groupby(by=["g", "zone_size"]).size().reset_index(name="n")

        # Rename columns
        df_szm.columns = ["i", "s", "n"]

        # Add matrix to object
        self.matrix = df_szm

    def compute_features(self):

        # Create feature table
        feat_names = [
            "szm_sze", "szm_lze", "szm_lgze", "szm_hgze", "szm_szlge", "szm_szhge", "szm_lzlge", "szm_lzhge",
            "szm_glnu", "szm_glnu_norm", "szm_zsnu", "szm_zsnu_norm", "szm_z_perc",
            "szm_gl_var", "szm_zs_var", "szm_zs_entr"
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

        # Make a local copy of the size zone matrix for processing
        df_sij = copy.deepcopy(self.matrix)
        df_sij.columns = ("i", "j", "sij")

        # Sum over grey levels
        df_si = df_sij.groupby(by="i")["sij"].agg(np.sum).reset_index().rename(columns={"sij": "si"})

        # Sum over zone sizes
        df_sj = df_sij.groupby(by="j")["sij"].agg(np.sum).reset_index().rename(columns={"sij": "sj"})

        # Constant definitions
        n_s = np.sum(df_sij.sij) * 1.0  # Number of size zones
        n_v = self.n_v * 1.0            # Number of voxels

        ###############################################
        # GLSZM features
        ###############################################

        # Small zone emphasis
        df_feat["szm_sze"] = np.sum(df_sj.sj / df_sj.j ** 2.0) / n_s

        # Large zone emphasis
        df_feat["szm_lze"] = np.sum(df_sj.sj * df_sj.j ** 2.0) / n_s

        # Grey level non-uniformity
        df_feat["szm_glnu"] = np.sum(df_si.si ** 2.0) / n_s

        # Grey level non-uniformity, normalised
        df_feat["szm_glnu_norm"] = np.sum(df_si.si ** 2.0) / n_s ** 2.0

        # Zone size non-uniformity
        df_feat["szm_zsnu"] = np.sum(df_sj.sj ** 2.0) / n_s

        # Zone size non-uniformity
        df_feat["szm_zsnu_norm"] = np.sum(df_sj.sj ** 2.0) / n_s ** 2.0

        # Zone percentage
        df_feat["szm_z_perc"] = n_s / n_v

        # Low grey level emphasis
        df_feat["szm_lgze"] = np.sum(df_si.si / df_si.i ** 2.0) / n_s

        # High grey level emphasis
        df_feat["szm_hgze"] = np.sum(df_si.si * df_si.i ** 2.0) / n_s

        # Small zone low grey level emphasis
        df_feat["szm_szlge"] = np.sum(df_sij.sij / (df_sij.i * df_sij.j) ** 2.0) / n_s

        # Small zone high grey level emphasis
        df_feat["szm_szhge"] = np.sum(df_sij.sij * df_sij.i ** 2.0 / df_sij.j ** 2.0) / n_s

        # Large zone low grey level emphasis
        df_feat["szm_lzlge"] = np.sum(df_sij.sij * df_sij.j ** 2.0 / df_sij.i ** 2.0) / n_s

        # Large zone high grey level emphasis
        df_feat["szm_lzhge"] = np.sum(df_sij.sij * df_sij.i ** 2.0 * df_sij.j ** 2.0) / n_s

        # Grey level variance
        mu = np.sum(df_sij.sij * df_sij.i) / n_s
        df_feat["szm_gl_var"] = np.sum((df_sij.i - mu) ** 2.0 * df_sij.sij) / n_s
        del mu

        # Zone size variance
        mu = np.sum(df_sij.sij * df_sij.j) / n_s
        df_feat["szm_zs_var"] = np.sum((df_sij.j - mu) ** 2.0 * df_sij.sij) / n_s
        del mu

        # Zone size entropy
        df_feat["szm_zs_entr"] = - np.sum(df_sij.sij * np.log2(df_sij.sij / n_s)) / n_s

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


def get_szm_features_deprecated(img_obj: ImageClass,
                                roi_obj: RoiClass,
                                settings: FeatureExtractionSettingsClass):
    """Extract size zone matrix-based features from the intensity roi"""

    # Generate an empty feature list
    feat_list = []

    if img_obj.is_missing or roi_obj.roi_intensity is None:
        # In case the input image or ROI are missing.
        n_slices = 1
    else:
        # Default case with input image and ROI available
        n_slices = img_obj.size[0]

    # Iterate over spatial arrangements
    for ii_spatial in settings.glszm_spatial_method:

        # Initiate list of size zone matrix objects
        szm_list = []

        # Perform 2D analysis
        if ii_spatial.lower() in ["2d", "2.5d"]:

            # Iterate over slices
            for ii_slice in np.arange(0, n_slices):

                # Perform analysis per slice
                szm_list += [SizeZoneMatrix(spatial_method=ii_spatial.lower(),
                                            slice_id=ii_slice)]

        # Perform 3D analysis
        if ii_spatial.lower() == "3d":
            # Perform analysis on the entire volume
            szm_list += [SizeZoneMatrix(spatial_method=ii_spatial.lower(),
                                        slice_id=None)]

        # Calculate size zone matrices
        for szm in szm_list:
            szm.calculate_matrix_deprecated(img_obj=img_obj,
                                            roi_obj=roi_obj)

        # Merge matrices according to the given method
        upd_list = combine_matrices(szm_list=szm_list,
                                    spatial_method=ii_spatial.lower())

        # Calculate features
        feat_run_list = []
        for szm in upd_list:
            feat_run_list += [szm.compute_features()]

        # Average feature values
        feat_list += [pd.concat(feat_run_list, axis=0).mean(axis=0, skipna=True).to_frame().transpose()]

    # Merge feature tables into a single table
    df_feat = pd.concat(feat_list, axis=1)

    return df_feat
