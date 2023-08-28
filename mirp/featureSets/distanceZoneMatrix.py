import numpy as np
import pandas as pd

import copy

from mirp.featureSets.utilities import is_list_all_none
from mirp.imageClass import ImageClass
from mirp.roiClass import RoiClass
from mirp.images.genericImage import GenericImage
from mirp.masks.baseMask import BaseMask
from mirp.importSettings import FeatureExtractionSettingsClass
from mirp.utilities import real_ndim


def get_dzm_features(
        image: GenericImage,
        mask: BaseMask,
        settings: FeatureExtractionSettingsClass):
    """Extract size zone matrix-based features from the intensity roi"""

    # Generate an empty feature list
    feat_list = []

    if image.is_empty() or mask.roi_intensity is None or mask.roi_morphology is None:
        # In case the input image or ROI masks are missing.
        n_slices = 1
    else:
        # Default case with input image and ROI available
        n_slices = image.image_dimension[0]

    # Iterate over spatial arrangements
    for ii_spatial in settings.gldzm_spatial_method:

        # Initiate list of distance zone matrix objects
        dzm_list = []

        # Perform 2D analysis
        if ii_spatial.lower() in ["2d", "2.5d"]:

            # Convert image to table
            df_img = mask.as_pandas_dataframe(
                image=image,
                intensity_mask=True,
                distance_map=True,
                by_slice=True)

            # Iterate over slices
            for ii_slice in np.arange(0, n_slices):

                # Perform analysis per slice
                dzm_list += [DistanceZoneMatrix(
                    spatial_method=ii_spatial.lower(),
                    slice_id=ii_slice)]

        # Perform 3D analysis
        elif ii_spatial.lower() == "3d":

            # Convert image to table
            df_img = mask.as_pandas_dataframe(
                image=image,
                intensity_mask=True,
                distance_map=True)

            # Perform analysis on the entire volume
            dzm_list += [DistanceZoneMatrix(
                spatial_method=ii_spatial.lower(),
                slice_id=None)]
        else:
            raise ValueError("Spatial methods for DZM should be \"2d\", \"2.5d\" or \"3d\".")

        # Calculate size zone matrices
        for dzm in dzm_list:
            dzm.calculate_matrix(
                image=image,
                mask=mask,
                df_img=df_img)

        # Merge matrices according to the given method
        upd_list = combine_matrices(
            dzm_list=dzm_list,
            spatial_method=ii_spatial.lower())

        # Calculate features
        feat_run_list = []
        for dzm in upd_list:
            feat_run_list += [dzm.compute_features()]

        # Average feature values
        feat_list += [pd.concat(feat_run_list, axis=0).mean(axis=0, skipna=True).to_frame().transpose()]

    # Merge feature tables into a single table
    feature_data = pd.concat(feat_list, axis=1)

    return feature_data


def combine_matrices(dzm_list, spatial_method):
    """Function to combine distance zone matrices prior to feature calculation."""

    # Initiate empty list
    use_list = []

    if spatial_method == "2d":
        # Average features over slice, maintain original distance zone matrices

        # Make copy of dzm_list
        for dzm in dzm_list:
            use_list += [dzm.copy()]

    elif spatial_method in ["2.5d", "3d"]:
        # Merge all distance zone matrices into a single representation

        # Select all matrices within the slice
        sel_matrix_list = []
        for dzm_id in np.arange(len(dzm_list)):
            sel_matrix_list += [dzm_list[dzm_id].matrix]

        # Check if any matrix has been created
        if is_list_all_none(sel_matrix_list):
            # No matrix was created
            use_list += [DistanceZoneMatrix(spatial_method=spatial_method, slice_id=None, matrix=None, n_v=0.0)]
        else:
            # Merge distance zone matrices
            merge_dzm = pd.concat(sel_matrix_list, axis=0)
            merge_dzm = merge_dzm.groupby(by=["i", "d"]).sum().reset_index()

            # Update the number of voxels
            merge_n_v = 0.0
            for dzm_id in np.arange(len(dzm_list)):
                merge_n_v += dzm_list[dzm_id].n_v

            # Create new distance zone matrix
            use_list += [DistanceZoneMatrix(spatial_method=spatial_method, slice_id=None, matrix=merge_dzm,
                                            n_v=merge_n_v)]
    else:
        use_list = None

    # Return to new dzm list to calling function
    return use_list


class DistanceZoneMatrix:

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
            mask: BaseMask,
            df_img: pd.DataFrame):

        # Check if the input image and roi exist
        if image.is_empty() or mask.roi_intensity is None or df_img is None:
            self.set_empty()
            return

        # Check if the roi contains any masked voxels. If this is not the case, don't construct the GLDZM.
        if not np.any(mask.roi_intensity.get_voxel_grid()):
            self.set_empty()
            return

        from skimage.measure import label

        # Define neighbour directions
        if self.spatial_method == "3d":
            connectivity = 3
            img_vol = copy.deepcopy(image.get_voxel_grid())
            roi_vol = copy.deepcopy(mask.roi_intensity.get_voxel_grid())
            df_dzm = copy.deepcopy(df_img)

        elif self.spatial_method in ["2d", "2.5d"]:
            connectivity = 2
            img_vol = image.get_voxel_grid()[self.slice, :, :]
            roi_vol = mask.roi_intensity.get_voxel_grid()[self.slice, :, :]
            df_dzm = copy.deepcopy(df_img[df_img.z == self.slice])

        else:
            raise ValueError(
                "The spatial method for grey level distance zone matrices should be one of \"2d\", \"2.5d\" or \"3d\".")

        # Check dimensionality and update connectivity if necessary.
        connectivity = min([connectivity, real_ndim(img_vol)])

        # Set voxels outside the roi to 0.0
        img_vol[~roi_vol] = 0.0

        # Count the number of voxels within the roi
        self.n_v = np.sum(roi_vol)

        # Label all connected voxels with the same grey level
        img_label = label(img_vol, background=0, connectivity=connectivity)

        # Add group labels
        df_dzm["vol_id"] = np.ravel(img_label)

        # Select minimum group distance for unique groups
        df_dzm = df_dzm[df_dzm.roi_int_mask].groupby(by=["g", "vol_id"])["border_distance"].agg(np.min).reset_index().rename(columns={"border_distance": "d"})

        # Count occurrence of grey level and distance
        df_dzm = df_dzm.groupby(by=["g", "d"]).size().reset_index(name="n")

        # Rename columns
        df_dzm.columns = ["i", "d", "n"]

        # Add matrix to object
        self.matrix = df_dzm

    def calculate_matrix_deprecated(self, img_obj, roi_obj, df_img):

        # Check if the input image and roi exist
        if img_obj.is_missing or roi_obj.roi_intensity is None or df_img is None:
            self.set_empty()
            return

        # Check if the roi contains any masked voxels. If this is not the case, don't construct the GLDZM.
        if not np.any(roi_obj.roi_intensity.get_voxel_grid()):
            self.set_empty()
            return

        from skimage.measure import label

        # Define neighbour directions
        if self.spatial_method == "3d":
            connectivity = 3
            img_vol = copy.deepcopy(img_obj.get_voxel_grid())
            roi_vol = copy.deepcopy(roi_obj.roi_intensity.get_voxel_grid())
            df_dzm = copy.deepcopy(df_img)

        elif self.spatial_method in ["2d", "2.5d"]:
            connectivity = 2
            img_vol = img_obj.get_voxel_grid()[self.slice, :, :]
            roi_vol = roi_obj.roi_intensity.get_voxel_grid()[self.slice, :, :]
            df_dzm = copy.deepcopy(df_img[df_img.z == self.slice])

        else:
            raise ValueError("The spatial method for grey level distance zone matrices should be one of \"2d\", \"2.5d\" or \"3d\".")

        # Check dimensionality and update connectivity if necessary.
        connectivity = min([connectivity, real_ndim(img_vol)])

        # Set voxels outside the roi to 0.0
        img_vol[~roi_vol] = 0.0

        # Count the number of voxels within the roi
        self.n_v = np.sum(roi_vol)

        # Label all connected voxels with the same grey level
        img_label = label(img_vol, background=0, connectivity=connectivity)

        # Add group labels
        df_dzm["vol_id"] = np.ravel(img_label)

        # Select minimum group distance for unique groups
        df_dzm = df_dzm[df_dzm.roi_int_mask].groupby(by=["g", "vol_id"])["border_distance"].agg(np.min).reset_index().rename(columns={"border_distance": "d"})

        # Count occurrence of grey level and distance
        df_dzm = df_dzm.groupby(by=["g", "d"]).size().reset_index(name="n")

        # Rename columns
        df_dzm.columns = ["i", "d", "n"]

        # Add matrix to object
        self.matrix = df_dzm

    def compute_features(self):

        # Create feature table
        feat_names = ["dzm_sde", "dzm_lde", "dzm_lgze", "dzm_hgze", "dzm_sdlge", "dzm_sdhge", "dzm_ldlge", "dzm_ldhge",
                      "dzm_glnu", "dzm_glnu_norm", "dzm_zdnu", "dzm_zdnu_norm", "dzm_z_perc",
                      "dzm_gl_var", "dzm_zd_var", "dzm_zd_entr"]
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

        # Create a copy of the distance zone matrix and rename columns
        df_dij = copy.deepcopy(self.matrix)
        df_dij.columns = ("i", "j", "dij")

        # Sum over grey levels
        df_di = df_dij.groupby(by="i")["dij"].agg(np.sum).reset_index().rename(columns={"dij": "di"})

        # Sum over zone distances
        df_dj = df_dij.groupby(by="j")["dij"].agg(np.sum).reset_index().rename(columns={"dij": "dj"})

        # Constant definitions
        n_s = np.sum(df_dij.dij) * 1.0  # Number of size zones
        n_v = self.n_v * 1.0            # Number of voxels

        ###############################################
        # GLDZM features
        ###############################################

        # Small distance emphasis
        df_feat["dzm_sde"] = np.sum(df_dj.dj / df_dj.j ** 2.0) / n_s

        # Large distance emphasis
        df_feat["dzm_lde"] = np.sum(df_dj.dj * df_dj.j ** 2.0) / n_s

        # Grey level non-uniformity
        df_feat["dzm_glnu"] = np.sum(df_di.di ** 2.0) / n_s

        # Grey level non-uniformity, normalised
        df_feat["dzm_glnu_norm"] = np.sum(df_di.di ** 2.0) / n_s ** 2.0

        # Zone distance non-uniformity
        df_feat["dzm_zdnu"] = np.sum(df_dj.dj ** 2.0) / n_s

        # Zone distance non-uniformity
        df_feat["dzm_zdnu_norm"] = np.sum(df_dj.dj ** 2.0) / n_s ** 2.0

        # Zone percentage
        df_feat["dzm_z_perc"] = n_s / n_v

        # Low grey level emphasis
        df_feat["dzm_lgze"] = np.sum(df_di.di / df_di.i ** 2.0) / n_s

        # High grey level emphasis
        df_feat["dzm_hgze"] = np.sum(df_di.di * df_di.i ** 2.0) / n_s

        # Small distance low grey level emphasis
        df_feat["dzm_sdlge"] = np.sum(df_dij.dij / (df_dij.i * df_dij.j) ** 2.0) / n_s

        # Small distance high grey level emphasis
        df_feat["dzm_sdhge"] = np.sum(df_dij.dij * df_dij.i ** 2.0 / df_dij.j ** 2.0) / n_s

        # Large distance low grey level emphasis
        df_feat["dzm_ldlge"] = np.sum(df_dij.dij * df_dij.j ** 2.0 / df_dij.i ** 2.0) / n_s

        # Large distance high grey level emphasis
        df_feat["dzm_ldhge"] = np.sum(df_dij.dij * df_dij.i ** 2.0 * df_dij.j ** 2.0) / n_s

        # Grey level variance
        mu = np.sum(df_dij.dij * df_dij.i) / n_s
        df_feat["dzm_gl_var"] = np.sum((df_dij.i - mu) ** 2.0 * df_dij.dij) / n_s
        del mu

        # Zone distance variance
        mu = np.sum(df_dij.dij * df_dij.j) / n_s
        df_feat["dzm_zd_var"] = np.sum((df_dij.j - mu) ** 2.0 * df_dij.dij) / n_s
        del mu

        # Zone distance entropy
        df_feat["dzm_zd_entr"] = - np.sum(df_dij.dij * np.log2(df_dij.dij / n_s)) / n_s

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


def get_dzm_features_deprecated(img_obj: ImageClass,
                                roi_obj: RoiClass,
                                settings: FeatureExtractionSettingsClass):
    """Extract size zone matrix-based features from the intensity roi"""

    # Generate an empty feature list
    feat_list = []

    if img_obj.is_missing or roi_obj.roi_intensity is None or roi_obj.roi_morphology is None:
        # In case the input image or ROI masks are missing.
        n_slices = 1
    else:
        # Default case with input image and ROI available
        n_slices = img_obj.size[0]

    # Iterate over spatial arrangements
    for ii_spatial in settings.gldzm_spatial_method:

        # Initiate list of distance zone matrix objects
        dzm_list = []

        # Perform 2D analysis
        if ii_spatial.lower() in ["2d", "2.5d"]:

            # Convert image to table
            df_img = roi_obj.as_pandas_dataframe(img_obj=img_obj,
                                                 intensity_mask=True,
                                                 distance_map=True,
                                                 by_slice=True)

            # Iterate over slices
            for ii_slice in np.arange(0, n_slices):

                # Perform analysis per slice
                dzm_list += [DistanceZoneMatrix(spatial_method=ii_spatial.lower(),
                                                slice_id=ii_slice)]

        # Perform 3D analysis
        elif ii_spatial.lower() == "3d":

            # Convert image to table
            df_img = roi_obj.as_pandas_dataframe(img_obj=img_obj,
                                                 intensity_mask=True,
                                                 distance_map=True)

            # Perform analysis on the entire volume
            dzm_list += [DistanceZoneMatrix(spatial_method=ii_spatial.lower(),
                                            slice_id=None)]
        else:
            raise ValueError("Spatial methods for DZM should be \"2d\", \"2.5d\" or \"3d\".")

        # Calculate size zone matrices
        for dzm in dzm_list:
            dzm.calculate_matrix_deprecated(img_obj=img_obj,
                                            roi_obj=roi_obj,
                                            df_img=df_img)

        # Merge matrices according to the given method
        upd_list = combine_matrices(dzm_list=dzm_list,
                                    spatial_method=ii_spatial.lower())

        # Calculate features
        feat_run_list = []
        for dzm in upd_list:
            feat_run_list += [dzm.compute_features()]

        # Average feature values
        feat_list += [pd.concat(feat_run_list, axis=0).mean(axis=0, skipna=True).to_frame().transpose()]

    # Merge feature tables into a single table
    df_feat = pd.concat(feat_list, axis=1)

    return df_feat
