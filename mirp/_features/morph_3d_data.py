import hashlib

import numpy as np
import pandas as pd
import copy

from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask


class Data3DMesh(object):

    def __init__(self):

        # Mesh faces
        self.mesh_faces: np.ndarray | None = None

        # Mesh vertices
        self.mesh_vertices: np.ndarray | None = None

        # Volume
        self.volume: float | None = None

        # Area
        self.area: float | None = None

        # Voxels within intensity mask and morphological mask
        self.data_int: pd.DataFrame | None = None
        self.data_morph: pd.DataFrame | None = None

        # Image spacing
        self.spacing: tuple[float, float, float] | None = None

    def compute(self, image: GenericImage, mask: BaseMask):
        # Skip processing if input image and/or roi are missing
        if image is None:
            raise ValueError(
                "image cannot be None, but may not have been provided in the calling function."
            )
        if mask is None:
            raise ValueError(
                "mask cannot be None, but may not have been provided in the calling function."
            )

        # Check if data actually exists
        if image.is_empty() or mask.roi_morphology.is_empty_mask():
            return

        from skimage.measure import marching_cubes

        # Image spacing
        self.spacing = image.image_spacing

        # Define tables based on morphological and intensity masks
        data = mask.as_pandas_dataframe(
            image=image,
            intensity_mask=True,
            morphology_mask=True
        )

        self.data_int = data[data.roi_int_mask].reset_index()
        self.data_morph = data[data.roi_morph_mask].reset_index()

        # Get ROI and pad with empty voxels
        morphology_mask = np.pad(
            mask.roi_morphology.get_voxel_grid(),
            pad_width=1,
            mode="constant",
            constant_values=0.0
        )

        # Use marching cubes to generate a mesh grid for the ROI
        vertices, faces, _, _ = marching_cubes(
            volume=morphology_mask,
            level=0.5,
            spacing=mask.roi_morphology.image_spacing
        )

        self.mesh_vertices = vertices
        self.mesh_faces = faces

        # Get vertices for each face
        vert_a = vertices[faces[:, 0], :]
        vert_b = vertices[faces[:, 1], :]
        vert_c = vertices[faces[:, 2], :]

        # noinspection PyUnreachableCode
        self.volume = np.abs(np.sum(
            1.0 / 6.0 * np.einsum("ij,ij->i", vert_a, np.cross(vert_b, vert_c, 1, 1))
        ))

        # noinspection PyUnreachableCode
        self.area = np.sum(np.sum(np.cross(vert_a, vert_b) ** 2.0, axis=1) ** 0.5) / 2.0

    def is_empty(self):
        return self.volume is None

    def is_singular(self):
        if self.is_empty():
            return True

        return len(self.data_morph) <= 1


class Data3DConvexHull(Data3DMesh):

    def __init__(self):
        super().__init__()

        self.convex_hull_vertices: np.ndarray | None = None
        self.convex_hull_volume: float | None = None
        self.convex_hull_area: float | None = None

    def compute_convex_hull(self):
        if self.is_empty():
            return

        from scipy.spatial import ConvexHull

        # Generate the convex hull from the mesh vertices
        convex_hull = ConvexHull(self.mesh_vertices)

        # Extract convex hull vertices, area and volume.
        self.convex_hull_vertices = self.mesh_vertices[convex_hull.vertices, :] - np.mean(self.mesh_vertices, axis=0)
        self.convex_hull_area = convex_hull.area
        self.convex_hull_volume = convex_hull.volume


class Data3DAxisAlignedBoundingBox(Data3DConvexHull):

    def __init__(self):
        super().__init__()

        self.bounding_box_volume: float | None = None
        self.bounding_box_area: float | None = None

    def compute_bounding_box(self):
        if self.is_empty():
            return

        dims = np.max(self.convex_hull_vertices, axis=0) - np.min(self.convex_hull_vertices, axis=0)
        self.bounding_box_volume = np.prod(dims)
        self.bounding_box_area = 2.0 * dims[0] * dims[1] + 2.0 * dims[0] * dims[2] + 2.0 * dims[1] * dims[2]


class Data3DOrientedMinimumBoundingBox(Data3DConvexHull):

    def __init__(self):
        super().__init__()

        self.bounding_box_volume: float | None = None
        self.bounding_box_area: float | None = None

    def compute_bounding_box(self):
        if self.is_empty():
            return

        rot_df = pd.DataFrame({
            "rot_axis_0": np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
            "rot_axis_1": np.array([1, 2, 1, 2, 0, 2, 0, 2, 0, 1, 0, 1]),
            "rot_axis_2": np.array([2, 1, 0, 0, 2, 0, 1, 1, 1, 0, 2, 2]),
            "aabb_axis_0": np.zeros(12),
            "aabb_axis_1": np.zeros(12),
            "aabb_axis_2": np.zeros(12),
            "vol": np.zeros(12)
        })

        # Rotate over different sequences
        for ii in np.arange(0, len(rot_df)):
            # Create a local copy
            work_pos = copy.deepcopy(self.convex_hull_vertices)

            # Rotate over sequence of rotation axes
            work_pos = self._get_optimally_rotated_volume(input_pos=work_pos, rot_axis=rot_df.rot_axis_0[ii])
            work_pos = self._get_optimally_rotated_volume(input_pos=work_pos, rot_axis=rot_df.rot_axis_1[ii])
            work_pos = self._get_optimally_rotated_volume(input_pos=work_pos, rot_axis=rot_df.rot_axis_2[ii])

            # Determine resultant minimum bounding box
            aabb_dims = np.max(work_pos, axis=0) - np.min(work_pos, axis=0)
            rot_df.loc[ii, "aabb_axis_0"] = aabb_dims[0]
            rot_df.loc[ii, "aabb_axis_1"] = aabb_dims[1]
            rot_df.loc[ii, "aabb_axis_2"] = aabb_dims[2]
            rot_df.loc[ii, "vol"] = np.prod(aabb_dims)

            del work_pos, aabb_dims

        # Find minimal volume of all rotations and return bounding box dimensions
        sel_row = rot_df.loc[rot_df.vol.idxmin(), :]
        dims = np.array([sel_row.aabb_axis_0, sel_row.aabb_axis_1, sel_row.aabb_axis_2])

        self.bounding_box_volume = np.prod(dims)
        self.bounding_box_area = 2.0 * dims[0] * dims[1] + 2.0 * dims[0] * dims[2] + 2.0 * dims[1] * dims[2]

    def _get_optimally_rotated_volume(
            self,
            input_pos: np.ndarray,
            rot_axis: int,
            n_minima: int = 3,
            res_init: float = 5.0
    ) -> np.ndarray:
        from scipy.spatial import ConvexHull

        # Find axis aligned bounding box of the point set
        aabb_max = np.max(input_pos, axis=0)
        aabb_min = np.min(input_pos, axis=0)

        # Center the point set at the AABB center
        output_pos = input_pos - 0.5 * (aabb_min + aabb_max)

        # Project model to plane
        proj_pos = copy.deepcopy(output_pos)
        proj_pos = np.delete(proj_pos, [rot_axis], axis=1)

        # Calculate 2D convex hull of the model projection in plane
        if np.shape(proj_pos)[0] >= 10:
            hull_2d = ConvexHull(points=proj_pos)
            hull_mat = proj_pos[hull_2d.vertices, :]
            del hull_2d, proj_pos
        else:
            hull_mat = proj_pos
            del proj_pos

        # Transpose hull_mat so that the array is (ndim, npoints) instead of (npoints, ndim)
        hull_mat = np.transpose(hull_mat)

        # Calculate bounding box surface of a series of rotated contours
        # Note we can program a min-search algorithm here as well

        # Calculate initial surfaces
        theta_init = np.arange(start=0.0, stop=90.0 + res_init, step=res_init) * np.pi / 180.0
        rot_area = np.array(list(map(
            lambda x: self._get_rotated_axis_aligned_bounding_box_area(theta=x, hull_mat=hull_mat), theta_init
        )))

        # Find local minima
        df_min = self._find_peaks(x=rot_area, direction="neg")

        # Check if any minimum was generated
        if len(df_min) > 0:
            # Investigate up to n_minima number of local minima, starting with the global minimum
            df_min = df_min.sort_values(by="val", ascending=True)

            # Determine max number of minima evaluated
            max_iter = np.min([n_minima, len(df_min)])

            # Initialise placeholder array
            theta_min = np.zeros(max_iter)

            # Iterate over local minima
            for k in np.arange(0, max_iter):
                # Find initial angle corresponding to i-th minimum
                sel_ind = df_min.ind.values[k]
                theta_curr = theta_init[sel_ind]

                # Zoom in to improve the approximation of theta
                theta_min[k] = self._find_optimal_rotation_angle(
                    hull_mat=hull_mat,
                    theta_sel=theta_curr,
                    res=res_init * np.pi / 180.0
                )

            # Calculate surface areas corresponding to theta_min and theta that minimises the surface
            rot_area = np.array(list(map(
                lambda x: self._get_rotated_axis_aligned_bounding_box_area(theta=x, hull_mat=hull_mat), theta_min
            )))
            theta_sel = theta_min[np.argmin(rot_area)]

        else:
            theta_sel = theta_init[0]

        # Rotate original point along the angle that minimises the projected AABB area
        output_pos = np.transpose(output_pos)
        rot_mat = self._get_rotation_matrix(theta=theta_sel, dim=3, rot_axis=rot_axis)
        output_pos = np.dot(rot_mat, output_pos)

        # Rotate output_pos back to (npoints, ndim)
        output_pos = np.transpose(output_pos)

        return output_pos

    def _get_rotated_axis_aligned_bounding_box_area(self, theta, hull_mat):
        # Function to calculate surface of the axis-aligned bounding box of a rotated 2D contour

        # Create rotation matrix and rotate over theta
        rot_mat = self._get_rotation_matrix(theta=theta, dim=2)
        rot_hull = np.dot(rot_mat, hull_mat)

        # Calculate bounding box surface of the rotated contour
        rot_aabb_dims = np.max(rot_hull, axis=1) - np.min(rot_hull, axis=1)
        rot_aabb_area = np.prod(rot_aabb_dims)

        return rot_aabb_area

    def _find_optimal_rotation_angle(self, hull_mat, theta_sel, res, max_rep=5):
        # Iterative approximator for finding angle theta that minimises surface area
        for jj in np.arange(0, max_rep):

            # Select new thetas in vicinity of
            theta = np.array([theta_sel-res, theta_sel-0.5*res, theta_sel, theta_sel+0.5*res, theta_sel+res])

            # Calculate projection areas for current angles theta
            rot_area = np.array(list(map(
                lambda x: self._get_rotated_axis_aligned_bounding_box_area(theta=x, hull_mat=hull_mat), theta)
            ))

            # Find global minimum and corresponding angle theta_sel
            theta_sel = theta[np.argmin(rot_area)]

            # Shrink resolution and iterate
            res /= 2.0

        return theta_sel

    @staticmethod
    def _get_rotation_matrix(theta, dim=2, rot_axis=-1):
        # Creates a 2d or 3d rotation matrix
        if dim == 2:
            rot_mat = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])

        elif dim == 3:
            if rot_axis == 0:
                rot_mat = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(theta), -np.sin(theta)],
                    [0.0, np.sin(theta), np.cos(theta)]
                ])

            elif rot_axis == 1:
                rot_mat = np.array([
                    [np.cos(theta), 0.0, np.sin(theta)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(theta), 0.0, np.cos(theta)]
                ])

            elif rot_axis == 2:
                rot_mat = np.array([
                    [np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0]
                ])
            else:
                rot_mat = None
        else:
            rot_mat = None

        return rot_mat

    def _find_peaks(self, x, direction="pos"):
        # Determines peak positions in array of values

        # Invert when looking for local minima
        if direction == "neg":
            x = -x

        # Generate segments where slope is negative
        df_segm = self._segmentise_input(x=(np.diff(x) < 0.0) * 1)

        # Start of slope coincides with position of peak (due to index shift induced by np.diff)
        ind_peak = df_segm.loc[df_segm.val == 1, "i"].values

        # Check right boundary
        if x[-1] > x[-2]:
            ind_peak = np.append(ind_peak, np.shape(x)[0]-1)

        # Construct dataframe with index and corresponding value
        if np.shape(ind_peak)[0] == 0:
            df_peak = pd.DataFrame({"ind": [], "val": []})
        else:
            if direction == "pos":
                # Positive direction
                df_peak = pd.DataFrame({"ind": ind_peak, "val": x[ind_peak]})
            else:
                # Negative direction
                df_peak = pd.DataFrame({"ind": ind_peak, "val": -x[ind_peak]})

        return df_peak

    @staticmethod
    def _segmentise_input(x):
        # Produces a list of segments from input x with values (0,1)

        # Create a difference vector
        y = np.diff(x)

        # Find start and end indices of sections with value 1
        ind_1_start = np.array(np.where(y == 1)).flatten()
        if np.shape(ind_1_start)[0] > 0:
            ind_1_start += 1
        ind_1_end = np.array(np.where(y == -1)).flatten()

        # Check for boundary effects
        if x[0] == 1:
            ind_1_start = np.insert(ind_1_start, 0, 0)
        if x[-1] == 1:
            ind_1_end = np.append(ind_1_end, np.shape(x)[0] - 1)

        # Generate segment df for segments with value 1
        if np.shape(ind_1_start)[0] == 0:
            df_one = pd.DataFrame({
                "i": [],
                "j": [],
                "val": []
            })
        else:
            df_one = pd.DataFrame({
                "i": ind_1_start,
                "j": ind_1_end,
                "val": np.ones(np.shape(ind_1_start)[0])
            })

        # Find start and end indices for section with value 0
        if np.shape(ind_1_start)[0] == 0:
            ind_0_start = np.array([0])
            ind_0_end = np.array([np.shape(x)[0] - 1])

        else:
            ind_0_end = ind_1_start - 1
            ind_0_start = ind_1_end + 1

            # Check for boundary effect
            if x[0] == 0:
                ind_0_start = np.insert(ind_0_start, 0, 0)
            if x[-1] == 0:
                ind_0_end = np.append(ind_0_end, np.shape(x)[0] - 1)

            # Check for out-of-range boundary effects
            if ind_0_end[0] < 0:
                ind_0_end = np.delete(ind_0_end, 0)
            if ind_0_start[-1] >= np.shape(x)[0]:
                ind_0_start = np.delete(ind_0_start, -1)

        # Generate segment df for segments with value 0
        if np.shape(ind_0_start)[0] == 0:
            df_zero = pd.DataFrame({
                "i": [],
                "j": [],
                "val": []
            })

        else:
            df_zero = pd.DataFrame({
                "i": ind_0_start,
                "j": ind_0_end,
                "val": np.zeros(np.shape(ind_0_start)[0])
            })

        df_segm = pd.concat(
            [df_one, df_zero],
            ignore_index=True
        ).sort_values(by="i").reset_index(drop=True)

        return df_segm


class Data3DPrincipleComponents(Data3DMesh):

    def __init__(self):
        super().__init__()

        self.semi_axes: tuple[float, float, float] | None = None

    def compute_semi_axes(self):
        if self.is_empty():
            return

        # Get position matrix
        pos_mat_pca = self.data_morph[["z", "y", "x"]].values

        # Subtract mean
        pos_mat_pca = np.multiply(
            (pos_mat_pca - np.mean(pos_mat_pca, axis=0)),
            self.spacing
        )

        # Get eigenvalues and vectors
        if not self.is_singular():
            eigen_val, eigen_vec = np.linalg.eigh(np.cov(pos_mat_pca, rowvar=False))
            self.semi_axes = tuple(2.0 * np.sqrt(np.sort(eigen_val)))

    def get_ellipsoid_surface_area(self, n_degree=10):
        # Approximates area of an ellipsoid using legendre polynomials

        # Let semi_axes[2] be the major semi-axis length, semi_axes[1] the minor semi-axis length and semi_axes[0] the
        # least axis length.

        # Import legendre evaluation function from numpy
        from numpy.polynomial.legendre import legval

        # Check if the semi-axes differ in length, otherwise the ellipsoid is spherical and legendre polynomials are not
        # required.
        if self.semi_axes[0] == self.semi_axes[1] and self.semi_axes[0] == self.semi_axes[2]:
            # Exact sphere calculation
            area_appr = 4.0 * np.pi * self.semi_axes[0] ** 2.0

        elif self.semi_axes[0] == self.semi_axes[1]:
            # Exact prolate spheroid (major semi-axis > other, equally long, semi-axes)
            ecc = np.sqrt(1.0 - (self.semi_axes[0] ** 2.0) / (self.semi_axes[2] ** 2.0))

            # Calculate area
            area_appr = (
                    2.0 * np.pi * self.semi_axes[0] ** 2.0
                    * (1.0 + self.semi_axes[2] * np.arcsin(ecc) / (self.semi_axes[0] * ecc))
            )

        elif self.semi_axes[1] == self.semi_axes[2]:
            # Exact oblate spheroid (major semi-axes equally long > shortest semi-axis)
            ecc = np.sqrt(1.0 - (self.semi_axes[0] ** 2.0) / (self.semi_axes[2] ** 2.0))

            # Calculate area:
            area_appr = 2.0 * np.pi * self.semi_axes[2] ** 2.0 * (1.0 + ((1.0 - ecc ** 2.0) / ecc) * np.arctanh(ecc))

        else:
            # Tri-axial ellipsoid
            # Calculate eccentricities
            ecc_alpha = np.sqrt(1.0 - (self.semi_axes[1] ** 2.0) / (self.semi_axes[2] ** 2.0))
            ecc_beta = np.sqrt(1.0 - (self.semi_axes[0] ** 2.0) / (self.semi_axes[2] ** 2.0))

            # Create a vector to generate coefficients for the Legendre polynomial
            nu = np.arange(start=0, stop=n_degree + 1, step=1) * 1.0

            # Define Legendre polynomial coefficients and evaluation point
            leg_coeff = (ecc_alpha * ecc_beta) ** nu / (1.0 - 4.0 * nu ** 2.0)
            leg_x = (ecc_alpha ** 2.0 + ecc_beta ** 2.0) / (2.0 * ecc_alpha * ecc_beta)

            # Calculate approximate area
            area_appr = 4.0 * np.pi * self.semi_axes[2] * self.semi_axes[1] * legval(x=leg_x, c=leg_coeff)

        return area_appr


class Data3DSpatial(Data3DMesh):

    def __init__(self):
        super().__init__()

        self.moran_i: float | None = None
        self.geary_c: float | None = None

    def compute_spatial_information(
            self,
            image: GenericImage,
            mask: BaseMask,
            allow_approximation: bool = True
    ):
        n_v_int = len(self.data_int)

        if (1 < n_v_int < 1000) or not allow_approximation:
            # Calculate geospatial features using a brute force approach
            self.moran_i, self.geary_c = self._spatial_distribution(data=self.data_int)

        elif n_v_int >= 1000:
            # Use monte carlo approach to estimate geospatial features

            m = hashlib.sha1(usedforsecurity=False)
            m = image.update_hash(m=m)
            m = mask.roi_morphology.update_hash(m=m)
            randomiser = np.random.default_rng(int(m.hexdigest(), 16))

            # Create lists for storing feature values
            moran_list, geary_list = [], []

            # Initiate iterations
            iter_nr = 1
            tol_aim = 0.002
            tol_sem = 1.000

            # Iterate until the sample error of the mean drops below the target tol_aim
            while tol_sem > tol_aim:

                # Select a small random subset of 100 points in the volume
                curr_points = randomiser.choice(n_v_int, size=100, replace=False)

                # Calculate Moran's I and Geary's C for the point subset
                moran_i, geary_c = self._spatial_distribution(data=self.data_int.loc[curr_points, :])

                # Append values to the lists
                moran_list.append(moran_i)
                geary_list.append(geary_c)

                # From the tenth iteration, estimate the sample error of the mean
                if iter_nr > 10:
                    tol_sem = np.max([np.std(moran_list), np.std(geary_list)]) / np.sqrt(iter_nr)

                # Update counter
                iter_nr += 1

            self.moran_i = np.mean(moran_list)
            self.geary_c = np.mean(geary_list)

    def _spatial_distribution(self, data: pd.DataFrame):
        n_v = len(data)

        # Generate point cloud
        pos_mat = data[["z", "y", "x"]].values
        pos_mat = np.multiply(pos_mat, self.spacing)

        if n_v < 2000:
            # Determine all interactions between voxels
            comb_iter = np.array([np.tile(np.arange(0, n_v), n_v), np.repeat(np.arange(0, n_v), n_v)])
            comb_iter = comb_iter[:, comb_iter[0, :] > comb_iter[1, :]]

            # Determine weighting for all interactions (inverse weighting with distance)
            w_ij = 1.0 / np.array(list(
                map(lambda i: np.sqrt(np.sum((pos_mat[comb_iter[0, i], :] - pos_mat[comb_iter[1, i], :]) ** 2.0)),
                    np.arange(np.shape(comb_iter)[1]))))

            # Create array of mean-corrected grey level intensities
            gl_dev = data.g.values - np.mean(data.g)

            # Moran's I
            nom = n_v * np.sum(np.multiply(np.multiply(w_ij, gl_dev[comb_iter[0, :]]), gl_dev[comb_iter[1, :]]))
            denom = np.sum(w_ij) * np.sum(gl_dev ** 2.0)
            if denom > 0.0:
                moran_i = nom / denom
            else:
                # If the denominator is 0.0, this basically means only one intensity is present in the volume, which
                # indicates ideal spatial correlation.
                moran_i = 1.0

            # Geary's C
            nom = (n_v - 1.0) * np.sum(np.multiply(w_ij, (gl_dev[comb_iter[0, :]] - gl_dev[comb_iter[1, :]]) ** 2.0))
            if denom > 0.0:
                geary_c = nom / (2.0 * denom)
            else:
                # If the denominator is 0.0, this basically means only one intensity is present in the volume.
                geary_c = 1.0
        else:
            # In practice, this code variant is only used if the ROI is too large to perform all distance calculations
            # in one go.

            # Create array of mean-corrected grey level intensities
            gl_dev = data.g.values - np.mean(data.g)

            moran_nom = 0.0
            geary_nom = 0.0
            w_denom = 0.0

            # Iterate over voxels
            for ii in np.arange(n_v - 1):
                # Get all jj > ii voxels
                jj = np.arange(start=ii + 1, stop=n_v)

                # Get distance weights
                w_iijj = 1.0 / np.sqrt(np.sum(np.power(pos_mat[ii, :] - pos_mat[jj, :], 2.0), axis=1))

                moran_nom += np.sum(np.multiply(np.multiply(w_iijj, gl_dev[ii]), gl_dev[jj]))
                geary_nom += np.sum(np.multiply(w_iijj, (gl_dev[ii] - gl_dev[jj]) ** 2.0))
                w_denom += np.sum(w_iijj)

            gl_denom = np.sum(gl_dev ** 2.0)

            # Moran's I index
            if gl_denom > 0.0:
                moran_i = n_v * moran_nom / (w_denom * gl_denom)
            else:
                moran_i = 1.0

            # Geary's C measure
            if gl_denom > 0.0:
                geary_c = (n_v - 1.0) * geary_nom / (2 * w_denom * gl_denom)
            else:
                geary_c = 1.0

        return moran_i, geary_c
