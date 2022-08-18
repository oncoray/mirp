import numpy as np
import pandas as pd

from mirp.imageClass import ImageClass
from mirp.roiClass import RoiClass
from mirp.importSettings import FeatureExtractionSettingsClass


def get_volumetric_morphological_features(img_obj: ImageClass,
                                          roi_obj: RoiClass,
                                          settings: FeatureExtractionSettingsClass):
    """
    Extract morphological features from the image volume
    :param img_obj: image object
    :param roi_obj: roi object with the requested ROI mask
    :param settings: settings object
    :return: pandas DataFrame with feature values
    """

    # Import functions
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist
    from skimage.measure import mesh_surface_area

    # Create feature table
    feat_names = ["morph_volume", "morph_vol_approx", "morph_area_mesh", "morph_av", "morph_comp_1", "morph_comp_2",
                  "morph_sph_dispr", "morph_sphericity", "morph_asphericity", "morph_com",
                  "morph_diam", "morph_pca_maj_axis", "morph_pca_min_axis", "morph_pca_least_axis",
                  "morph_pca_elongation", "morph_pca_flatness", "morph_vol_dens_aabb", "morph_area_dens_aabb",
                  "morph_vol_dens_aee", "morph_area_dens_aee", "morph_vol_dens_conv_hull",
                  "morph_area_dens_conv_hull", "morph_integ_int", "morph_moran_i", "morph_geary_c"]

    if not settings.ibsi_compliant:
        feat_names += ["morph_vol_dens_ombb", "morph_area_dens_ombb",  "morph_vol_dens_mvee", "morph_area_dens_mvee"]

    df_feat = pd.DataFrame(np.full(shape=(1, len(feat_names)), fill_value=np.nan))
    df_feat.columns = feat_names

    # Skip calculations if input information is missing
    if img_obj.is_missing or roi_obj.roi_intensity is None or roi_obj.roi_morphology is None:
        return df_feat

    # Number of voxels within masks
    n_v_morph = np.sum(roi_obj.roi_morphology.get_voxel_grid())
    n_v_int = np.sum(roi_obj.roi_intensity.get_voxel_grid())

    # Check if any voxels are within the morphological mask
    if n_v_morph == 0:
        return df_feat

    ####################################################################################################################
    # Geometric features
    ####################################################################################################################

    # Surface area and volume from surface mesh
    mesh_verts, mesh_faces, mesh_norms = mesh_voxels(roi_obj=roi_obj)

    # Volume
    volume = mesh_volume(vertices=mesh_verts, faces=mesh_faces)
    df_feat["morph_volume"] = volume

    # Approximate volume
    df_feat["morph_vol_approx"] = n_v_morph * np.prod(roi_obj.roi_morphology.spacing)

    # Surface area
    area = mesh_surface_area(verts=mesh_verts, faces=mesh_faces)
    df_feat["morph_area_mesh"] = area

    # Surface to volume ratio
    df_feat["morph_av"] = area / volume

    # Compactness 1
    sphericity_base_feature = 36 * np.pi * volume ** 2.0 / area ** 3.0
    df_feat["morph_comp_1"] = 1.0 / (6.0 * np.pi) * sphericity_base_feature ** (1.0 / 2.0)

    # Compactness 2
    df_feat["morph_comp_2"] = sphericity_base_feature

    # Spherical disproportion
    df_feat["morph_sph_dispr"] = sphericity_base_feature ** (-1.0 / 3.0)

    # Sphericity
    df_feat["morph_sphericity"] = sphericity_base_feature ** (1.0 / 3.0)

    # Asphericity
    df_feat["morph_asphericity"] = sphericity_base_feature ** (-1.0 / 3.0) - 1.0
    del sphericity_base_feature

    ####################################################################################################################
    # Convex hull - based features
    ####################################################################################################################

    # Generate the convex hull from the mesh vertices
    conv_hull = ConvexHull(mesh_verts)

    # Extract convex hull vertices
    hull_verts = mesh_verts[conv_hull.vertices, :] - np.mean(mesh_verts, axis=0)

    # Maximum 3D diameter
    df_feat["morph_diam"] = np.max(pdist(hull_verts))

    # Volume density - convex hull
    df_feat["morph_vol_dens_conv_hull"] = volume / conv_hull.volume

    # Area density - convex hull
    df_feat["morph_area_dens_conv_hull"] = area / conv_hull.area
    del conv_hull

    ####################################################################################################################
    # Bounding box - based features
    ####################################################################################################################

    # Volume density - axis-aligned bounding box
    aabb_dims = np.max(hull_verts, axis=0) - np.min(hull_verts, axis=0)
    df_feat["morph_vol_dens_aabb"] = volume / np.product(aabb_dims)

    # Area density - axis-aligned bounding box
    df_feat["morph_area_dens_aabb"] = area / (2.0 * aabb_dims[0] * aabb_dims[1] +
                                              2.0 * aabb_dims[0] * aabb_dims[2] +
                                              2.0 * aabb_dims[1] * aabb_dims[2])
    del aabb_dims

    if not settings.ibsi_compliant:
        # Volume density - oriented minimum bounding box
        ombb_dims = get_minimum_oriented_bounding_box(pos_mat=hull_verts)
        df_feat["morph_vol_dens_ombb"] = volume / np.product(ombb_dims)

        # Area density - oriented minimum bounding box
        df_feat["morph_area_dens_ombb"] = area / (2.0 * ombb_dims[0] * ombb_dims[1] +
                                                  2.0 * ombb_dims[0] * ombb_dims[2] +
                                                  2.0 * ombb_dims[1] * ombb_dims[2])
        del ombb_dims

    ####################################################################################################################
    # Minimum volume enclosing ellipsoid - based features
    ####################################################################################################################
    if not settings.ibsi_compliant:
        # Calculate semi_axes of minimum volume enclosing ellipsoid
        semi_axes = get_minimum_volume_enclosing_ellipsoid(pos_mat=hull_verts, tolerance=10E-4)

        # Volume density - minimum volume enclosing ellipsoid
        df_feat["morph_vol_dens_mvee"] = 3 * volume / (4 * np.pi * np.prod(semi_axes))

        # Area density - minimum volume enclosing ellipsoid
        df_feat["morph_area_dens_mvee"] = area / get_ellipsoid_surface_area(semi_axes, n_degree=20)
        del semi_axes, hull_verts

    ####################################################################################################################
    # Principal component analysis - based features
    ####################################################################################################################

    # Generate position table and get position matrix
    df_img = roi_obj.as_pandas_dataframe(img_obj=img_obj, intensity_mask=True, morphology_mask=True)

    # Define tables based on morphological and intensity masks
    df_int = df_img[df_img.roi_int_mask].reset_index()
    df_morph = df_img[df_img.roi_morph_mask].reset_index()
    del df_img

    # Get position matrix
    pos_mat_pca = df_morph[["z", "y", "x"]].values

    # Subtract mean
    pos_mat_pca = np.multiply((pos_mat_pca - np.mean(pos_mat_pca, axis=0)), roi_obj.roi_morphology.spacing)

    # Get eigenvalues and vectors
    if n_v_morph > 1:
        eigen_val, eigen_vec = np.linalg.eigh(np.cov(pos_mat_pca, rowvar=False))
        semi_axes = 2.0 * np.sqrt(np.sort(eigen_val))

        # Major axis length
        df_feat["morph_pca_maj_axis"] = semi_axes[2] * 2.0

        # Minor axis length
        df_feat["morph_pca_min_axis"] = semi_axes[1] * 2.0

        # Least axis length
        df_feat["morph_pca_least_axis"] = semi_axes[0] * 2.0

        # Elongation
        df_feat["morph_pca_elongation"] = semi_axes[1] / semi_axes[2]

        # Flatness
        df_feat["morph_pca_flatness"] = semi_axes[0] / semi_axes[2]

        # Volume density - approximate enclosing ellipsoid
        if not np.any(semi_axes == 0):
            df_feat["morph_vol_dens_aee"] = 3 * volume / (4 * np.pi * np.prod(semi_axes))
        else:
            df_feat["morph_vol_dens_aee"] = np.nan

        # Area density - approximate enclosing ellipsoid
        if not np.any(semi_axes == 0):
            df_feat["morph_area_dens_aee"] = area / get_ellipsoid_surface_area(semi_axes, n_degree=20)
        else:
            df_feat["morph_area_dens_aee"] = np.nan
        del semi_axes, pos_mat_pca

    ####################################################################################################################
    # Geospatial analysis - based features
    ####################################################################################################################

    if (1000 > n_v_int > 1) or settings.no_approximation:
        # Calculate geospatial features using a brute force approach
        moran_i, geary_c = geospatial(df_int=df_int, spacing=roi_obj.roi_intensity.spacing)

        df_feat["morph_moran_i"] = moran_i
        df_feat["morph_geary_c"] = geary_c

    elif n_v_int >= 1000:
        # Use monte carlo approach to estimate geospatial features

        # Create lists for storing feature values
        moran_list, geary_list = [], []

        # Initiate iterations
        iter_nr = 1
        tol_aim = 0.002
        tol_sem = 1.000

        # Iterate until the sample error of the mean drops below the target tol_aim
        while tol_sem > tol_aim:

            # Select a small random subset of 100 points in the volume
            curr_points = np.random.choice(n_v_int, size=100, replace=False)

            # Calculate Moran's I and Geary's C for the point subset
            moran_i, geary_c = geospatial(df_int=df_int.loc[curr_points, :], spacing=roi_obj.roi_intensity.spacing)

            # Append values to the lists
            moran_list.append(moran_i)
            geary_list.append(geary_c)

            # From the tenth iteration, estimate the sample error of the mean
            if iter_nr > 10:
                tol_sem = np.max([np.std(moran_list), np.std(geary_list)]) / np.sqrt(iter_nr)

            # Update counter
            iter_nr += 1

            del curr_points, moran_i, geary_c

        # Calculate approximate Moran's I and Geary's C
        df_feat["morph_moran_i"] = np.mean(moran_list)
        df_feat["morph_geary_c"] = np.mean(geary_list)

        del iter_nr

    ####################################################################################################################
    # Intensity and hybrid analysis - based features
    ####################################################################################################################

    if n_v_int > 0:
        # Centre of mass shift
        # Calculate centres of mass for the morphological and intensity masks
        com_morph = np.array([np.mean(df_morph.z), np.mean(df_morph.y), np.mean(df_morph.x)])
        com_int = np.array([np.sum(df_int.g * df_int.z), np.sum(df_int.g * df_int.y),
                            np.sum(df_int.g * df_int.x)]) / np.sum(df_int.g)

        # Calculate shift
        df_feat["morph_com"] = np.sqrt(np.sum(np.multiply((com_morph - com_int), roi_obj.roi.spacing) ** 2.0))
        del com_morph, com_int

        # Integrated intensity
        df_feat["morph_integ_int"] = volume * np.mean(df_int.g)

    return df_feat


def mesh_voxels(roi_obj):
    """Generate a closed mesh from the morphological mask"""

    from skimage.measure import marching_cubes

    # Get ROI and pad with empty voxels
    morphology_mask = np.pad(roi_obj.roi_morphology.get_voxel_grid(), pad_width=1, mode="constant", constant_values=0.0)

    # Use marching cubes to generate a mesh grid for the ROI
    vertices, faces, norms, values = marching_cubes(volume=morphology_mask, level=0.5, spacing=tuple(roi_obj.roi_morphology.spacing))

    return vertices, faces, norms


def mesh_volume(vertices, faces):
    # Calculate volume enclosed by triangular mesh

    # Shift vertices to origin
    vertices += -np.min(vertices, axis=0)

    # Get vertices for each face
    a = vertices[faces[:, 0], :]
    b = vertices[faces[:, 1], :]
    c = vertices[faces[:, 2], :]

    # Vectorised volume calculation. The volume contributed by each face is 1/6 * ( a dot (b cross c) ). Each
    # contributing volume is then summed, and the absolute value taken as volume is always positive.
    volume = np.abs(np.sum(1.0 / 6.0 * np.einsum("ij,ij->i", a, np.cross(b, c, 1, 1))))

    return volume


def get_ellipsoid_surface_area(semi_axes, n_degree=10):
    # Approximates area of an ellipsoid using legendre polynomials

    # Let semi_axes[2] be the major semi-axis length, semi_axes[1] the minor semi-axis length and semi_axes[0] the
    # least axis length.

    # Import legendre evaluation function from numpy
    from numpy.polynomial.legendre import legval

    # Check if the semi-axes differ in length, otherwise the ellipsoid is spherical and legendre polynomials are not required
    if semi_axes[0] == semi_axes[1] and semi_axes[0] == semi_axes[2]:
        # Exact sphere calculation
        area_appr = 4.0 * np.pi * semi_axes[0] ** 2.0
    elif semi_axes[0] == semi_axes[1]:
        # Exact prolate spheroid (major semi-axis > other, equally long, semi-axes)
        ecc = np.sqrt(1 - (semi_axes[0] ** 2.0) / (semi_axes[2] ** 2.0))

        # Calculate area
        area_appr = 2.0 * np.pi * semi_axes[0] ** 2.0 * (1 + semi_axes[2] * np.arcsin(ecc) / (semi_axes[0] * ecc))

    elif semi_axes[1] == semi_axes[2]:
        # Exact oblate spheroid (major semi-axes equally long > shortest semi-axis)
        ecc = np.sqrt(1 - (semi_axes[0] ** 2.0) / (semi_axes[2] ** 2.0))

        # Calculate area:
        area_appr = 2.0 * np.pi * semi_axes[2] ** 2.0 * (1 + ((1 - ecc ** 2.0) / ecc) * np.arctanh(ecc))

    else:
        # Tri-axial ellipsoid
        # Calculate eccentricities
        ecc_alpha = np.sqrt(1 - (semi_axes[1] ** 2.0) / (semi_axes[2] ** 2.0))
        ecc_beta = np.sqrt(1 - (semi_axes[0] ** 2.0) / (semi_axes[2] ** 2.0))

        # Create a vector to generate coefficients for the Legendre polynomial
        nu = np.arange(start=0, stop=n_degree + 1, step=1) * 1.0

        # Define Legendre polynomial coefficients and evaluation point
        leg_coeff = (ecc_alpha * ecc_beta) ** nu / (1.0 - 4.0 * nu ** 2.0)
        leg_x = (ecc_alpha ** 2.0 + ecc_beta ** 2.0) / (2.0 * ecc_alpha * ecc_beta)

        # Calculate approximate area
        area_appr = 4.0 * np.pi * semi_axes[2] * semi_axes[1] * legval(x=leg_x, c=leg_coeff)
    # else:
    #     # Exact spheroid calculation
    #     area_appr = 4.0 * np.pi * semi_axes[0] ** 2.0

    return area_appr


def get_minimum_oriented_bounding_box(pos_mat):
    # Implementation of Chan and Tan's algorithm (C.K. Chan, S.T. Tan. Determination of the minimum bounding box of an
    # arbitrary solid: an iterative approach. Comp Struc 79 (2001) 1433-1449

    import copy
    from scipy.spatial import ConvexHull

    ##########################
    # Internal functions
    ##########################

    def get_rotated_axis_aligned_bounding_box_area(theta, hull_mat):
        # Function to calculate surface of the axis-aligned bounding box of a rotated 2D contour

        # Create rotation matrix and rotate over theta
        rot_mat = get_rotation_matrix(theta=theta, dim=2)
        rot_hull = np.dot(rot_mat, hull_mat)

        # Calculate bounding box surface of the rotated contour
        rot_aabb_dims = np.max(rot_hull, axis=1) - np.min(rot_hull, axis=1)
        rot_aabb_area = np.product(rot_aabb_dims)

        return rot_aabb_area

    def find_optimal_rotation_angle(hull_mat, theta_sel, res, max_rep=5):
        # Iterative approximator for finding angle theta that minimises surface area
        for jj in np.arange(0, max_rep):

            # Select new thetas in vicinity of
            theta = np.array([theta_sel-res, theta_sel-0.5*res, theta_sel, theta_sel+0.5*res, theta_sel+res])

            # Calculate projection areas for current angles theta
            rot_area = np.array(list(map(lambda x: get_rotated_axis_aligned_bounding_box_area(theta=x, hull_mat=hull_mat), theta)))

            # Find global minimum and corresponding angle theta_sel
            theta_sel = theta[np.argmin(rot_area)]

            # Shrink resolution and iterate
            res /= 2.0

        return theta_sel

    def get_optimally_rotated_volume(input_pos, rot_axis, n_minima=3, res_init=5.0):
        # Function to that rotates input_pos to find the rotation that minimises the projection of input_pos on the
        # plane normal to the rot_axis

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
        theta_init = np.arange(start=0.0, stop=90.0+res_init, step=res_init) * np.pi / 180.0
        rot_area = np.array(list(map(lambda x: get_rotated_axis_aligned_bounding_box_area(theta=x, hull_mat=hull_mat), theta_init)))

        # Find local minima
        df_min = find_peaks(x=rot_area, direction="neg")

        # Check if any minimum was generated
        if len(df_min) > 0:
            # Investigate up to n_minima number of local minima, starting with the global minimum
            df_min = df_min.sort_values(by="val", ascending=True)

            # Determine max number of minima evaluated
            max_iter = np.min([n_minima, len(df_min)])

            # Initialise place holder array
            theta_min = np.zeros(max_iter)

            # Iterate over local minima
            for k in np.arange(0, max_iter):

                # Find initial angle corresponding to i-th minimum
                sel_ind = df_min.ind.values[k]
                theta_curr = theta_init[sel_ind]

                # Zoom in to improve the approximation of theta
                theta_min[k] = find_optimal_rotation_angle(hull_mat=hull_mat, theta_sel=theta_curr, res=res_init*np.pi/180.0)

            # Calculate surface areas corresponding to theta_min and theta that minimises the surface
            rot_area = np.array(list(map(lambda x: get_rotated_axis_aligned_bounding_box_area(theta=x, hull_mat=hull_mat), theta_min)))
            theta_sel = theta_min[np.argmin(rot_area)]

        else:
            theta_sel = theta_init[0]

        # Rotate original point along the angle that minimises the projected AABB area
        output_pos = np.transpose(output_pos)
        rot_mat = get_rotation_matrix(theta=theta_sel, dim=3, rot_axis=rot_axis)
        output_pos = np.dot(rot_mat, output_pos)

        # Rotate output_pos back to (npoints, ndim)
        output_pos = np.transpose(output_pos)

        return output_pos

    ##########################
    # Main function
    ##########################

    rot_df = pd.DataFrame({"rot_axis_0":  np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
                           "rot_axis_1":  np.array([1, 2, 1, 2, 0, 2, 0, 2, 0, 1, 0, 1]),
                           "rot_axis_2":  np.array([2, 1, 0, 0, 2, 0, 1, 1, 1, 0, 2, 2]),
                           "aabb_axis_0": np.zeros(12),
                           "aabb_axis_1": np.zeros(12),
                           "aabb_axis_2": np.zeros(12),
                           "vol":         np.zeros(12)})

    # Rotate over different sequences
    for ii in np.arange(0, len(rot_df)):
        # Create a local copy
        work_pos = copy.deepcopy(pos_mat)

        # Rotate over sequence of rotation axes
        work_pos = get_optimally_rotated_volume(input_pos=work_pos, rot_axis=rot_df.rot_axis_0[ii])
        work_pos = get_optimally_rotated_volume(input_pos=work_pos, rot_axis=rot_df.rot_axis_1[ii])
        work_pos = get_optimally_rotated_volume(input_pos=work_pos, rot_axis=rot_df.rot_axis_2[ii])

        # Determine resultant minimum bounding box
        aabb_dims = np.max(work_pos, axis=0) - np.min(work_pos, axis=0)
        rot_df.loc[ii, "aabb_axis_0"] = aabb_dims[0]
        rot_df.loc[ii, "aabb_axis_1"] = aabb_dims[1]
        rot_df.loc[ii, "aabb_axis_2"] = aabb_dims[2]
        rot_df.loc[ii, "vol"] = np.product(aabb_dims)

        del work_pos, aabb_dims

    # Find minimal volume of all rotations and return bounding box dimensions
    sel_row = rot_df.loc[rot_df.vol.idxmin(), :]
    ombb_dims = np.array([sel_row.aabb_axis_0, sel_row.aabb_axis_1, sel_row.aabb_axis_2])

    return ombb_dims


def get_rotation_matrix(theta, dim=2, rot_axis=-1):
    # Creates a 2d or 3d rotation matrix
    if dim == 2:
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

    elif dim == 3:
        if rot_axis == 0:
            rot_mat = np.array([[1.0, 0.0,           0.0],
                                [0.0, np.cos(theta), -np.sin(theta)],
                                [0.0, np.sin(theta), np.cos(theta)]])
        elif rot_axis == 1:
            rot_mat = np.array([[np.cos(theta),  0.0, np.sin(theta)],
                                [0.0,            1.0, 0.0],
                                [-np.sin(theta), 0.0, np.cos(theta)]])
        elif rot_axis == 2:
            rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                                [np.sin(theta), np.cos(theta),  0.0],
                                [0.0,           0.0,            1.0]])
        else:
            rot_mat = None
    else:
        rot_mat = None

    return rot_mat


def get_minimum_volume_enclosing_ellipsoid(pos_mat, tolerance=10E-4):
    # Calculates the semi-axes of the minimum volume enclosing ellipsoid. This algorithm is based on N. Moshtagh,
    # Minimum volume enclosing ellipsoids (2005), DOI 10.1.1.116.7691

    import copy
    import numpy.linalg as npla

    # Cast to np.matrix
    mat_p = copy.deepcopy(pos_mat)
    mat_p = np.transpose(mat_p)
    mat_p = np.asmatrix(mat_p)

    # Dimension of the point set
    ndim = np.shape(mat_p)[0]

    # Number of points in the point set
    npoint = np.shape(mat_p)[1]

    # Add a row of 1s to the ndim x npoint matrix input_pos - so mat_q is (ndim+1) x npoint now.
    mat_q = np.append(mat_p, np.ones(shape=(1, npoint)), axis=0)

    # Initialise settings for Khachiyan algorithm
    iter_count = 1
    err = 1.0

    # Initial u vector
    vec_u = 1.0 * np.ones(shape=npoint) / npoint

    # Khachiyan algorithm
    while err > tolerance:
        # Matrix multiplication:
        # np.diag(u) : if u is a vector, places the elements of u
        # in the diagonal of an NxN matrix of zeros
        mat_x = mat_q * np.diag(vec_u) * np.transpose(mat_q)

        # npla.inv(mat_x) returns the matrix inverse of mat_x
        # np.diag(mat_M) when mat_M is a matrix returns the diagonal vector of mat_M
        vec_m = np.diag(np.transpose(mat_q) * npla.inv(mat_x) * mat_q)

        # Find the value and location of the maximum element in the vector M
        max_m = np.max(vec_m)
        ind_j = np.argmax(vec_m)

        # Calculate the step size for the ascent
        step_size = (max_m - ndim - 1.0) / ((ndim + 1.0) * (max_m - 1.0))

        # Calculate the vector vec_new_u:
        # Take the vector vec_u, and multiply all the elements in it by (1-step_size)
        vec_new_u = vec_u * (1.0 - step_size)

        # Increment the jth element of new_u by step_size
        vec_new_u[ind_j] = vec_new_u[ind_j] + step_size

        # Store the error by taking finding the square root of the SSD
        # between new_u and u
        # The SSD or sum-of-square-differences, takes two vectors
        # of the same size, creates a new vector by finding the
        # difference between corresponding elements, squaring
        # each difference and adding them all together.
        err = npla.norm(vec_new_u - vec_u)

        # Increment iter_count and replace vec_u
        iter_count += 1
        vec_u = vec_new_u

    # Put the elements of the vector u into the diagonal of a matrix
    # U with the rest of the elements as 0
    mat_u = np.asmatrix(np.diag(vec_u))

    # Compute the A-matrix
    vec_u = np.transpose(np.asmatrix(vec_u))
    mat_a = (1.0 / ndim) * npla.inv(mat_p * mat_u * np.transpose(mat_p) - (mat_p * vec_u) * np.transpose(mat_p * vec_u))

    # Compute the center - This is not required as output.
    # mat_c = mat_p * vec_u

    # Perform singular value decomposition
    s = npla.svd(mat_a, compute_uv=False)

    # The semi-axis lengths are the inverse square root of the of the singular values
    semi_axes_length = np.sort(1.0/np.sqrt(s))

    return semi_axes_length


def segmentise_input(x):
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
        ind_1_end = np.append(ind_1_end, np.shape(x)[0]-1)

    # Generate segment df for segments with value 1
    if np.shape(ind_1_start)[0] == 0:
        df_one = pd.DataFrame({"i":   [],
                               "j":   [],
                               "val": []})
    else:
        df_one = pd.DataFrame({"i":   ind_1_start,
                               "j":   ind_1_end,
                               "val": np.ones(np.shape(ind_1_start)[0])})

    # Find start and end indices for section with value 0
    if np.shape(ind_1_start)[0] == 0:
        ind_0_start = np.array([0])
        ind_0_end = np.array([np.shape(x)[0]-1])
    else:
        ind_0_end = ind_1_start - 1
        ind_0_start = ind_1_end + 1

        # Check for boundary effect
        if x[0] == 0:
            ind_0_start = np.insert(ind_0_start, 0, 0)
        if x[-1] == 0:
            ind_0_end = np.append(ind_0_end, np.shape(x)[0]-1)

        # Check for out-of-range boundary effects
        if ind_0_end[0] < 0:
            ind_0_end = np.delete(ind_0_end, 0)
        if ind_0_start[-1] >= np.shape(x)[0]:
            ind_0_start = np.delete(ind_0_start, -1)

    # Generate segment df for segments with value 0
    if np.shape(ind_0_start)[0] == 0:
        df_zero = pd.DataFrame({"i":   [],
                                "j":   [],
                                "val": []})
    else:
        df_zero = pd.DataFrame({"i":    ind_0_start,
                                "j":    ind_0_end,
                                "val":  np.zeros(np.shape(ind_0_start)[0])})

    df_segm = df_one.append(df_zero).sort_values(by="i").reset_index(drop=True)

    return df_segm


def find_peaks(x, direction="pos"):
    # Determines peak positions in array of values

    # Invert when looking for local minima
    if direction == "neg":
        x = -x

    # Generate segments where slope is negative
    df_segm = segmentise_input(x=(np.diff(x) < 0.0) * 1)

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


def geospatial(df_int, spacing):

    # Define constants
    n_v = len(df_int)

    # Generate point cloud
    pos_mat = df_int[["z", "y", "x"]].values
    pos_mat = np.multiply(pos_mat, spacing)

    if n_v < 2000:
        # Determine all interactions between voxels
        comb_iter = np.array([np.tile(np.arange(0, n_v), n_v), np.repeat(np.arange(0, n_v), n_v)])
        comb_iter = comb_iter[:, comb_iter[0, :] > comb_iter[1, :]]

        # Determine weighting for all interactions (inverse weighting with distance)
        w_ij = 1.0 / np.array(list(
            map(lambda i: np.sqrt(np.sum((pos_mat[comb_iter[0, i], :] - pos_mat[comb_iter[1, i], :]) ** 2.0)),
                np.arange(np.shape(comb_iter)[1]))))

        # Create array of mean-corrected grey level intensities
        gl_dev = df_int.g.values - np.mean(df_int.g)

        # Moran's I
        nom = n_v * np.sum(np.multiply(np.multiply(w_ij, gl_dev[comb_iter[0, :]]), gl_dev[comb_iter[1, :]]))
        denom = np.sum(w_ij) * np.sum(gl_dev ** 2.0)
        if denom > 0.0:
            moran_i = nom / denom
        else:
            # If the denominator is 0.0, this basically means only one intensity is present in the volume, which indicates ideal spatial correlation.
            moran_i = 1.0

        # Geary's C
        nom = (n_v - 1.0) * np.sum(np.multiply(w_ij, (gl_dev[comb_iter[0, :]] - gl_dev[comb_iter[1, :]]) ** 2.0))
        if denom > 0.0:
            geary_c = nom / (2.0 * denom)
        else:
            # If the denominator is 0.0, this basically means only one intensity is present in the volume.
            geary_c = 1.0
    else:
        # In practice, this code variant is only used if the ROI is too large to perform all distance calculations in one go.

        # Create array of mean-corrected grey level intensities
        gl_dev = df_int.g.values - np.mean(df_int.g)

        moran_nom = 0.0
        geary_nom = 0.0
        w_denom = 0.0

        # Iterate over voxels
        for ii in np.arange(n_v-1):
            # Get all jj > ii voxels
            jj = np.arange(start=ii+1, stop=n_v)

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
            geary_c = (n_v - 1.0) * geary_nom / (2*w_denom*gl_denom)
        else:
            geary_c = 1.0

    return moran_i, geary_c
