import numpy as np
import pandas as pd
import copy


# def mesh_area(map_obj, spacing, dims):
#     # Calculates mesh area of ROI surface
#
#     from skimage.measure import marching_cubes, mesh_surface_area
#
#     #Get ROI
#     img_mat = np.reshape(map_obj.vox_type.values, dims)  # Cast ROI to 3D array (mask: NaN - not ROI; 1.0 - ROI)
#     img_mat[np.isnan(img_mat)] = 0.0  # Set missing voxels outside the ROI to 0
#     img_mat = np.pad(img_mat, pad_width=1, mode="constant", constant_values=0.0)  # Pad 3D array with empty voxels
#
#     # Use marching cubes to generate a mesh grid for the ROI
#     verts, faces = marching_cubes(volume=img_mat, level=0.5, spacing=spacing)
#
#     # Calculate mesh surface
#     area = mesh_surface_area(verts=verts, faces=faces)  # Calculate mask surface
#
#     return area
#
#
# def mesh_volume(verts, faces):
#     # Calculate volume enclosed by triangular mesh
#
#     # Shift vertices to origin
#     verts += -np.min(verts, axis=0)
#
#     # Get vertices for each face
#     a = verts[faces[:,0],:]
#     b = verts[faces[:,1],:]
#     c = verts[faces[:,2],:]
#
#     # Vectorised volume calculation. The volume contributed by each face is 1/6 * ( a dot (b cross c) ). Each
#     # contributing volume is then summed, and the absolute value taken as volume is always positive.
#     volume = np.abs(np.sum(1.0 / 6.0 * np.einsum("ij,ij->i", a, np.cross(b, c, 1, 1))))
#
#     return volume
#
#
# def mesh_voxels(map_obj, spacing, dims):
#     # Generates a closed mesh of a voxel space
#
#     from skimage.measure import marching_cubes
#
#     # Get ROI
#     img_mat = np.reshape(map_obj.in_roi.values, dims)  # Cast ROI to 3D array (mask: False - not ROI; True - ROI)
#     img_mat = np.pad(img_mat, pad_width=1, mode="constant", constant_values=0.0)  # Pad 3D array with empty voxels
#
#     # Use marching cubes to generate a mesh grid for the ROI
#     verts, faces = marching_cubes(volume=img_mat, level=0.5, spacing=spacing)
#
#     # Determine normals
#     norms = meshNormals(verts=verts, faces=faces, normalise=True)
#
#     return verts, faces, norms
#
#
# def meshNormals(verts, faces, normalise=True):
#     # Calculates normals of mesh faces
#
#     # Calculate norms using cross product and normalisation
#     u = verts[faces[:,0],:] - verts[faces[:,1],:]
#     v = verts[faces[:,1],:] - verts[faces[:,2],:]
#
#     # Calculate cross product and normalise
#     norms = np.cross(u, v, axisa=1, axisb=1)
#     if normalise==True:
#         norms /= np.linalg.norm(norms, ord=None, axis=1)[:,None]
#
#     return norms
#
#
# def ellipsoidArea(semi_axes, n_degree=10):
#     # Approximates area of an ellipsoid using legendre polynomials
#
#     # Let semi_axes[2] be the major semi-axis length, semi_axes[1] the minor semi-axis length and semi_axes[0] the
#     # least axis length.
#
#     # Import legendre evaluation function from numpy
#     from numpy.polynomial.legendre import legval
#
#     # Calculate eccentricities
#     ecc_alpha = np.sqrt(1 - (semi_axes[1] ** 2.0) / (semi_axes[2] ** 2.0))
#     ecc_beta  = np.sqrt(1 - (semi_axes[0] ** 2.0) / (semi_axes[2] ** 2.0))
#
#     # Create a vector to generate coefficients for the Legendre polynomial
#     nu = np.arange(start=0, stop=n_degree + 1, step=1) * 1.0
#
#     # Define Legendre polynomial coefficients and evaluation point
#     leg_coeff = (ecc_alpha * ecc_beta) ** nu / (1.0 - 4.0 * nu ** 2.0)
#     leg_x     = (ecc_alpha ** 2.0 + ecc_beta ** 2.0) / (2.0 * ecc_alpha * ecc_beta)
#
#     # Calculate approximate area
#     area_appr = 4.0 * np.pi * semi_axes[2] * semi_axes[1] * legval(x=leg_x, c=leg_coeff)
#
#     return area_appr
#
#
# def minOrientedBoundBox(pos_mat):
#     # Implementation of Chan and Tan's algorithm (C.K. Chan, S.T. Tan. Determination of the minimum bounding box of an
#     # arbitrary solid: an iterative approach. Comp Struc 79 (2001) 1433-1449
#
#     import copy
#     from scipy.spatial import ConvexHull
#
#     ##########################
#     # Internal functions
#     ##########################
#
#     def calcRotAABBSurface(theta, hull_mat):
#         # Function to calculate surface of the axis-aligned bounding box of a rotated 2D contour
#
#         # Create rotation matrix and rotate over theta
#         rot_mat = rotMatrix(theta=theta, dim=2)
#         rot_hull = np.dot(rot_mat, hull_mat)
#
#         # Calculate bounding box surface of the rotated contour
#         rot_aabb_dims = np.max(rot_hull, axis=1) - np.min(rot_hull, axis=1)
#         rot_aabb_area = np.product(rot_aabb_dims)
#
#         return rot_aabb_area
#
#     def approxMinTheta(hull_mat, theta_sel, res, max_rep=5):
#         # Iterative approximator for finding angle theta that minimises surface area
#         for i in np.arange(0,max_rep):
#
#             # Select new thetas in vicinity of
#             theta     = np.array([theta_sel-res, theta_sel-0.5*res, theta_sel, theta_sel+0.5*res, theta_sel+res])
#
#             # Calculate projection areas for current angles theta
#             rot_area  = np.array(map(lambda x: calcRotAABBSurface(theta=x, hull_mat=hull_mat), theta))
#
#             # Find global minimum and corresponding angle theta_sel
#             theta_sel = theta[np.argmin(rot_area)]
#
#             # Shrink resolution and iterate
#             res = res / 2.0
#
#         return theta_sel
#
#     def rotateMinimalProjection(input_pos, rot_axis, n_minima=3, res_init=5.0):
#         # Function to that rotates input_pos to find the rotation that minimises the projection of input_pos on the
#         # plane normal to the rot_axis
#
#         # Find axis aligned bounding box of the point set
#         aabb_max = np.max(input_pos, axis=0)
#         aabb_min = np.min(input_pos, axis=0)
#
#         # Center the point set at the AABB center
#         output_pos = input_pos - 0.5 * (aabb_min + aabb_max)
#
#         # Project model to plane
#         proj_pos = copy.deepcopy(output_pos)
#         proj_pos = np.delete(proj_pos, [rot_axis], axis=1)
#
#         # Calculate 2D convex hull of the model projection in plane
#         if np.shape(proj_pos)[0] >= 10:
#             hull_2d = ConvexHull(points=proj_pos)
#             hull_mat = proj_pos[hull_2d.vertices, :]
#             del hull_2d, proj_pos
#         else:
#             hull_mat = proj_pos
#             del proj_pos
#
#         # Transpose hull_mat so that the array is (ndim, npoints) instead of (npoints, ndim)
#         hull_mat = np.transpose(hull_mat)
#
#         # Calculate bounding box surface of a series of rotated contours
#         # Note we can program a min-search algorithm here as well
#
#         # Calculate initial surfaces
#         theta_init = np.arange(start=0.0, stop=90.0+res_init, step=res_init) * np.pi / 180.0
#         rot_area   = np.array(map(lambda x: calcRotAABBSurface(theta=x, hull_mat=hull_mat), theta_init))
#
#         # Find local minima
#         df_min     = sigProcFindPeaks(x=rot_area, dir="neg")
#
#         # Check if any minimum was generated
#         if len(df_min) > 0:
#             # Investigate up to n_minima number of local minima, starting with the global minimum
#             df_min = df_min.sort_values(by="val", ascending=True)
#
#             # Determine max number of minima evaluated
#             max_iter = np.min([n_minima, len(df_min)])
#
#             # Initialise place holder array
#             theta_min = np.zeros(max_iter)
#
#             # Iterate over local minima
#             for k in np.arange(0, max_iter):
#
#                 # Find initial angle corresponding to i-th minimum
#                 sel_ind      = df_min.ind.values[k]
#                 theta_curr   = theta_init[sel_ind]
#
#                 # Zoom in to improve the approximation of theta
#                 theta_min[k] = approxMinTheta(hull_mat=hull_mat, theta_sel=theta_curr, res=res_init*np.pi/180.0)
#
#             # Calculate surface areas corresponding to theta_min and theta that minimises the surface
#             rot_area  = np.array(map(lambda x: calcRotAABBSurface(theta=x, hull_mat=hull_mat), theta_min))
#             theta_sel = theta_min[np.argmin(rot_area)]
#
#         else:
#             theta_sel = theta_init[0]
#
#         # Rotate original point along the angle that minimises the projected AABB area
#         output_pos = np.transpose(output_pos)
#         rot_mat = rotMatrix(theta=theta_sel, dim=3, rot_axis=rot_axis)
#         output_pos = np.dot(rot_mat, output_pos)
#
#         # Rotate output_pos back to (npoints, ndim)
#         output_pos = np.transpose(output_pos)
#
#         return output_pos
#
#     ##########################
#     # Main function
#     ##########################
#
#     rot_df = pd.DataFrame({"rot_axis_0":  np.array([0,0,0,0,1,1,1,1,2,2,2,2]),
#                            "rot_axis_1":  np.array([1,2,1,2,0,2,0,2,0,1,0,1]),
#                            "rot_axis_2":  np.array([2,1,0,0,2,0,1,1,1,0,2,2]),
#                            "aabb_axis_0": np.zeros(12),
#                            "aabb_axis_1": np.zeros(12),
#                            "aabb_axis_2": np.zeros(12),
#                            "vol":         np.zeros(12)})
#
#     # Rotate over different sequences
#     for i in np.arange(0, len(rot_df)):
#         # Create a local copy
#         work_pos = copy.deepcopy(pos_mat)
#
#         # Rotate over sequence of rotation axes
#         work_pos = rotateMinimalProjection(input_pos=work_pos, rot_axis=rot_df.rot_axis_0[i])
#         work_pos = rotateMinimalProjection(input_pos=work_pos, rot_axis=rot_df.rot_axis_1[i])
#         work_pos = rotateMinimalProjection(input_pos=work_pos, rot_axis=rot_df.rot_axis_2[i])
#
#         # Determine resultant minimum bounding box
#         aabb_dims = np.max(work_pos, axis=0) - np.min(work_pos, axis=0)
#         rot_df.ix[i, "aabb_axis_0"] = aabb_dims[0]
#         rot_df.ix[i, "aabb_axis_1"] = aabb_dims[1]
#         rot_df.ix[i, "aabb_axis_2"] = aabb_dims[2]
#         rot_df.ix[i, "vol"]         = np.product(aabb_dims)
#
#         del work_pos, aabb_dims
#
#     # Find minimal volume of all rotations and return bounding box dimensions
#     sel_row   = rot_df.ix[np.argmin(rot_df.vol),:]
#     ombb_dims = np.array([sel_row.aabb_axis_0, sel_row.aabb_axis_1, sel_row.aabb_axis_2])
#
#     return ombb_dims
#
#
# def rotMatrix(theta, dim=2, rot_axis=-1):
#     # Creates a 2d or 3d rotation matrix
#     if dim == 2:
#         rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
#                             [np.sin(theta), np.cos(theta)]])
#
#     if dim == 3:
#         if rot_axis == 0:
#             rot_mat = np.array([[1.0, 0.0,           0.0],
#                                 [0.0, np.cos(theta), -np.sin(theta)],
#                                 [0.0, np.sin(theta), np.cos(theta)]])
#         if rot_axis == 1:
#             rot_mat = np.array([[np.cos(theta),  0.0, np.sin(theta)],
#                                 [0.0,            1.0, 0.0],
#                                 [-np.sin(theta), 0.0, np.cos(theta)]])
#         if rot_axis == 2:
#             rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0.0],
#                                 [np.sin(theta), np.cos(theta),  0.0],
#                                 [0.0,           0.0,            1.0]])
#
#     return rot_mat
#
#
# def minVolEnclosingEllipsoid(pos_mat, tolerance=10E-4):
#     # Calculates the semi-axes of the minimum volume enclosing ellipsoid. This algorithm is based on N. Moshtagh,
#     # Minimum volume enclosing ellipsoids (2005), DOI 10.1.1.116.7691
#
#     import copy
#     import numpy.linalg as npla
#
#     # Cast to np.matrix
#     mat_P = copy.deepcopy(pos_mat)
#     mat_P = np.transpose(mat_P)
#     mat_P = np.asmatrix(mat_P)
#
#     # Dimension of the point set
#     ndim = np.shape(mat_P)[0]
#
#     # Number of points in the point set
#     npoint = np.shape(mat_P)[1]
#
#     # Add a row of 1s to the ndim x npoint matrix input_pos - so mat_Q is (ndim+1) x npoint now.
#     mat_Q = np.append(mat_P, np.ones(shape=(1,npoint)), axis=0)
#
#     # Initialise settings for Khachiyan algorithm
#     iter_count = 1
#     err        = 1
#
#     # Initial u vector
#     vec_u = 1.0 * np.ones(shape=npoint) / npoint
#
#     # Khachiyan algorithm
#     while err > tolerance:
#         # Matrix multiplication:
#         # np.diag(u) : if u is a vector, places the elements of u
#         # in the diagonal of an NxN matrix of zeros
#         mat_X = mat_Q * np.diag(vec_u) * np.transpose(mat_Q)
#
#         # npla.inv(mat_X) returns the matrix inverse of mat_X
#         # np.diag(mat_M) when mat_M is a matrix returns the diagonal vector of mat_M
#         vec_M = np.diag(np.transpose(mat_Q) * npla.inv(mat_X) * mat_Q)
#
#         # Find the value and location of the maximum element in the vector M
#         max_M = np.max(vec_M)
#         ind_j   = np.argmax(vec_M)
#
#         # Calculate the step size for the ascent
#         step_size = (max_M - ndim - 1.0) / ((ndim + 1.0) * (max_M - 1.0))
#
#         # Calculate the vector vec_new_u:
#         # Take the vector vec_u, and multiply all the elements in it by (1-step_size)
#         vec_new_u = vec_u * (1.0 - step_size)
#
#         # Increment the jth element of new_u by step_size
#         vec_new_u[ind_j] = vec_new_u[ind_j] + step_size
#
#         # Store the error by taking finding the square root of the SSD
#         # between new_u and u
#         # The SSD or sum-of-square-differences, takes two vectors
#         # of the same size, creates a new vector by finding the
#         # difference between corresponding elements, squaring
#         # each difference and adding them all together.
#         err = npla.norm(vec_new_u - vec_u)
#
#         # Increment iter_count and replace vec_u
#         iter_count += 1
#         vec_u = vec_new_u
#
#     # Put the elements of the vector u into the diagonal of a matrix
#     # U with the rest of the elements as 0
#     mat_U = np.asmatrix(np.diag(vec_u))
#
#     # Compute the A-matrix
#     vec_u = np.transpose(np.asmatrix(vec_u))
#     mat_A = (1.0 / ndim) * npla.inv(mat_P * mat_U * np.transpose(mat_P) - (mat_P * vec_u) * np.transpose(mat_P * vec_u))
#
#     # Compute the center
#     mat_C = mat_P * vec_u
#
#     # Perform singular value decomposition
#     s = npla.svd(mat_A, compute_uv=False)
#
#     # The semi-axis lengths are the inverse square root of the of the singular values
#     semi_axes_length = np.sort(1.0/np.sqrt(s))
#
#     return semi_axes_length
#
#
# def sigProcSegmentise(x):
#     # Produces a list of segments from input x with values (0,1)
#
#     # Create a difference vector
#     y = np.diff(x)
#
#     # Find start and end indices of sections with value 1
#     ind_1_start = np.array(np.where(y==1)).flatten()
#     if np.shape(ind_1_start)[0] > 0: ind_1_start += 1
#     ind_1_end   = np.array(np.where(y==-1)).flatten()
#
#     # Check for boundary effects
#     if x[0]==1:  ind_1_start = np.insert(ind_1_start, 0, 0)
#     if x[-1]==1: ind_1_end   = np.append(ind_1_end, np.shape(x)[0]-1)
#
#     # Generate segment df for segments with value 1
#     if np.shape(ind_1_start)[0]==0:
#         df_one = pd.DataFrame({"i":   [],
#                                "j":   [],
#                                "val": []})
#     else:
#         df_one = pd.DataFrame({"i":   ind_1_start,
#                                "j":   ind_1_end,
#                                "val": np.ones(np.shape(ind_1_start)[0])})
#
#     # Find start and end indices for section with value 0
#     if np.shape(ind_1_start)[0]==0:
#         ind_0_start = np.array([0])
#         ind_0_end   = np.array([np.shape(x)[0]-1])
#     else:
#         ind_0_end   = ind_1_start - 1
#         ind_0_start = ind_1_end   + 1
#
#         # Check for boundary effect
#         if x[0]==0:  ind_0_start = np.insert(ind_0_start, 0, 0)
#         if x[-1]==0: ind_0_end   = np.append(ind_0_end, np.shape(x)[0]-1)
#
#         # Check for out-of-range boundary effects
#         if ind_0_end[0] < 0:                  ind_0_end   = np.delete(ind_0_end, 0)
#         if ind_0_start[-1] >= np.shape(x)[0]: ind_0_start = np.delete(ind_0_start, -1)
#
#     # Generate segment df for segments with value 0
#     if np.shape(ind_0_start)[0]==0:
#         df_zero = pd.DataFrame({"i":   [],
#                                 "j":   [],
#                                 "val": []})
#     else:
#         df_zero = pd.DataFrame({"i":    ind_0_start,
#                                 "j":    ind_0_end,
#                                 "val":  np.zeros(np.shape(ind_0_start)[0])})
#
#     df_segm = df_one.append(df_zero).sort_values(by="i").reset_index(drop=True)
#
#     return df_segm
#
#
# def sigProcFindPeaks(x, dir="pos"):
#     # Determines peak positions in array of values
#
#     # Invert when looking for local minima
#     if dir=="neg": x = -x
#
#     # Generate segments where slope is negative
#
#     df_segm = sigProcSegmentise(x=(np.diff(x) < 0.0)*1)
#
#     # Start of slope coincides with position of peak (due to index shift induced by np.diff)
#     ind_peak = df_segm.loc[df_segm.val==1,"i"].values
#
#     # Check right boundary
#     if x[-1] > x[-2]: ind_peak = np.append(ind_peak, np.shape(x)[0]-1)
#
#     # Construct dataframe with index and corresponding value
#     if np.shape(ind_peak)[0]==0:
#         df_peak = pd.DataFrame({"ind": [],
#                                 "val": []})
#     else:
#         if dir=="pos":
#             df_peak = pd.DataFrame({"ind": ind_peak,
#                                     "val": x[ind_peak]})
#         if dir=="neg":
#             df_peak = pd.DataFrame({"ind":  ind_peak,
#                                     "val": -x[ind_peak]})
#     return df_peak
#
#
# def geospatial(df_int, spacing):
#
#     # Define constants
#     n_v = len(df_int)
#
#     # Generate point cloud
#     pos_mat = df_int.as_matrix(["z", "y", "x"])
#     pos_mat = np.multiply(pos_mat, spacing)
#
#     # Determine all interactions between voxels
#     comb_iter = np.array([np.tile(np.arange(0, n_v), n_v), np.repeat(np.arange(0, n_v), n_v)])
#     comb_iter = comb_iter[:, comb_iter[0, :] > comb_iter[1, :]]
#
#     # Determine weighting for all interactions (inverse weighting with distance)
#     w_ij = 1.0 / np.array(
#         map(lambda i: np.sqrt(np.sum((pos_mat[comb_iter[0, i], :] - pos_mat[comb_iter[1, i], :]) ** 2.0)),
#             np.arange(np.shape(comb_iter)[1])))
#
#     # Create array of mean-corrected grey level intensities
#     gl_dev = df_int.g.values - np.mean(df_int.g)
#
#     # Moran's I
#     nom = n_v * np.sum(np.multiply(np.multiply(w_ij, gl_dev[comb_iter[0, :]]), gl_dev[comb_iter[1, :]]))
#     denom = np.sum(w_ij) * np.sum(gl_dev ** 2.0)
#     moran_i = nom / denom
#
#     # Geary's C
#     nom = (n_v - 1.0) * np.sum(np.multiply(w_ij, (gl_dev[comb_iter[0, :]] - gl_dev[comb_iter[1, :]]) ** 2.0))
#     geary_c = nom / (2.0 * denom)
#
#     return moran_i, geary_c


def contourBased(map_obj, spacing, dims):
    from skimage.measure import marching_cubes_lewiner
    from featureDefinitions import morph_cont_features

    # Close internal volumes
    roi_cld = np.reshape((map_obj.vol_id.values>=0), dims)
    roi_cld = np.pad(roi_cld, pad_width=1, mode="constant", constant_values=0.0)  # Pad 3D roi with non-roi voxels

    # Use marching cubes to generate a mesh grid for the roi
    verts, faces = marching_cubes_lewiner(volume=roi_cld, level=0.5, spacing=spacing)

    # Subtract centroid and convert to matrix
    verts -= np.mean(verts, axis=0)
    verts = np.asmatrix(verts)

    # Rotate vertices
    theta = 0.0
    phi   = 0.0
    verts_rot = verts * rotMatrix(theta=theta, dim=3, rot_axis=0) * rotMatrix(theta=phi, dim=3, rot_axis=1)

    # Generate roi from mesh and find border voxels
    roi       = mesh2grid(verts=verts_rot, faces=faces, spacing=spacing)

    # Calculate features
    df_feat = morph_cont_features(roi_obj=roi, spacing=spacing)


    pass


def getPerimeter(roi_slice, spacing):

    import scipy.ndimage as ndi

    def coord2Index(x, y, dims):
        # Translate coordinates to indices
        conv_index = x + y * dims[1]

        # Mark invalid transitions
        conv_index[np.logical_or(x < 0, x >= dims[1])] = -99999
        conv_index[np.logical_or(y < 0, y >= dims[0])] = -99999

        return conv_index

    def voxelIsBorder(df, index):
        # Determine whether selected voxel is a border voxel

        border_vox = np.zeros(np.size(index), dtype=bool)
        border_vox[index >= 0] = df.border.values[index[index >= 0]]

        return border_vox


    # Find disconnected parts of the roi
    roi_grp_slice, n_grp = ndi.label(roi_slice, ndi.generate_binary_structure(2, 2))

    # Remove the smallest disconnected groups
    if n_grp > 1:
        # Determine group size
        grp_id, grp_size = np.unique(roi_grp_slice, return_counts=True)

        # Remove group 0 (voxels outside roi)
        grp_mask = grp_id > 0
        grp_id = grp_id[grp_mask]
        grp_size = grp_size[grp_mask]

        # Select largest group
        sel_grp = grp_id[np.argmax(grp_size)]

        # Update roi mask
        roi_slice = roi_grp_slice == sel_grp
        del grp_id, grp_size, grp_mask, sel_grp
    del roi_grp_slice, n_grp

    # Determine border voxels
    roi_b_slice = np.logical_xor(roi_slice, ndi.binary_erosion(roi_slice, ndi.generate_binary_structure(2, 1)))
    n_border_voxels = np.sum(roi_b_slice)
    roi_dims    = np.shape(roi_slice)

    # Create data frame
    vox_indices = np.unravel_index(indices=np.arange(start=0, stop=roi_slice.size), shape=np.shape(roi_slice))
    df_slice = pd.DataFrame({"index_id": np.arange(start=0, stop=roi_slice.size),
                             "border":   np.ravel(roi_b_slice),
                             "y_ind":    vox_indices[0],
                             "x_ind":    vox_indices[1],
                             "y_pos":    vox_indices[0] * spacing[0],
                             "x_pos":    vox_indices[1] * spacing[1],
                             "r":        np.zeros(roi_slice.size),
                             "theta":    np.full(roi_slice.size, np.nan),
                             "visited":  np.zeros(roi_slice.size, dtype=bool)})
    del vox_indices

    # Recenter on border centroid
    border_mask = df_slice.border.values

    df_slice.loc[:, ["y_pos", "x_pos"]] -= np.mean(df_slice.loc[border_mask, ["y_pos", "x_pos"]], axis=0)

    # Calculate radius
    df_slice.loc[border_mask, "r"] = np.linalg.norm(df_slice.loc[border_mask, ["y_pos", "x_pos"]], axis=1)

    # Calculate theta angle and wrap to [0, 2pi)
    df_slice.loc[border_mask, "theta"] = np.arctan2(df_slice.y_pos[border_mask], df_slice.x_pos[border_mask])
    df_slice.loc[df_slice.theta < 0, "theta"] += 2.0 * np.pi

    # Initialise neighbours
    df_nbrs = pd.DataFrame({"index_id": np.arange(start=0, stop=8),
                            "y_ind":    np.array([ 0,  1,  1,  1,  0, -1, -1, -1]),
                            "x_ind":    np.array([ 1,  1,  0, -1, -1, -1,  0,  1]),
                            "to_ind":   np.zeros(8, dtype=int),
                            "to_phi":   np.zeros(8, dtype=float),
                            "border":   np.zeros(8, dtype=bool),
                            "pref":     np.zeros(8, dtype=float)})

    # Set angle phi for the particular neighbour direction and wrap to[-pi,pi)
    df_nbrs["to_phi"] = np.arctan2(df_nbrs.y_ind, df_nbrs.x_ind)
    df_nbrs.loc[df_nbrs.to_phi >= np.pi, "to_phi"] -= 2.0 * np.pi

    # Initialise perimeter chain
    perim_chain = np.full(2*n_border_voxels, np.nan)
    perim_iter  = 1

    # Select first border voxel (theta closest to 0)
    sel_voxel = df_slice.index_id.values[np.argmin(np.abs(df_slice.theta))]

    # Set flag for visited voxel
    df_slice.loc[sel_voxel, "visited"] = True

    # Update chain
    perim_chain[0] = sel_voxel

    # Initialise angle for previous jump; in this case we assume clockwise perimeter tracing, so we start facing downward
    sel_phi = -0.5 * np.pi

    # Iterate until all border voxels have been visited
    while (perim_iter < 2 * n_border_voxels) and (np.sum(df_slice.visited) < n_border_voxels):
        # Calculate indices of neighbouring voxels
        df_nbrs["to_ind"] = coord2Index(y=df_nbrs.y_ind.values + df_slice.y_ind.values[sel_voxel],
                                        x=df_nbrs.x_ind.values + df_slice.x_ind.values[sel_voxel],
                                        dims=roi_dims)

        # Update preferential directions for previously selected jump angle
        df_nbrs["pref"] = df_nbrs.to_phi - sel_phi
        df_nbrs.loc[df_nbrs.pref >= np.pi, "pref"] -= 2.0 * np.pi

        # Find neighbouring voxels that are part of the border
        df_nbrs["border"] = voxelIsBorder(df=df_slice, index=df_nbrs.to_ind.values)

        # Determine next jump (highest pref score over all border voxels)
        df_nbrs_valid = df_nbrs.loc[df_nbrs.border == True, ]
        sel_ind_id    = df_nbrs_valid.index_id.values[np.argmax(df_nbrs_valid.pref.values)]
        sel_voxel     = df_nbrs.to_ind.values[sel_ind_id]
        sel_phi       = df_nbrs.to_phi.values[sel_ind_id]

        # Set flag for visited voxel
        df_slice.loc[sel_voxel, "visited"] = True

        # Update chain
        perim_chain[perim_iter] = sel_voxel
        perim_iter += 1

    # Remove redundant elements from the perimeter chain
    perim_chain = perim_chain[~np.isnan(perim_chain)].astype(int)

    # Define return data frame
    df_ret = df_slice.iloc[perim_chain,].reset_index(drop=True).drop(["index_id","border","visited"], axis=1)

    return df_ret


def pointInTriangle(point_P, vertex_A, vertex_B, vertex_C):
    # Implementation of the barycentric technique for determining whether a point P is in a triangle face with vertices A, B and C
    # See http://blackpawn.com/texts/pointinpoly/

    # Compute vectors
    v_0 = vertex_C - vertex_A
    v_1 = vertex_B - vertex_A
    v_2 = point_P - vertex_A

    # Compute dot products
    dot_00 = np.dot(v_0, v_0)
    dot_01 = np.dot(v_0, v_1)
    dot_02 = np.dot(v_0, v_2)
    dot_11 = np.dot(v_1, v_1)
    dot_12 = np.dot(v_1, v_2)

    # Compute barycentric coordinates
    inv_denom = 1.0 / (dot_00 * dot_11 - dot_01 * dot_01)
    u = (dot_11 * dot_02 - dot_01 * dot_12) * inv_denom
    v = (dot_00 * dot_12 - dot_01 * dot_02) * inv_denom

    # Check whether point is in triangle
    return (u >= 0.0) and (v >= 0.0) and (u + v < 1.0)


def ray_line_intersect(ray_orig, ray_dir, vert_1, vert_2):

    epsilon = 0.000001

    # Define edge
    edge_line = vert_1 - vert_2

    # Define ray vertices
    r_vert_1 = ray_orig
    r_vert_2 = ray_orig + ray_dir
    edge_ray = - ray_dir

    # Calculate determinant - if close to 0, lines are parallel and will not intersect
    det = np.cross(edge_ray, edge_line)
    if (det > -epsilon) and (det < epsilon): return np.nan

    # Calculate inverse of the determinant
    inv_det = 1.0 / det

    # Calculate determinant
    a11 = np.cross(r_vert_1, r_vert_2)
    a21 = np.cross(vert_1,   vert_2)

    # Solve for x
    a12 = edge_ray[0]
    a22 = edge_line[0]
    x = np.linalg.det(np.array([[a11, a12], [a21, a22]])) * inv_det

    # Solve for y
    b12 = edge_ray[1]
    b22 = edge_line[1]
    y = np.linalg.det(np.array([[a11, b12], [a21, b22]])) * inv_det

    t = np.array([x, y])

    # Check whether the solution falls within the line segment
    u1 = np.around(np.dot(edge_line, edge_line), 5)
    u2 = np.around(np.dot(edge_line, vert_1-t), 5)
    if (u2 / u1) < 0.0 or (u2 / u1) > 1.0: return np.nan

    # Return scalar length from ray origin
    t_scal = np.linalg.norm(ray_orig-t)
    return t_scal


def rayTriangleIntersection(ray_orig, ray_dir, vert_1, vert_2, vert_3):
    # Implementation of the Moeller-Trumbore intersection algorithm to determine intersection point between ray and triangle
    # This point satisfies ray_orig + t * ray_dir = (1 - u -v) * vert_1 + u * vert_2 + v * vert_3
    # See DOI: 10.1145/1198555.1198746
    # Back facing triangles are allowed

    epsilon = 0.000001

    # Define triangle edges
    edge_1 = vert_2 - vert_1
    edge_2 = vert_3 - vert_1

    # Calculate determinant
    p_vec = np.cross(ray_dir, edge_2)
    det   = np.dot(edge_1, p_vec)

    # If determinant is near zero, the ray lies in the plane of the triangle, or is parallel to the triangle
    if (det > -epsilon) and (det < epsilon): return np.nan

    # Calculate inverse of the determinant
    inv_det = 1.0 / det

    # Calculate displacement vector between ray_origin and vert_1
    t_vec = ray_orig - vert_1

    # Calculate scalar parameter u and test bound
    u_scal = np.dot(t_vec, p_vec) * inv_det
    if (u_scal < 0.0) or (u_scal > 1.0): return np.nan

    # Calculate scalar parameter v and test bound
    q_vec = np.cross(t_vec, edge_1)
    v_scal = np.dot(ray_dir, q_vec) * inv_det
    if (v_scal < 0.0) or (u_scal + v_scal > 1.0): return np.nan

    # Calculate scalar parameter t
    t_scal = np.dot(edge_2, q_vec) * inv_det
    if t_scal > epsilon: return t_scal
    else:                return np.nan


def mesh2grid(verts, faces, spacing, origin=None):

    # Remove superfluous dimensions
    verts = np.squeeze(np.asarray(verts))

    # Determine triangle vertices
    vert_1 = verts[faces[:, 0], :]
    vert_2 = verts[faces[:, 1], :]
    vert_3 = verts[faces[:, 2], :]

    # Determine AABB
    min_pos = np.min(verts, axis=0)
    max_pos = np.max(verts, axis=0)

    # Set origin
    if origin is None:
        origin = min_pos - 0.25 * spacing

    # Determine voxel grid
    vox_grid_dims = np.ceil((max_pos - origin) / spacing).astype(int)
    vox_grid = np.zeros(vox_grid_dims, dtype=int)

    # Set ray direction
    ray_dir = np.array((1.0, 0.0, 0.0))
    vox_span = origin[0] + np.arange(0, vox_grid_dims[0]) * spacing[0]

    # Iterate over voxels in x and y plane
    for y_ind in np.arange(0, vox_grid_dims[1]):
        for x_ind in np.arange(0, vox_grid_dims[2]):

            # Set ray origin
            ray_origin = np.array((origin[0] - 1.0,
                                   origin[1] + y_ind * spacing[1],
                                   origin[2] + x_ind * spacing[2]))

            vox_col = np.zeros(np.shape(vox_span), dtype=int)

            # Set mask to test only likely mesh triangles (in-plane mesh coordinates contain in-plane ray coordinates)
            simplex_mask = np.logical_and(np.abs(np.sum(np.sign(np.vstack((vert_1[:, 1], vert_2[:, 1], vert_3[:, 1])) - ray_origin[1]), axis=0)) < 3,
                                          np.abs(np.sum(np.sign(np.vstack((vert_1[:, 2], vert_2[:, 2], vert_3[:, 2])) - ray_origin[2]), axis=0)) < 3)

            # Go to next iterator if mask is empty
            if np.sum(simplex_mask) == 0:   continue

            # Find scalar parameter t
            t_scal = np.array(
                list(map(lambda i_sel: rayTriangleIntersection(ray_orig=ray_origin, ray_dir=ray_dir, vert_1=vert_1[i_sel, :],
                                                               vert_2=vert_2[i_sel, :], vert_3=vert_3[i_sel, :]),
                    np.squeeze(np.where(simplex_mask)))))

            # Remove invalid and redundant results
            t_scal = np.unique(t_scal[np.isfinite(t_scal)])
            if t_scal.size == 0: continue

            # Update vox_col based on t_scal
            for t_curr in t_scal: vox_col[vox_span > t_curr + ray_origin[0]] += 1

            # Voxels in the roi cross an uneven number of meshes from the origin
            vox_grid[:, y_ind, x_ind] += vox_col % 2

    return vox_grid.astype(dtype=bool)


def poly2grid(verts, lines, spacing, origin, shape):

    # Set up line vertices
    vert_1 = verts[lines[:, 0], :]
    vert_2 = verts[lines[:, 1], :]

    # Remove lines with length 0 and center on origin
    line_mask = np.sum(np.abs(vert_1 - vert_2), axis=1) > 0.0
    vert_1 = vert_1[line_mask] - origin
    vert_2 = vert_2[line_mask] - origin

    # Find extent of contours in x
    x_min_ind = int(np.max([np.floor(np.min(verts[:, 1]) / spacing[1]), 0.0]))
    x_max_ind = int(np.min([np.ceil(np.max(verts[:, 1]) / spacing[1]), shape[1] * 1.0]))

    # Set up voxel grid and y-span
    vox_grid = np.zeros(shape, dtype=int)
    vox_span = origin[0] + np.arange(0, shape[0]) * spacing[0]

    # Set ray origin and direction (starts at negative y, and travels towards positive y
    ray_origin = np.array([-1.0, 0.0])
    ray_dir    = np.array([1.0, 0.0])

    for x_ind in np.arange(x_min_ind, x_max_ind):
        # Update ray origin
        ray_origin[1] = origin[1] + x_ind * spacing[1]

        # Scan both forward and backward to resolve points located on the polygon
        vox_col_frwd = np.zeros(np.shape(vox_span), dtype=int)
        vox_col_bkwd = np.zeros(np.shape(vox_span), dtype=int)

        # Find lines that are intersected by the ray
        ray_hit = np.sum(np.sign(np.vstack((vert_1[:, 1], vert_2[:, 1])) - ray_origin[1]), axis=0)

        # If the ray crosses a vertex, the sum of the sign is 0 when the ray does not hit an vertex point, and -1 or 1 when it does.
        # In the latter case, we only keep of the vertices for each hit.
        simplex_mask = np.logical_or(ray_hit == 0, ray_hit == 1)

        # Go to next iterator if mask is empty
        if np.sum(simplex_mask) == 0:   continue

        # Determine the selected vertices
        selected_verts = np.squeeze(np.where(simplex_mask))

        # Find intercept
        t_scal = np.array([ray_line_intersect(ray_orig=ray_origin, ray_dir=ray_dir, vert_1=vert_1[ii, :], vert_2=vert_2[ii, :]) for ii in selected_verts])

        # Remove invalid results
        t_scal = t_scal[np.isfinite(t_scal)]
        if t_scal.size == 0: continue

        # Update vox_col based on t_scal
        for t_curr in t_scal: vox_col_frwd[vox_span > t_curr + ray_origin[0]] += 1
        for t_curr in t_scal: vox_col_bkwd[vox_span < t_curr + ray_origin[0]] += 1

        # Voxels in the roi cross an uneven number of meshes from the origin
        vox_grid[:, x_ind] += np.logical_and(vox_col_frwd % 2, vox_col_bkwd % 2)

    return vox_grid.astype(dtype=bool)
