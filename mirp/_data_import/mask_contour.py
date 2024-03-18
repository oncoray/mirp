import copy
import numpy as np

from mirp._data_import.generic_file import ImageFile


class ContourClass:

    def __init__(self, contour: np.ndarray, reference_sop_uid: str | None):
        # Convert contour to internal vertices, i.e. from (x, y, z) to (z, y, x)
        vertices = np.zeros(contour.shape, dtype=np.float64)
        vertices[:, 0] = contour[:, 2]
        vertices[:, 1] = contour[:, 1]
        vertices[:, 2] = contour[:, 0]

        self.contour = [vertices]
        self.reference_sop_uid = reference_sop_uid
        self.representation = "world"

    def copy(self):
        return copy.deepcopy(self)

    def to_voxel_coordinates(self, image: ImageFile):

        if self.representation == "voxel":
            return self

        self.contour = [
            np.transpose(image.to_voxel_coordinates(np.transpose(contour)))
            for contour in self.contour
            if contour is not None
        ]

        self.representation = "voxel"

        return self

    def to_world_coordinates(self, image: ImageFile):

        if self.representation == "world":
            return self

        self.contour = [
            np.transpose(image.to_world_coordinates(np.transpose(contour)))
            for contour in self.contour
            if contour is not None
        ]

        self.representation = "world"

        return self

    def which_slice(self) -> list[int]:

        if not self.representation == "voxel":
            raise ValueError("Contours should be represented in voxel space, not world space. Contact the devs.")

        slice_id = list(np.unique(np.concatenate([
            np.rint(contour[:, 0]).astype(int)
            for contour in self.contour
            if contour is not None
        ])))

        return slice_id

    def merge(
            self,
            other_contours=None,
            slice_id: None | int = None):

        # This function helps flatten nested contours.
        def _contour_extractor(contour_object_list, slice_id: None | int):
            new_contour_list = []
            for contour_object in contour_object_list:
                for contour in contour_object.contour:
                    if contour is not None:
                        if slice_id is not None:
                            # In case a certain slice is selected, only keep contour data pertaining to that slice.
                            contour = contour[np.rint(contour[:, 0]) == float(slice_id), :]

                        new_contour_list += [contour]

            return new_contour_list

        new_contour = self.copy()

        if other_contours is None:
            new_contour.contour = _contour_extractor([new_contour], slice_id=slice_id)
        else:
            new_contour.contour = _contour_extractor([new_contour] + other_contours, slice_id=slice_id)

        return new_contour

    def contour_to_grid_ray_cast(self, image: ImageFile):

        if not self.representation == "voxel":
            raise ValueError("Contours should be represented in voxel space, not world space. Contact the devs.")

        if self.contour is None:
            return None, None

        contour_slice_id = self.which_slice()
        if len(contour_slice_id) == 0:
            return None, None

        # Initiate a slice list and a mask list
        slice_list = []
        mask_list = []

        for current_slice_id in contour_slice_id:
            vertices = []
            lines = []
            vertex_offset = 0

            # Extract vertices and lines across contours.
            for contour in self.contour:
                current_vertices = contour[np.rint(contour[:, 0]) == float(current_slice_id), :][:, (1, 2)]
                if len(current_vertices) == 0:
                    continue

                # Reduce numerical issues by rounding of vertex positions.
                vertices += [np.around(current_vertices, decimals=5)]

                current_lines = np.vstack((
                    [np.arange(0, current_vertices.shape[0])],
                    [np.arange(-1, current_vertices.shape[0] - 1)])
                ).transpose()

                # Ensure that the lines are self-referential within each contour, i.e. -1 should never point to the last
                # vertex, because that vertex may not belong to the same contour.
                current_lines[0, 1] = current_vertices.shape[0] - 1
                current_lines += vertex_offset

                lines += [current_lines]

                # Update vertex_offset to account for each set of contours.
                vertex_offset += current_vertices.shape[0]

            # Merge vertices and lines
            vertices = np.concatenate(vertices, axis=0)
            lines = np.concatenate(lines, axis=0)

            slice_list.append(current_slice_id)
            mask_list.append(poly2grid(
                verts=vertices,
                lines=lines,
                spacing=np.array([1.0, 1.0]),
                origin=np.array([0.0, 0.0]),
                shape=np.array([image.image_dimension[1], image.image_dimension[2]])
            ))

        return slice_list, mask_list


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
            if np.sum(simplex_mask) == 0:
                continue

            # Find scalar parameter t
            t_scal = np.array(
                list(map(lambda i_sel: ray_triangle_intersect(ray_orig=ray_origin, ray_dir=ray_dir, vert_1=vert_1[i_sel, :],
                                                              vert_2=vert_2[i_sel, :], vert_3=vert_3[i_sel, :]),
                         np.squeeze(np.where(simplex_mask)))))

            # Remove invalid and redundant results
            t_scal = np.unique(t_scal[np.isfinite(t_scal)])
            if t_scal.size == 0:
                continue

            # Update vox_col based on t_scal
            for t_curr in t_scal:
                vox_col[vox_span > t_curr + ray_origin[0]] += 1

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
    ray_dir = np.array([1.0, 0.0])

    for x_ind in np.arange(x_min_ind, x_max_ind):
        # Update ray origin
        ray_origin[1] = origin[1] + x_ind * spacing[1]

        # Scan both forward and backward to resolve points located on the polygon
        vox_col_frwd = np.zeros(np.shape(vox_span), dtype=int)
        vox_col_bkwd = np.zeros(np.shape(vox_span), dtype=int)

        # Find lines that are intersected by the ray
        ray_hit = np.sum(np.sign(np.vstack((vert_1[:, 1], vert_2[:, 1])) - ray_origin[1]), axis=0)

        # If the ray crosses a vertex, the sum of the sign is 0 when the ray does not hit a vertex point, and -1 or 1 when it does.
        # In the latter case, we only keep of the vertices for each hit.
        simplex_mask = np.logical_or(ray_hit == 0, ray_hit == 1)

        # Go to next iterator if mask is empty
        if np.sum(simplex_mask) == 0:
            continue

        # Determine the selected vertices
        selected_verts = np.squeeze(np.where(simplex_mask))

        # Find intercept
        t_scal = np.array([ray_line_intersect(ray_orig=ray_origin, ray_dir=ray_dir, vert_1=vert_1[ii, :], vert_2=vert_2[ii, :]) for ii in selected_verts])

        # Remove invalid results
        t_scal = t_scal[np.isfinite(t_scal)]
        if t_scal.size == 0:
            continue

        # Update vox_col based on t_scal
        for t_curr in t_scal:
            vox_col_frwd[vox_span > t_curr + ray_origin[0]] += 1
        for t_curr in t_scal:
            vox_col_bkwd[vox_span < t_curr + ray_origin[0]] += 1

        # Voxels in the roi cross an uneven number of meshes from the origin
        vox_grid[:, x_ind] += np.logical_and(vox_col_frwd % 2, vox_col_bkwd % 2)

    return vox_grid.astype(dtype=bool)


def ray_triangle_intersect(ray_orig, ray_dir, vert_1, vert_2, vert_3):
    # Implementation of the Moeller-Trumbore intersection algorithm to determine intersection point
    # between ray and triangle
    # This point satisfies ray_orig + t * ray_dir = (1 - u -v) * vert_1 + u * vert_2 + v * vert_3
    # See DOI: 10.1145/1198555.1198746
    # Back facing triangles are allowed

    epsilon = 0.000001

    # Define triangle edges
    edge_1 = vert_2 - vert_1
    edge_2 = vert_3 - vert_1

    # Calculate determinant
    p_vec = np.cross(ray_dir, edge_2)
    det = np.dot(edge_1, p_vec)

    # If determinant is near zero, the ray lies in the plane of the triangle, or is parallel to the triangle
    if (det > -epsilon) and (det < epsilon):
        return np.nan

    # Calculate inverse of the determinant
    inv_det = 1.0 / det

    # Calculate displacement vector between ray_origin and vert_1
    t_vec = ray_orig - vert_1

    # Calculate scalar parameter u and test bound
    u_scal = np.dot(t_vec, p_vec) * inv_det
    if (u_scal < 0.0) or (u_scal > 1.0):
        return np.nan

    # Calculate scalar parameter v and test bound
    q_vec = np.cross(t_vec, edge_1)
    v_scal = np.dot(ray_dir, q_vec) * inv_det
    if (v_scal < 0.0) or (u_scal + v_scal > 1.0):
        return np.nan

    # Calculate scalar parameter t
    t_scal = np.dot(edge_2, q_vec) * inv_det
    if t_scal > epsilon:
        return t_scal
    else:
        return np.nan


def ray_line_intersect(
        ray_orig: np.ndarray,
        ray_dir: np.ndarray,
        vert_1: np.ndarray,
        vert_2: np.ndarray
):

    epsilon = 0.000001

    # Define edge
    edge_line = vert_1 - vert_2

    # Define ray vertices
    r_vert_1 = ray_orig
    r_vert_2 = ray_orig + ray_dir
    edge_ray = - ray_dir

    # Calculate determinant - if close to 0, lines are parallel and will not intersect
    det = np.cross(edge_ray, edge_line)
    if (det > -epsilon) and (det < epsilon):
        return np.nan

    # Calculate inverse of the determinant
    inv_det = 1.0 / det

    # Calculate determinant
    a11 = np.cross(r_vert_1, r_vert_2)
    a21 = np.cross(vert_1, vert_2)

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
    if (u2 / u1) < 0.0 or (u2 / u1) > 1.0:
        return np.nan

    # Return scalar length from ray origin
    t_scal = np.linalg.norm(ray_orig-t)
    return t_scal
