import numpy as np

from mirp.imageClass import ImageClass


class ContourClass:

    def __init__(self, contour, sop_instance_uid):
        # Convert contour to internal vertices, i.e. from (x, y, z) to (z, y, x)
        vertices = np.zeros(contour.shape, dtype=np.float64)
        vertices[:, 0] = contour[:, 2]
        vertices[:, 1] = contour[:, 1]
        vertices[:, 2] = contour[:, 0]

        self.contour=vertices
        self.sop_instance_uid = sop_instance_uid

    def contour_to_grid_ray_cast(self, img_obj):
        from mirp.morphologyUtilities import poly2grid

        # Convert contours to voxel space
        contour_vox = img_obj.to_voxel_coordinates(np.transpose(self.contour))
        contour_vox = np.transpose(contour_vox)

        # Reduce numerical issues by rounding precision
        contour_vox[:, 0] = np.rint(contour_vox[:, 0])
        contour_vox[:, (1, 2)] = np.around(contour_vox[:, (1, 2)], decimals=5)

        # Set contour slices
        contour_slice = np.unique(contour_vox[:, 0])

        # Remove contour slices that lie outside the [0, size[0]] range.
        # contour_slice = [current_contour_slice for current_contour_slice in contour_slice if
        #                  0 <= current_contour_slice < img_obj.size[0]]

        if len(contour_slice) == 0:
            return None, None

        # Initiate a slice list and a mask list
        slice_list = []
        mask_list = []

        # Iterate over slices
        for curr_slice in contour_slice:

            # Select vertices and lines within the current slice
            vertices = contour_vox[contour_vox[:, 0] == curr_slice, :][:, (1, 2)]
            lines = np.vstack(([np.arange(0, vertices.shape[0])], [np.arange(-1, vertices.shape[0] - 1)])).transpose()

            slice_list.append(curr_slice)
            mask_list.append(poly2grid(verts=vertices,
                                       lines=lines,
                                       spacing=np.array([1.0, 1.0]),
                                       origin=np.array([0.0, 0.0]),
                                       shape=np.array([img_obj.size[1], img_obj.size[2]])))

        return slice_list, mask_list
