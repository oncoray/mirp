import numpy as np

from mirp.imageClass import ImageClass

class ContourClass:

    def __init__(self, contour):
        # Convert contour to internal vertices, i.e. from (x, y, z) to (z, y, x)
        vertices = np.zeros(contour.shape, dtype=np.float64)
        vertices[:, 0] = contour[:, 2]
        vertices[:, 1] = contour[:, 1]
        vertices[:, 2] = contour[:, 0]

        self.contour=vertices

    def contour_to_image_space(self, img_obj: ImageClass):
        # Transforms patient reference back to image space. Uses the 0x0020, 0x0037 tag
        spatial_affine = np.ones((3), dtype=np.float)

        # z-coordinates
        spatial_affine[0] = np.inner(img_obj.spacing, [img_obj.orientation[0], img_obj.orientation[3], img_obj.orientation[6]])

        # y-coordinates
        spatial_affine[1] = np.inner(img_obj.spacing, [img_obj.orientation[1], img_obj.orientation[4], img_obj.orientation[7]])

        # x-coordinates
        spatial_affine[2] = np.inner(img_obj.spacing, [img_obj.orientation[2], img_obj.orientation[5], img_obj.orientation[8]])

        contour_vox = np.divide(self.contour - img_obj.origin, spatial_affine)

        return contour_vox

    def contour_to_grid_ray_cast(self, img_obj):
        from mirp.morphologyUtilities import poly2grid

        # Convert contours to voxel space
        contour_vox = self.contour_to_image_space(img_obj=img_obj)

        # Reduce numerical issues by rounding precision
        contour_vox[:, 0] = np.rint(contour_vox[:, 0])
        contour_vox[:, (1, 2)] = np.around(contour_vox[:, (1, 2)], decimals=5)

        # Set contour slices
        contour_slice = np.unique(contour_vox[:, 0])

        # Initiate a slice list and a mask list
        slice_list = []
        mask_list = []

        # Iterate over slices
        for curr_slice in contour_slice:

            # Select vertices and lines within the current slice
            vertices = contour_vox[contour_vox[:, 0] == curr_slice, :][:, (1, 2)]
            lines = np.vstack(([np.arange(0, vertices.shape[0])], [np.arange(-1, vertices.shape[0] - 1)])).transpose()

            slice_list.append(np.int(curr_slice))
            mask_list.append(poly2grid(verts=vertices, lines=lines, spacing=np.array([1.0, 1.0]), origin=np.array([0.0, 0.0]),
                                       shape=np.array([img_obj.size[1], img_obj.size[2]])))

        return slice_list, mask_list
