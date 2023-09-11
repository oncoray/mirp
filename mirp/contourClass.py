import copy

import numpy as np

from mirp.importData.imageGenericFile import ImageFile
from typing import Union, List


class ContourClass:

    def __init__(self, contour: np.ndarray):
        # Convert contour to internal vertices, i.e. from (x, y, z) to (z, y, x)
        vertices = np.zeros(contour.shape, dtype=np.float64)
        vertices[:, 0] = contour[:, 2]
        vertices[:, 1] = contour[:, 1]
        vertices[:, 2] = contour[:, 0]

        self.contour = [vertices]
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

    def which_slice(self) -> List[int]:

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
            slice_id: Union[None, int] = None):

        # This function helps flatten nested contours.
        def _contour_extractor(contour_object_list, slice_id: Union[None, int]):
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
        from mirp.morphologyUtilities import poly2grid

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
