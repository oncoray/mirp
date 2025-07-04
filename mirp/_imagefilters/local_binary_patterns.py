from typing import Generator

import numpy as np
import copy

from mirp._images.generic_image import GenericImage
from mirp._images.transformed_image import LocalBinaryPatternImage
from mirp.settings.generic import SettingsClass
from mirp._imagefilters.generic import GenericFilter


class LocalBinaryPatternFilter(GenericFilter):

    def __init__(self, image: GenericImage, settings: SettingsClass, name: str):

        super().__init__(image=image, settings=settings, name=name)

        self.ibsi_compliant = False

        self.separate_slices = settings.img_transform.lbp_separate_slices
        self.lbp_method = settings.img_transform.lbp_method
        self.d = settings.img_transform.lbp_distance

    def generate_object(self):
        # Generator for transformation objects.
        lbp_method = copy.deepcopy(self.lbp_method)
        if not isinstance(lbp_method, list):
            lbp_method = [lbp_method]

        separate_slices = copy.deepcopy(self.separate_slices)
        if not isinstance(separate_slices, list):
            separate_slices = [separate_slices]

        distance = copy.deepcopy(self.d)
        if not isinstance(distance, list):
            distance = [distance]

        # Iterate over options to yield filter objects with specific settings. A copy of the parent object is made to
        # avoid updating by reference.
        for current_separate_slices in separate_slices:
            for current_lbp_method in lbp_method:
                for current_distance in distance:
                    filter_object = copy.deepcopy(self)
                    filter_object.separate_slices = current_separate_slices
                    filter_object.lbp_method = current_lbp_method
                    filter_object.d = current_distance

                    yield filter_object

    def transform(self, image: GenericImage) -> LocalBinaryPatternImage:
        # Create placeholder Laplacian-of-Gaussian response map.
        response_map = LocalBinaryPatternImage(
            image_data=None,
            separate_slices=self.separate_slices,
            distance=self.d,
            lbp_method = self.lbp_method,
            template=image
        )
        response_map.ibsi_compliant = self.ibsi_compliant and image.ibsi_compliant

        if image.is_empty():
            return response_map

        response_map.set_voxel_grid(
            voxel_grid=self.transform_grid(voxel_grid=image.get_voxel_grid())
        )

        return response_map

    def transform_grid(
            self,
            voxel_grid: np.ndarray
    ):
        # Voxel grid as a contiguous flattened array.
        dims = voxel_grid.shape
        voxel_original = np.ravel(voxel_grid)

        # # Get directions corresponding to distance (d) and separate_slices. For each direction, determine the
        # position of voxels with a valid neighbour (i.e. not out-of-volume) and the
        # position of their neighbour.
        neighbour_vectors = list(self._generate_neighbour_direction())
        weights = 2 ** np.arange(len(neighbour_vectors))

        # Initialise response map as a flattened array.
        lbp = np.zeros((len(neighbour_vectors), len(voxel_original)), dtype = bool)
        for ii, neighbour_vector in enumerate(neighbour_vectors):
            mask, voxel_neighbour = self._lookup_neighbour_voxel_value(
                voxels=voxel_original,
                dims=dims,
                lookup_vector=neighbour_vector
            )
            lbp[ii, mask] = voxel_neighbour - voxel_original[mask] >= 0.0

        if self.lbp_method == "default":
            voxel_response = np.sum(np.multiply(lbp, weights[:, np.newaxis]), axis = 0)

        elif self.lbp_method == "variance":
            voxel_response = np.var(lbp, axis = 0)

        elif self.lbp_method == "rotation_invariant":
            voxel_response = np.sum(np.multiply(lbp, weights[:, np.newaxis]), axis=0)

            for ii in np.arange(1, len(neighbour_vectors)):
                lbp = np.roll(lbp, 1, axis = 0)
                voxel_response = np.min(np.vstack((
                    voxel_response,
                    np.sum(np.multiply(lbp, weights[:, np.newaxis]), axis=0)
                )), axis =0)

        else:
            raise ValueError(f"Unknown method: {self.lbp_method}")

        return np.reshape(voxel_response, shape = dims)

    @staticmethod
    def _coord_to_index(z, y, x, dims):
        # Translate coordinates to indices
        index = x + y * dims[2] + z * dims[2] * dims[1]

        # Mark invalid transitions
        index[np.logical_or(x < 0, x >= dims[2])] = -99999
        index[np.logical_or(y < 0, y >= dims[1])] = -99999
        index[np.logical_or(z < 0, z >= dims[0])] = -99999

        return index

    @staticmethod
    def _index_to_coord(index, dims):
        z = index // (dims[2] * dims[1])
        index -= z * (dims[2] * dims[1])
        y = index // (dims[2])
        x = index - y * dims[2]

        return z, y, x

    def _lookup_neighbour_voxel_value(self, voxels, dims, lookup_vector):
        z, y, x = self._index_to_coord(index=np.arange(len(voxels)), dims=dims)
        neighbour_index = self._coord_to_index(
            z = z + lookup_vector[0],
            y = y + lookup_vector[1],
            x = x + lookup_vector[2],
            dims = dims
        )

        mask = neighbour_index > 0
        return mask, voxels[neighbour_index[mask]]

    def _generate_neighbour_direction(self) -> Generator[tuple[int, ...], None, None]:
        from mirp._features.utilities import rep

        if self.separate_slices:
            m = 8
            nbrs = np.array([
                np.zeros(m, int),
                np.round(self.d * np.sin(2 * np.pi * np.arange(m, dtype=float) / m)),
                np.round(self.d * np.cos(2 * np.pi * np.arange(m, dtype=float) / m))
            ], dtype = int)

            # Remove duplicates
            _, indices = np.unique(nbrs, return_index=True, axis=1)
            nbrs = nbrs[:, indices.sort()].squeeze()

            # Compute distance to eliminate
            neighbour_distance = np.sqrt(np.sum(np.multiply(nbrs, nbrs), axis=0))
            index = neighbour_distance > 0.0

            for ii, flag in enumerate(index):
                if flag:
                    yield tuple(nbrs[:, ii].flatten())

        else:
            # Base transition vector
            trans = np.arange(start=-np.ceil(self.d + 1.0), stop=np.ceil(self.d + 1.0) + 1)
            n = np.size(trans)

            # Build transition array [z,y,x]
            nbrs = np.array([
                rep(x=trans, each=n * n, times=1),
                rep(x=trans, each=n, times=n, use_inversion=True),
                rep(x=trans, each=1, times=n * n, use_inversion=True)
            ], dtype=int)

            # Filter neighbours based on distance. That is, all voxels that fall within distance d and d-1.0 (a single
            # rim of voxels), and excluding the central voxel.
            neighbour_distance = np.sqrt(np.sum(np.multiply(nbrs, nbrs), axis = 0))
            index = np.logical_and(neighbour_distance <= self.d, neighbour_distance > self.d - 1.0)
            index = np.logical_and(index, neighbour_distance > 0.0)

            for ii, flag in enumerate(index):
                if flag:
                    yield tuple(nbrs[:, ii].flatten())
