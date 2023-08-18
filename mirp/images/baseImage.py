import numpy as np
from typing import Union, Optional, List, Tuple


class BaseImage:

    def __init__(
            self,
            image_modality: Optional[str] = None,
            image_origin: Optional[Tuple[float]] = None,
            image_orientation: Optional[np.ndarray] = None,
            image_spacing: Optional[Tuple[float]] = None,
            image_dimensions: Optional[Tuple[int]] = None,
            **kwargs
    ):
        # Make cooperative
        super().__init__()

        # Set modality.
        if isinstance(image_modality, str):
            image_modality = image_modality.lower()
        self.modality: Union[None, str] = image_modality

        # Set affine-transformation related values.
        self.image_origin = image_origin
        self.image_orientation = image_orientation
        self.image_spacing = image_spacing
        self.image_dimension = image_dimensions

    def get_slice_position(self) -> np.ndarray:
        if self.image_spacing is None:
            raise ValueError("The image_spacing attribute has not been set.")
        if self.image_dimension is None:
            raise ValueError("The image_dimension attribute has not been set.")

        slice_distance = self.image_spacing[0]
        slice_positions = [0.0] + [slice_distance] * (self.image_dimension[0] - 1)
        return np.cumsum(slice_positions)

    def world_coordinates(self):

        # Create grid.
        voxel_coordinate_z, voxel_coordinate_y, voxel_coordinate_x = np.mgrid[
            :self.image_dimension[0],
            :self.image_dimension[1],
            :self.image_dimension[2]
        ]

        # Set voxel coordinates.
        voxel_coordinates = np.array([
            voxel_coordinate_z.flatten().astype(float),
            voxel_coordinate_y.flatten().astype(float),
            voxel_coordinate_x.flatten().astype(float)])

        # Convert world coordinates.
        return self.to_world_coordinates(x=voxel_coordinates)

    def to_world_coordinates(
            self,
            x: np.ndarray,
            origin: Union[None, np.ndarray] = None,
            orientation: Union[None, np.ndarray] = None,
            spacing: Union[None, np.ndarray] = None,
            trim_result: bool = True
    ) -> np.ndarray:
        """
        This mirrors ImageClass.to_world_coordinates
        :param x:
        :param origin:
        :param orientation:
        :param spacing:
        :param trim_result:
        :return:
        """

        # Add empty dimension if the coordinates are unset.
        if len(x.shape) == 1:
            squeeze_on_exit = True
            x = x[:, np.newaxis]

        else:
            squeeze_on_exit = False

        # Setup voxel coordinates
        voxel_coordinates = np.ones([4, x.shape[1]])
        voxel_coordinates[0:3, :] = x

        # Define affine matrix.
        affine_matrix = self.get_affine_matrix(
            origin=origin,
            orientation=orientation,
            spacing=spacing,
            inverse=False
        )

        # World coordinates
        world_coordinates = np.matmul(affine_matrix, voxel_coordinates)

        if trim_result:
            world_coordinates = world_coordinates[0:3, :]

        if squeeze_on_exit:
            world_coordinates = np.squeeze(world_coordinates, axis=1)

        return world_coordinates

    def to_voxel_coordinates(
            self,
            x: np.ndarray,
            origin: Union[None, np.ndarray] = None,
            orientation: Union[None, np.ndarray] = None,
            spacing: Union[None, np.ndarray] = None,
            trim_result: bool = True
    ) -> np.ndarray:
        """
        This mirrors ImageClass.to_voxel_coordinates
        :param x:
        :param origin:
        :param orientation:
        :param spacing:
        :param trim_result:
        :return:
        """

        # Add empty dimension if the coordinates are unset.
        if len(x.shape) == 1:
            squeeze_on_exit = True
            x = x[:, np.newaxis]

        else:
            squeeze_on_exit = False

        # Setup voxel coordinates
        world_coordinates = np.ones([4, x.shape[1]])
        world_coordinates[0:3, :] = x

        # Define inverse affine matrix.
        inverse_affine_matrix = self.get_affine_matrix(
            origin=origin,
            orientation=orientation,
            spacing=spacing,
            inverse=True
        )

        # Convert from world to voxel coordinates
        voxel_coordinates = np.matmul(inverse_affine_matrix, world_coordinates)

        if trim_result:
            voxel_coordinates = voxel_coordinates[0:3, :]

        if squeeze_on_exit:
            voxel_coordinates = np.squeeze(voxel_coordinates, axis=1)

        return voxel_coordinates

    def get_affine_matrix(
            self,
            origin: Union[None, np.ndarray] = None,
            orientation: Union[None, np.ndarray] = None,
            spacing: Union[None, np.ndarray] = None,
            inverse: bool = False
    ) -> np.ndarray:
        """
        This mirrors ImageClass.get_affine_matrix.

        An affine matrix can be used to convert between local (voxel) and world coordinates:
          W = A_origin * A_orientation * A_spacing * X
            = A_affine * X

        Here we use the fact that all right-hand matrices except X are square matrices with the same dimensions,
        and thus are associative. Here we compute the result of A_origin * A_orientation * A_spacing.

        :param origin: Origin of the voxel grid in world coordinates. If None, the origin of the image is used.
        :param orientation: Orientation of the voxel grid in the world coordinate frame. If None, the orientation of
        the image is used.
        :param spacing: Spacing of voxels in world units. If None, the spacing of the image is used.
        :param inverse: Return

        :return: affine 4 x 4 matrix
        """

        # Matrix multiplication of orientation and spacing (scale) matrices (A_orientation * A_spacing).
        matrix = np.matmul(
            self._get_orientation_matrix(orientation=orientation), self._get_spacing_matrix(spacing=spacing)
        )

        # Matrix multiplication of the origin matrix and the previous result (A_origin * (A_orientation * A_spacing)).
        matrix = np.matmul(self._get_origin_matrix(origin=origin), matrix)

        # Compute inverse affine matrix for translating from world coordinates to voxel coordinates.
        if inverse:
            matrix = np.linalg.inv(matrix)

        return matrix

    def _get_orientation_matrix(
            self,
            size: int = 4,
            orientation: Union[None, np.ndarray] = None
    ) -> np.ndarray:
        """
        This mirrors ImageClass._get_orientation_matrix
        :param size:
        :param orientation:
        :return:
        """

        if size not in [3, 4]:
            raise ValueError(f"The size argument should be 3 or 4. Found: {size}")

        if orientation is None:
            orientation = self.image_orientation

        # Create identity matrix and insert orientation matrix.
        orientation_matrix = np.identity(n=size, dtype=float)
        orientation_matrix[0:3, 0:3] = orientation

        return orientation_matrix

    def _get_spacing_matrix(
            self,
            size: int = 4,
            spacing: Union[None, np.ndarray] = None,
            inverse: bool = False
    ) -> np.ndarray:
        """
        This mirrors ImageClass._get_spacing_matrix
        :param size:
        :param spacing:
        :param inverse:
        :return:
        """

        if size not in [3, 4]:
            raise ValueError(f"The size argument should be 3 or 4. Found: {size}")

        if spacing is None:
            spacing = np.array(self.image_spacing)

        if size == 4:
            spacing = np.append(spacing, 1.0)

        if inverse:
            spacing = 1.0 / spacing

        # Create identity matrix.
        spacing_matrix = np.identity(n=size, dtype=float)

        # Fill diagonal of the identity matrix with spacing.
        np.fill_diagonal(spacing_matrix, spacing)

        return spacing_matrix

    def _get_origin_matrix(
            self,
            origin: Union[None, np.ndarray] = None,
            inverse: bool = False
    ) -> np.ndarray:
        """
        This mirrors ImageClass._get_origin_matrix
        :param origin:
        :param inverse:
        :return:
        """

        if origin is None:
            origin = np.array(self.image_origin)

        if inverse:
            origin = -1.0 * origin

        origin_matrix = np.identity(n=4, dtype=float)
        origin_matrix[0:3, 3] = origin

        return origin_matrix