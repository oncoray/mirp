import copy

import numpy as np


class BaseImage:

    def __init__(
            self,
            sample_name: None | str = None,
            image_modality: None | str = None,
            image_origin: None | tuple[float, ...] = None,
            image_orientation: None | np.ndarray = None,
            image_spacing: None | tuple[float, ...] = None,
            image_dimensions: None | tuple[int, ...] = None,
            separate_slices: None | bool = None,
            metadata: None | dict[str, str] = None,
            **kwargs
    ):
        # Make cooperative
        super().__init__()

        # Set modality.
        if isinstance(image_modality, str):
            image_modality = image_modality.lower()
        self.modality = image_modality

        # Set affine-transformation related values.
        self.image_origin = copy.deepcopy(image_origin)
        self.image_orientation = copy.deepcopy(image_orientation)
        self.image_spacing = copy.deepcopy(image_spacing)
        self.image_dimension = copy.deepcopy(image_dimensions)

        # Set sample name.
        self.sample_name = sample_name

        # Initialise associated masks, which is required for passing image
        self.image_data = None
        self.associated_masks = None

        # Set metadata. Entries with empty values (None or "") are removed from the metadata dict to avoid
        # polluting the dictionary with unset entries.
        if isinstance(metadata, dict) and len(metadata) > 0:
            metadata = copy.deepcopy(metadata)
            remove_keys = [key for key, value in metadata.items() if value is None or value == ""]
            for current_key in remove_keys:
                metadata.pop(current_key, None)

            if len(metadata) > 0:
                self.object_metadata = metadata
            else:
                self.object_metadata = dict()
        else:
            self.object_metadata = dict()

        # Determines whether slices in the stack should be treated separately.
        self.separate_slices = separate_slices

    def copy(self, drop_image=False) -> Self:
        image = copy.deepcopy(self)

        if drop_image:
            image.drop_image()

        return image

    def drop_image(self):
        self.image_data = None

    def is_isotropic(self) -> bool:
        if self.separate_slices:
            spacing = np.array(self.image_spacing)[[1, 2]]
        else:
            spacing = np.array(self.image_spacing)

        return np.all(spacing == spacing[0])

    def set_modality(self, modality: None | str):
        from mirp._data_import.utilities import supported_image_modalities
        if modality is None:
            return

        if not isinstance(modality, str):
            raise ValueError(f"modality is expected to be a character string. Found: {modality}")

        if modality == "generic":
            raise ValueError(f"modality cannot be 'generic'")

        modality = supported_image_modalities(modality)
        if self.modality is None or self.modality == "generic":
            self.modality = modality[0]

    @staticmethod
    def get_dir_path():
        # BaseImage does not have an associated directory path.
        return None

    @staticmethod
    def get_file_name():
        # BaseImage also has no associated file name.
        return None

    def get_image_origin(self, as_str=False):
        if not as_str:
            return self.image_origin

        if self.image_origin is None:
            return "unset_image_origin"

        return str(self.image_origin)

    def get_image_orientation(self, as_str=False):
        if not as_str:
            return self.image_orientation

        if self.image_orientation is None:
            return "unset_image_orientation"

        return str(np.ravel(self.image_orientation))

    def get_image_spacing(self, as_str=False):
        if not as_str:
            return self.image_spacing

        if self.image_spacing is None:
            return "unset_image_spacing"

        return str(self.image_spacing)

    def get_image_dimension(self, as_str=False):
        if not as_str:
            return self.image_dimension

        if self.image_dimension is None:
            return "unset_image_dimension"

        return str(self.image_dimension)

    def remove_metadata(self, force=False):
        if force:
            self.object_metadata = dict()

    def associate_with_mask(
            self,
            mask_list,
            association_strategy: None | set[str] = None
    ):
        if mask_list is None or len(mask_list) == 0 or association_strategy is None:
            return

        # Match on sample name.
        if "sample_name" in association_strategy and self.sample_name is not None:
            matching_mask_list = [
                mask_file for mask_file in mask_list
                if self.sample_name == mask_file.sample_name
            ]

            if len(matching_mask_list) > 0:
                self.associated_masks = matching_mask_list
                return

        return

    def on_file_system(self):
        # The BaseImage object is by its nature not on the file system.
        return False

    def check_associated_masks(self):
        if self.associated_masks is None:
            return

        for mask in self.associated_masks:
            self._check_associated_mask_image_data(mask=mask)

    def _check_associated_mask_image_data(self, mask):
        """
        Check whether image and associated mask plausibly share the same frame of reference. This method is only
        used during import of BaseImage and BaseMask.
        """

        problem_list = []
        # Mismatch in grid dimension
        if not np.array_equal(self.get_image_dimension(), mask.get_image_dimension()):
                problem_list += [
                    f"different dimensions: \n\t\timage: {self.get_image_dimension()}\n\t\tmask: {mask.get_image_dimension()}"
                ]

        # Mismatch in origin
        if not np.allclose(self.get_image_origin(), mask.get_image_origin()):
            problem_list += [
                f"different origin: \n\t\timage: {self.get_image_origin()}\n\t\tmask: {mask.get_image_origin()}"
            ]

        # Mismatch in spacing
        if not np.allclose(self.get_image_spacing(), mask.get_image_spacing()):
            problem_list += [
                f"different spacing: \n\t\timage: {self.get_image_spacing()}\n\t\tmask: {mask.get_image_spacing()}"
            ]

        # Mismatch in orientation
        if not np.allclose(self.get_image_orientation(), mask.get_image_orientation()):
            problem_list += [
                f"different orientation: \n\t\timage: {np.ravel(self.get_image_orientation())}\n\t\tmask: "
                f"{np.ravel(mask.get_image_orientation())}"
            ]

        if len(problem_list) > 0:
            warnings.warn(
                f"Image and mask may not have the same frame of "
                f"reference. Please check if segmentation masks are placed correctly:\n\t" + "\n\t".join(problem_list),
                UserWarning
            )

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
            origin: None | np.ndarray = None,
            orientation: None | np.ndarray = None,
            spacing: None | np.ndarray = None,
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
            origin: None | np.ndarray = None,
            orientation: None | np.ndarray = None,
            spacing: None | np.ndarray = None,
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
            origin: None | np.ndarray = None,
            orientation: None | np.ndarray = None,
            spacing: None | np.ndarray = None,
            inverse: bool = False
    ) -> np.ndarray:
        """
        This mirrors ImageClass.get_affine_matrix.

        An affine matrix can be used to convert between local (voxel) and world coordinates:
          W = A_origin * A_orientation * A_spacing * X
            = A_affine * X

        Here we use the fact that all right-hand matrices except X are square matrices with the same dimensions,
        and thus are associative. Here we compute the result of A_origin * A_orientation * A_spacing.

        Parameters
        ----------
        origin: np.ndarray, optional, default: None
            Origin of the voxel grid in world coordinates. If None, the origin of the image is used.

        orientation: np.ndarray, optional, default: None
            Orientation of the voxel grid in the world coordinate frame. If None, the orientation of
            the image is used.

        spacing: np.ndarray, optional, default: None
            Spacing of voxels in world units. If None, the spacing of the image is used.

        inverse: bool, default: False
            Return inverse of the affine matrix

        Returns
        -------
        np.ndarray
            Affine matrix or inverse affine matrix (if `inverse=True`)
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
            orientation: None | np.ndarray = None
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
            spacing: None | np.ndarray = None,
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
            origin: None | np.ndarray = None,
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
