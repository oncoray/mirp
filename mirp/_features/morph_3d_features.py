from functools import cache
from typing import Generator

import numpy as np

from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.base_feature import Feature
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class Feature3DMorph(Feature):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_table_name(self):
        table_elements = self._get_base_table_name_element()
        self.table_name = "_".join(table_elements)


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


class Feature3DMesh(Feature3DMorph):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def clear_cache(self):
        super().clear_cache()
        self._get_data.cache_clear()

    @staticmethod
    @cache
    def _get_data(
            image: GenericImage,
            mask: BaseMask
    ) -> Data3DMesh:
        data = Data3DMesh()
        data.compute(image=image, mask=mask)

        return data

    @staticmethod
    def _compute(data: Data3DMesh, image: GenericImage | None = None, mask: BaseMask | None = None):
        raise NotImplementedError("Implement _compute for feature-specific computation.")

    def compute(self, image: GenericImage, mask: BaseMask):
        # Get data.
        data = self._get_data(image=image, mask=mask)

        # Compute feature value.
        if data.is_empty():
            self.value = np.nan
        else:
            self.value = self._compute(data=data, image=image, mask=mask)


class Feature3DMorphVolume(Feature3DMesh):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - volume"
        self.abbr_name = "morph_volume"
        self.ibsi_id = "RNU0"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DMesh, **kwargs) -> float:
        return data.volume


class Feature3DMorphApproximateVolume(Feature3DMesh):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - approximate volume"
        self.abbr_name = "morph_vol_approx"
        self.ibsi_id = "YEKZ"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DMesh, mask: BaseMask | None = None, **kwargs) -> float:
        return np.sum(mask.roi_morphology.get_voxel_grid()) * np.prod(mask.roi_morphology.image_spacing)


class Feature3DMorphSurfaceArea(Feature3DMesh):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - surface area"
        self.abbr_name = "morph_area_mesh"
        self.ibsi_id = "C0JK"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DMesh, **kwargs) -> float:
        return data.area


def get_morphology_3d_class_dict() -> dict[str, Feature3DMorph]:
    class_dict = {
        "morph_volume": Feature3DMorphVolume,
        "morph_vol_approx": Feature3DMorphApproximateVolume,
        "morph_area_mesh": Feature3DMorphSurfaceArea,
        "morph_av": 1,
        "morph_comp_1": 1,
        "morph_comp_2": 1,
        "morph_sph_dispr": 1,
        "morph_sphericity": 1,
        "morph_asphericity": 1,
        "morph_com": 1,
        "morph_diam": 1,
        "morph_pca_maj_axis": 1,
        "morph_pca_min_axis": 1,
        "morph_pca_least_axis": 1,
        "morph_pca_elongation": 1,
        "morph_pca_flatness": 1,
        "morph_vol_dens_aabb": 1,
        "morph_area_dens_aabb": 1,
        "morph_vol_dens_aee": 1,
        "morph_area_dens_aee": 1,
        "morph_vol_dens_conv_hull": 1,
        "morph_area_dens_conv_hull": 1,
        "morph_integ_int": 1,
        "morph_moran_i": 1,
        "morph_geary_c": 1,
        "morph_vol_dens_ombb": 1,
        "morph_area_dens_ombb": 1,
        "morph_vol_dens_mvee": 1,
        "morph_area_dens_mvee": 1
    }

    return class_dict


def generate_morph_3d_features(
        settings: FeatureExtractionSettingsClass,
        features: None | list[str] = None
) -> Generator[Feature3DMorph, None, None]:
    class_dict = get_morphology_3d_class_dict()
    morph_features = list(class_dict.keys())

    # Populate features if available.
    if features is None and settings.has_morphology_family():
        features = morph_features

    # Terminate early if no features are set, and none are required.
    if features is None:
        return

    # Select only morphological features, and return if none are present.
    features = [feature for feature in features if feature in morph_features]
    if len(features) == 0:
        return

    for feature in features:
        yield class_dict[feature]()
