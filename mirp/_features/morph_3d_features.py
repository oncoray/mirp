from functools import cache
from typing import Generator

import numpy as np

from mirp._features.morph_3d_data import (
    Data3DMesh, Data3DConvexHull, Data3DAxisAlignedBoundingBox, Data3DOrientedMinimumBoundingBox,
    Data3DPrincipleComponents, Data3DSpatial
)
from mirp._images.generic_image import GenericImage
from mirp._masks.base_mask import BaseMask
from mirp._features.base_feature import Feature
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass


class Feature3DMorph(Feature):

    def __init__(self, allow_approximation: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.allow_approximation = allow_approximation

    def create_table_name(self):
        table_elements = self._get_base_table_name_element()
        self.table_name = "_".join(table_elements)


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


class Feature3DConvexHull(Feature3DMesh):

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
    ) -> Data3DConvexHull:
        # Get parent_data from cache.
        parent_data = super()._get_data(image=image, mask=mask)

        # Instantiate child using parent attributes.
        data = Data3DConvexHull()
        data.__dict__.update(parent_data.__dict__)

        # Compute convex hull vertices.
        data.compute_convex_hull()

        return data

    @staticmethod
    def _compute(
            data: Data3DConvexHull,
            image: GenericImage | None = None,
            mask: BaseMask | None = None
    ):
        raise NotImplementedError("Implement _compute for feature-specific computation.")


class Feature3DAxisAlignedBoundingBox(Feature3DConvexHull):
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
    ) -> Data3DAxisAlignedBoundingBox:
        # Get parent_data from cache.
        parent_data = super()._get_data(image=image, mask=mask)

        # Instantiate child using parent attributes.
        data = Data3DAxisAlignedBoundingBox()
        data.__dict__.update(parent_data.__dict__)

        # Compute bounding box volume and area.
        data.compute_bounding_box()

        return data

    @staticmethod
    def _compute(
            data: Data3DAxisAlignedBoundingBox,
            image: GenericImage | None = None,
            mask: BaseMask | None = None
    ):
        raise NotImplementedError("Implement _compute for feature-specific computation.")


class Feature3DOrientedMinimumBoundingBox(Feature3DConvexHull):
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
    ) -> Data3DOrientedMinimumBoundingBox:
        # Get parent_data from cache.
        parent_data = super()._get_data(image=image, mask=mask)

        # Instantiate child using parent attributes.
        data = Data3DOrientedMinimumBoundingBox()
        data.__dict__.update(parent_data.__dict__)

        # Compute bounding box volume and area.
        data.compute_bounding_box()

        return data

    @staticmethod
    def _compute(
            data: Data3DOrientedMinimumBoundingBox,
            image: GenericImage | None = None,
            mask: BaseMask | None = None
    ):
        raise NotImplementedError("Implement _compute for feature-specific computation.")


class Feature3DPCA(Feature3DMesh):
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
    ) -> Data3DPrincipleComponents:
        # Get parent_data from cache.
        parent_data = super()._get_data(image=image, mask=mask)

        # Instantiate child using parent attributes.
        data = Data3DPrincipleComponents()
        data.__dict__.update(parent_data.__dict__)

        # Compute semi-axes using principle component analysis.
        data.compute_semi_axes()

        return data

    def compute(self, image: GenericImage, mask: BaseMask):
        # Get data.
        data = self._get_data(image=image, mask=mask)

        # Compute feature value.
        if data.is_empty() or data.is_singular() or data.semi_axes is None:
            self.value = np.nan
        else:
            self.value = self._compute(data=data, image=image, mask=mask)

    @staticmethod
    def _compute(
            data: Data3DPrincipleComponents,
            image: GenericImage | None = None,
            mask: BaseMask | None = None
    ):
        raise NotImplementedError("Implement _compute for feature-specific computation.")



class Feature3DSpatial(Feature3DMesh):
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
            mask: BaseMask,
            allow_approximation: bool
    ) -> Data3DSpatial:
        # Get parent_data from cache.
        parent_data = super()._get_data(image=image, mask=mask)

        # Instantiate child using parent attributes.
        data = Data3DSpatial()
        data.__dict__.update(parent_data.__dict__)

        # Compute bounding box volume and area.
        data.compute_spatial_information(
            image=image,
            mask=mask,
            allow_approximation=allow_approximation
        )

        return data

    def compute(self, image: GenericImage, mask: BaseMask):
        # Get data.
        data = self._get_data(image=image, mask=mask, allow_approximation=self.allow_approximation)

        # Compute feature value.
        if data.is_empty() or data.is_singular() or data.semi_axes is None:
            self.value = np.nan
        else:
            self.value = self._compute(data=data, image=image, mask=mask)

    @staticmethod
    def _compute(
            data: Data3DSpatial,
            image: GenericImage | None = None,
            mask: BaseMask | None = None
    ):
        raise NotImplementedError("Implement _compute for feature-specific computation.")


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
    def _compute(data: Data3DMesh, **kwargs) -> float:
        return len(data.data_morph) * np.prod(data.spacing)


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


class Feature3DMorphSurfaceVolumeRatio(Feature3DMesh):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - surface to volume ratio"
        self.abbr_name = "morph_av"
        self.ibsi_id = "2PR5"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DMesh, **kwargs) -> float:
        return data.area / data.volume


class Feature3DMorphCompactness1(Feature3DMesh):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - compactness 1"
        self.abbr_name = "morph_comp_1"
        self.ibsi_id = "SKGS"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DMesh, **kwargs) -> float:
        x = 36.0 * np.pi * data.volume ** 2.0 / data.area ** 3.0
        return 1.0 / (6.0 * np.pi) * x ** (1.0 / 2.0)


class Feature3DMorphCompactness2(Feature3DMesh):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - compactness 2"
        self.abbr_name = "morph_comp_2"
        self.ibsi_id = "BQWJ"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DMesh, **kwargs) -> float:
        x = 36.0 * np.pi * data.volume ** 2.0 / data.area ** 3.0
        return x


class Feature3DMorphSphericalDisproportion(Feature3DMesh):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - spherical disproportion"
        self.abbr_name = "morph_sph_dispr"
        self.ibsi_id = "KRCK"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DMesh, **kwargs) -> float:
        x = 36.0 * np.pi * data.volume ** 2.0 / data.area ** 3.0
        return x ** (-1.0 / 3.0)


class Feature3DMorphSphericity(Feature3DMesh):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - sphericity"
        self.abbr_name = "morph_sphericity"
        self.ibsi_id = "QCFX"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DMesh, **kwargs) -> float:
        x = 36.0 * np.pi * data.volume ** 2.0 / data.area ** 3.0
        return x ** (1.0 / 3.0)


class Feature3DMorphAsphericity(Feature3DMesh):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - asphericity"
        self.abbr_name = "morph_asphericity"
        self.ibsi_id = "25C7"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DMesh, **kwargs) -> float:
        x = 36.0 * np.pi * data.volume ** 2.0 / data.area ** 3.0
        return x ** (-1.0 / 3.0) - 1.0


class Feature3DMorphCentreOfMassShift(Feature3DMesh):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - centre of mass shift"
        self.abbr_name = "morph_com"
        self.ibsi_id = "KLMA"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DMesh, **kwargs) -> float:
        # Compute centre of mass for morphological and intensity-weighted mask.
        com_morph = np.array([
            np.mean(data.data_morph.z),
            np.mean(data.data_morph.y),
            np.mean(data.data_morph.x)
        ])
        com_int = np.array([
            np.sum(data.data_int.g * data.data_int.z),
            np.sum(data.data_int.g * data.data_int.y),
            np.sum(data.data_int.g * data.data_int.x)
        ]) / np.sum(data.data_int.g)

        # Calculate shift in centre of mass.
        return np.sqrt(np.sum(np.multiply(com_morph - com_int, data.spacing) ** 2.0))


class Feature3DMorphIntegratedIntensity(Feature3DMesh):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - "
        self.abbr_name = "morph_integ_int"
        self.ibsi_id = "99N0"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DMesh, **kwargs) -> float:
        return data.volume * np.mean(data.data_int.g)


class Feature3DMorphMaximum3DDiameter(Feature3DConvexHull):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - maximum 3D diameter"
        self.abbr_name = "morph_diam"
        self.ibsi_id = "L0JK"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DConvexHull, **kwargs) -> float:
        from scipy.spatial.distance import pdist
        return np.max(pdist(data.convex_hull_vertices))


class Feature3DMorphConvexHullVolumeDensity(Feature3DConvexHull):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - convex hull volume density"
        self.abbr_name = "morph_vol_dens_conv_hull"
        self.ibsi_id = "R3ER"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DConvexHull, **kwargs) -> float:
        return data.volume / data.convex_hull_volume


class Feature3DMorphConvexHullAreaDensity(Feature3DConvexHull):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - convex hull area density"
        self.abbr_name = "morph_area_dens_conv_hull"
        self.ibsi_id = "7T7F"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DConvexHull, **kwargs) -> float:
        return data.area / data.convex_hull_area


class Feature3DMorphAxisAlignedBoundingBoxVolumeDensity(Feature3DAxisAlignedBoundingBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - axis-aligned bounding box volume density"
        self.abbr_name = "morph_vol_dens_aabb"
        self.ibsi_id = "PBX1"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DAxisAlignedBoundingBox, **kwargs) -> float:
        return data.volume / data.bounding_box_volume


class Feature3DMorphAxisAlignedBoundingBoxAreaDensity(Feature3DAxisAlignedBoundingBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - axis-aligned bounding box area density"
        self.abbr_name = "morph_area_dens_aabb"
        self.ibsi_id = "R59B"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DAxisAlignedBoundingBox, **kwargs) -> float:
        return data.area / data.bounding_box_area


class Feature3DMorphOrientedMinimumBoundingBoxVolumeDensity(Feature3DOrientedMinimumBoundingBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - oriented minimum bounding box volume density"
        self.abbr_name = "morph_vol_dens_ombb"
        self.ibsi_id = "ZH1A"
        self.ibsi_compliant = False

    @staticmethod
    def _compute(data: Data3DOrientedMinimumBoundingBox, **kwargs) -> float:
        return data.volume / data.bounding_box_volume


class Feature3DMorphOrientedMinimumBoundingBoxAreaDensity(Feature3DOrientedMinimumBoundingBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - oriented minimum bounding box area density"
        self.abbr_name = "morph_area_dens_ombb"
        self.ibsi_id = "IQYR"
        self.ibsi_compliant = False

    @staticmethod
    def _compute(data: Data3DOrientedMinimumBoundingBox, **kwargs) -> float:
        return data.volume / data.bounding_box_volume


class Feature3DMorphMajorAxisLength(Feature3DPCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - major axis length"
        self.abbr_name = "morph_pca_maj_axis"
        self.ibsi_id = "TDIC"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DPrincipleComponents, **kwargs) -> float:
        return data.semi_axes[2] * 2.0


class Feature3DMorphMinorAxisLength(Feature3DPCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - minor axis length"
        self.abbr_name = "morph_pca_min_axis"
        self.ibsi_id = "P9VJ"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DPrincipleComponents, **kwargs) -> float:
        return data.semi_axes[1] * 2.0


class Feature3DMorphShortestAxisLength(Feature3DPCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - shortest axis length"
        self.abbr_name = "morph_pca_least_axis"
        self.ibsi_id = "7J51"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DPrincipleComponents, **kwargs) -> float:
        return data.semi_axes[0] * 2.0


class Feature3DMorphElongation(Feature3DPCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - elongation"
        self.abbr_name = "morph_pca_elongation"
        self.ibsi_id = "Q3CK"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DPrincipleComponents, **kwargs) -> float:
        return data.semi_axes[1] / data.semi_axes[2]


class Feature3DMorphFlatness(Feature3DPCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - flatness"
        self.abbr_name = "morph_pca_flatness"
        self.ibsi_id = "N17B"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DPrincipleComponents, **kwargs) -> float:
        return data.semi_axes[0] / data.semi_axes[2]


class Feature3DMorphApproximateEnclosingEllipsoidVolumeDensity(Feature3DPCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - approximate enclosing ellipsoid volume density"
        self.abbr_name = "morph_vol_dens_aee"
        self.ibsi_id = "6BDE"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DPrincipleComponents, **kwargs) -> float:
        if np.any(data.semi_axes == 0.0):
            return np.nan
        return 3.0 * data.volume / (4.0 * np.pi * np.prod(data.semi_axes))


class Feature3DMorphApproximateEnclosingEllipsoidAreaDensity(Feature3DPCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - approximate enclosing ellipsoid area density"
        self.abbr_name = "morph_area_dens_aee"
        self.ibsi_id = "RDD2"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DPrincipleComponents, **kwargs) -> float:
        if np.any(data.semi_axes == 0.0):
            return np.nan
        return data.area / data.get_ellipsoid_surface_area(n_degree=20)


class Feature3DMorphMoranIndex(Feature3DSpatial):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - Moran's I index"
        self.abbr_name = "morph_moran_i"
        self.ibsi_id = "N365"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DSpatial, **kwargs) -> float:
        return data.moran_i


class Feature3DMorphGearyMeasure(Feature3DSpatial):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Morphology (3D) - Geary's C measure"
        self.abbr_name = "morph_geary_c"
        self.ibsi_id = "NPT7"
        self.ibsi_compliant = True

    @staticmethod
    def _compute(data: Data3DSpatial, **kwargs) -> float:
        return data.geary_c


def get_morphology_3d_class_dict() -> dict[str, Feature3DMorph]:
    class_dict = {
        "morph_volume": Feature3DMorphVolume,
        "morph_vol_approx": Feature3DMorphApproximateVolume,
        "morph_area_mesh": Feature3DMorphSurfaceArea,
        "morph_av": Feature3DMorphSurfaceVolumeRatio,
        "morph_comp_1": Feature3DMorphCompactness1,
        "morph_comp_2": Feature3DMorphCompactness2,
        "morph_sph_dispr": Feature3DMorphSphericalDisproportion,
        "morph_sphericity": Feature3DMorphSphericity,
        "morph_asphericity": Feature3DMorphAsphericity,
        "morph_com": Feature3DMorphCentreOfMassShift,
        "morph_diam": Feature3DMorphMaximum3DDiameter,
        "morph_pca_maj_axis": Feature3DMorphMajorAxisLength,
        "morph_pca_min_axis": Feature3DMorphMinorAxisLength,
        "morph_pca_least_axis": Feature3DMorphShortestAxisLength,
        "morph_pca_elongation": Feature3DMorphElongation,
        "morph_pca_flatness": Feature3DMorphFlatness,
        "morph_vol_dens_aabb": Feature3DMorphAxisAlignedBoundingBoxVolumeDensity,
        "morph_area_dens_aabb": Feature3DMorphAxisAlignedBoundingBoxAreaDensity,
        "morph_vol_dens_aee": Feature3DMorphApproximateEnclosingEllipsoidVolumeDensity,
        "morph_area_dens_aee": Feature3DMorphApproximateEnclosingEllipsoidAreaDensity,
        "morph_vol_dens_conv_hull": Feature3DMorphConvexHullVolumeDensity,
        "morph_area_dens_conv_hull": Feature3DMorphConvexHullAreaDensity,
        "morph_integ_int": Feature3DMorphIntegratedIntensity,
        "morph_moran_i": Feature3DMorphMoranIndex,
        "morph_geary_c": Feature3DMorphGearyMeasure,
        "morph_vol_dens_ombb": Feature3DMorphOrientedMinimumBoundingBoxVolumeDensity,
        "morph_area_dens_ombb": Feature3DMorphOrientedMinimumBoundingBoxAreaDensity,
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
        yield class_dict[feature](allow_approximation = not settings.no_approximation)
