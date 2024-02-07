import copy
import warnings
from typing import Union, List
from xml.etree.ElementTree import Element
from xml.etree import ElementTree as ElemTree

from mirp.settings.settingsGeneric import SettingsClass
from mirp.settings.settingsImageTransformation import get_image_transformation_settings
from mirp.settings.settingsFeatureExtraction import get_feature_extraction_settings
from mirp.settings.settingsMaskResegmentation import get_mask_resegmentation_settings
from mirp.settings.settingsPerturbation import get_perturbation_settings
from mirp.settings.settingsImageProcessing import get_post_processing_settings
from mirp.settings.settingsInterpolation import get_image_interpolation_settings, get_mask_interpolation_settings
from mirp.settings.settingsGeneral import get_general_settings
from mirp.settings.utilities import update_settings_from_branch


def create_settings_object(
    xml_tree: None | Element = None,
    **kwargs
) -> SettingsClass:
    kwargs = copy.deepcopy(kwargs)

    if isinstance(xml_tree, Element):
        # General settings
        update_settings_from_branch(
            kwargs=kwargs,
            branch=xml_tree.find("general"),
            settings=get_general_settings()
        )

        # Post-processing settings
        update_settings_from_branch(
            kwargs=kwargs,
            branch=xml_tree.find("post_processing"),
            settings=get_post_processing_settings()
        )

        # Image interpolation settings
        if xml_tree.find("img_interpolate") is not None and xml_tree.find("img_interpolate").find("new_non_iso_spacing") is not None:
            warnings.warn(
                f"The new_non_iso_spacing tag has been deprecated. Use the new_spacing tag instead.",
                DeprecationWarning
            )

        update_settings_from_branch(
            kwargs=kwargs,
            branch=xml_tree.find("img_interpolate"),
            settings=get_image_interpolation_settings()
        )

        # Mask interpolation settings
        update_settings_from_branch(
            kwargs=kwargs,
            branch=xml_tree.find("roi_interpolate"),
            settings=get_mask_interpolation_settings()
        )

        # Perturbation settings
        update_settings_from_branch(
            kwargs=kwargs,
            branch=xml_tree.find("vol_adapt"),
            settings=get_perturbation_settings()
        )

        # Mask resegmentation settings
        update_settings_from_branch(
            kwargs=kwargs,
            branch=xml_tree.find("roi_resegment"),
            settings=get_mask_resegmentation_settings()
        )

        # Feature extraction settings
        if xml_tree.find("feature_extr") is not None and xml_tree.find("feature_extr").find("glcm_merge_method") is not None:
            warnings.warn(
                "The glcm_merge_method tag has been deprecated. Use the glcm_spatial_method tag instead. This takes"
                " the following values: `2d_average`, `2d_slice_merge`, '2.5d_direction_merge', '2.5d_volume_merge',"
                " '3d_average', and `3d_volume_merge`",
                DeprecationWarning
            )

        if xml_tree.find("feature_extr") is not None and xml_tree.find("feature_extr").find("glrlm_merge_method") is not None:
            warnings.warn(
                "The glrlm_merge_method tag has been deprecated. Use the glrlm_spatial_method tag instead. This "
                "takes the following values: `2d_average`, `2d_slice_merge`, '2.5d_direction_merge', "
                "'2.5d_volume_merge', '3d_average', and `3d_volume_merge`",
                DeprecationWarning
            )

        update_settings_from_branch(
            kwargs=kwargs,
            branch=xml_tree.find("feature_extr"),
            settings=get_feature_extraction_settings()
        )

        # Image transformation settings
        if xml_tree.find("img_transform") is not None and xml_tree.find("img_transform").find("log_average") is not None:
            warnings.warn(
                "The log_average tag has been deprecated. Use the laplacian_of_gaussian_pooling_method tag "
                "instead with the value `mean` to emulate log_average=True.",
                DeprecationWarning
            )

        if xml_tree.find("img_transform") is not None and xml_tree.find("img_transform").find("riesz_steered") is not None:
            warnings.warn(
                "The riesz_steered tag has been deprecated. Steerable Riesz filter are now identified by the name "
                "of the filter kernel (filter_kernels parameter).",
                DeprecationWarning
            )

        update_settings_from_branch(
            kwargs=kwargs,
            branch=xml_tree.find("img_transform"),
            settings=get_image_transformation_settings()
        )

        # Deep learning branch
        if xml_tree.find("deep_learning") is not None:
            warnings.warn(
                "deep_learning parameter branch has been deprecated. Parameters for image "
                "processing for deep learning can now be set directly using the deep_learning_preprocessing function.",
                DeprecationWarning
            )

    # Create settings class.
    settings = SettingsClass(**kwargs)

    return settings


def import_configuration_settings(
        compute_features: bool,
        path: Union[None, str] = None,
        **kwargs
) -> List[SettingsClass]:
    import os.path

    # Make a copy of the kwargs argument to avoid updating by reference.
    kwargs = copy.deepcopy(kwargs)

    # Prevent checking of feature parameters if features are not computed.
    if not compute_features:
        kwargs.update({
            "base_feature_families": "none",
            "response_map_feature_families": "none"
        })
    else:
        if "base_feature_families" not in kwargs:
            kwargs.update({"base_feature_families": "all"})
        if "response_map_feature_families" not in kwargs:
            kwargs.update({"response_map_feature_families": "statistics"})

    # Keywords only.
    if path is None:
        return [create_settings_object(xml_tree=None, **kwargs)]

    # Check path to xml file.
    if not os.path.exists(path):
        raise FileNotFoundError(f"The settings file could not be found at {path}.")

    # Load xml file.
    tree = ElemTree.parse(path)

    return [create_settings_object(xml_tree=branch, **kwargs) for branch in tree.getroot().findall("config")]

