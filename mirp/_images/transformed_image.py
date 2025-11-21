import pandas as pd
from typing import Any

from mirp._images.generic_image import GenericImage


class TransformedImage(GenericImage):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    @staticmethod
    def get_default_ivh_discretisation_method():
        return "fixed_bin_number"


class LogarithmTransformedImage(TransformedImage):
    def __init__(
            self,
            template: None | GenericImage = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Update image parameters using the template.
        if isinstance(template, GenericImage):
            self.update_from_template(template=template)

    def get_file_name_descriptor(self) -> list[str]:
        descriptors = super().get_file_name_descriptor()
        descriptors += ["lgrthm"]

        return descriptors

    def get_export_attributes(self) -> dict[str, Any]:
        parent_attributes = super().get_export_attributes()
        attributes = [("filter_type", "logarithm_transformation")]
        parent_attributes.update(dict(attributes))

        return parent_attributes

    def parse_feature_names(self, x: None | pd.DataFrame) -> pd.DataFrame:
        x = super().parse_feature_names(x=x)
        feature_name_prefix = ["lgrthm"]

        if len(feature_name_prefix) > 0:
            feature_name_prefix = "_".join(feature_name_prefix)
            feature_name_prefix += "_"
            x.columns = feature_name_prefix + x.columns

        return x


class ExponentialTransformedImage(TransformedImage):
    def __init__(
            self,
            template: None | GenericImage = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Update image parameters using the template.
        if isinstance(template, GenericImage):
            self.update_from_template(template=template)

    def get_file_name_descriptor(self) -> list[str]:
        descriptors = super().get_file_name_descriptor()
        descriptors += ["exp"]

        return descriptors

    def get_export_attributes(self) -> dict[str, Any]:
        parent_attributes = super().get_export_attributes()
        attributes = [("filter_type", "exponential_transformation")]
        parent_attributes.update(dict(attributes))

        return parent_attributes

    def parse_feature_names(self, x: None | pd.DataFrame) -> pd.DataFrame:
        x = super().parse_feature_names(x=x)
        feature_name_prefix = ["exp"]

        if len(feature_name_prefix) > 0:
            feature_name_prefix = "_".join(feature_name_prefix)
            feature_name_prefix += "_"
            x.columns = feature_name_prefix + x.columns

        return x


class LocalBinaryPatternImage(TransformedImage):
    def __init__(
            self,
            distance: None | float = None,
            separate_slices: None | bool = None,
            lbp_method: None | str = None,
            template: None | GenericImage = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Update image parameters using the template.
        if isinstance(template, GenericImage):
            self.update_from_template(template=template)

        self.distance = distance
        self.separate_slices = separate_slices
        self.lbp_method = lbp_method

    def get_file_name_descriptor(self) -> list[str]:
        descriptors = super().get_file_name_descriptor()
        descriptors += [
            "lbp",
            "2d" if self.separate_slices else "3d"
        ]

        # Don't add anything if the method is "default".
        if self.lbp_method == "variance":
            descriptors += ["var"]
        elif self.lbp_method == "kurtosis":
            descriptors += ["kurt"]
        elif self.lbp_method == "rotation_invariant":
            descriptors += ["rot_invar"]

        descriptors += ["d" + str(self.distance)]

        return descriptors

    def get_export_attributes(self) -> dict[str, Any]:
        parent_attributes = super().get_export_attributes()
        attributes = [
            ("filter_type", "lbp"),
            ("filter_direction", "2d" if self.separate_slices else "3d"),
            ("distance", self.distance),
            ("lbp_method", self.lbp_method)
        ]
        parent_attributes.update(dict(attributes))

        return parent_attributes

    def parse_feature_names(self, x: None | pd.DataFrame) -> pd.DataFrame:
        x = super().parse_feature_names(x=x)
        feature_name_prefix = [
            "lbp",
            "2d" if self.separate_slices else "3d"
        ]

        # Don't add anything if the method is "default".
        if self.lbp_method == "variance":
            feature_name_prefix += ["var"]
        elif self.lbp_method == "kurtosis":
            feature_name_prefix += ["kurt"]
        elif self.lbp_method == "rotation_invariant":
            feature_name_prefix += ["rot_invar"]

        feature_name_prefix += ["d" + str(self.distance)]

        if len(feature_name_prefix) > 0:
            feature_name_prefix = "_".join(feature_name_prefix)
            feature_name_prefix += "_"
            x.columns = feature_name_prefix + x.columns

        return x
