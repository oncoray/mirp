import copy
from typing import Optional, List, Dict, Any

from mirp.images.genericImage import GenericImage


class TransformedImage(GenericImage):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)


class GaborTransformedImage(TransformedImage):
    def __init__(
            self,
            sigma_parameter: Optional[float] = None,
            gamma_parameter: Optional[float] = None,
            lambda_parameter: Optional[float] = None,
            theta_parameter: Optional[float] = None,
            pool_theta: Optional[bool] = None,
            response_type: Optional[str] = None,
            rotation_invariance: Optional[bool] = None,
            pooling_method: Optional[str] = None,
            boundary_condition: Optional[str] = None,
            riesz_order: Optional[int, List[int]] = None,
            riesz_steering: Optional[bool] = None,
            riesz_sigma_parameter: Optional[float] = None,
            template: Optional[GenericImage] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Filter parameters
        self.sigma_parameter = sigma_parameter
        self.gamma_parameter = gamma_parameter
        self.lambda_parameter = lambda_parameter
        self.theta_parameter = theta_parameter
        self.pool_theta = pool_theta
        self.response_type = response_type
        self.rotation_invariance = rotation_invariance
        self.pooling_method = pooling_method
        self.boundary_condition = boundary_condition
        self.riesz_transformed = riesz_order is not None
        self.riesz_order = copy.deepcopy(riesz_order)
        self.riesz_steering = riesz_steering
        self.riesz_sigma_parameter = riesz_sigma_parameter

        # Update image parameters using the template.
        if isinstance(template, GenericImage):
            self.update_from_template(template=template)

    def get_file_name_descriptor(self) -> List[str]:
        descriptors = super().get_file_name_descriptor()

        descriptors += [
            "gabor",
            "s", str(self.sigma_parameter),
            "g", str(self.gamma_parameter),
            "l", str(self.lambda_parameter)
        ]

        if not self.pool_theta:
            descriptors += ["t", str(self.theta_parameter)]

        descriptors += ["2D" if self.separate_slices else "3D"]

        if self.rotation_invariance and not self.separate_slices:
            descriptors += ["invar"]

        return descriptors

    def get_export_attributes(self) -> Dict[Any]:
        parent_attributes = super().get_export_attributes()

        attributes = [
            ("filter_type", "gabor"),
            ("sigma_parameter", self.sigma_parameter),
            ("gamma_parameter", self.gamma_parameter),
            ("lambda_parameter", self.lambda_parameter),
            ("theta_parameter", self.theta_parameter),
            ("pool_theta", self.pool_theta),
            ("response_type", self.response_type),
            ("rotation_invariance", self.rotation_invariance),
            ("boundary_condition", self.boundary_condition)
        ]

        if self.pooling_method is not None:
            attributes += [("pooling_method", self.pooling_method)]

        if self.riesz_transformed:
            attributes += [("riesz_order", self.riesz_order)]

            if self.riesz_steering:
                attributes += [("riesz_sigma_parameter", self.riesz_sigma_parameter)]

        parent_attributes.update(dict(attributes))

        return parent_attributes


class GaussianTransformedImage(TransformedImage):
    def __init__(
            self,
            sigma_parameter: Optional[float] = None,
            sigma_cutoff_parameter: Optional[float] = None,
            boundary_condition: Optional[str] = None,
            riesz_order: Optional[int, List[int]] = None,
            riesz_steering: Optional[bool] = None,
            riesz_sigma_parameter: Optional[float] = None,
            template: Optional[GenericImage] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Filter parameters
        self.sigma_parameter = sigma_parameter
        self.sigma_cutoff_parameter = sigma_cutoff_parameter
        self.boundary_condition = boundary_condition
        self.riesz_transformed = riesz_order is not None
        self.riesz_order = copy.deepcopy(riesz_order)
        self.riesz_steering = riesz_steering
        self.riesz_sigma_parameter = riesz_sigma_parameter

        # Update image parameters using the template.
        if isinstance(template, GenericImage):
            self.update_from_template(template=template)

    def get_file_name_descriptor(self) -> List[str]:
        descriptors = super().get_file_name_descriptor()

        descriptors += [
            "gaussian",
            "s", str(self.sigma_parameter)
        ]

        return descriptors

    def get_export_attributes(self) -> Dict[Any]:
        parent_attributes = super().get_export_attributes()

        attributes = [
            ("filter_type", "gaussian"),
            ("sigma_parameter", self.sigma_parameter),
            ("sigma_cutoff_parameter", self.sigma_cutoff_parameter),
            ("boundary_condition", self.boundary_condition)
        ]

        if self.riesz_transformed:
            attributes += [("riesz_order", self.riesz_order)]

            if self.riesz_steering:
                attributes += [("riesz_sigma_parameter", self.riesz_sigma_parameter)]

        parent_attributes.update(dict(attributes))

        return parent_attributes


class LaplacianOfGaussianTransformedImage(TransformedImage):
    def __init__(
            self,
            sigma_parameter: Optional[float] = None,
            sigma_cutoff_parameter: Optional[float] = None,
            pooling_method: Optional[str] = None,
            boundary_condition: Optional[str] = None,
            riesz_order: Optional[int, List[int]] = None,
            riesz_steering: Optional[bool] = None,
            riesz_sigma_parameter: Optional[float] = None,
            template: Optional[GenericImage] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Filter parameters
        self.sigma_parameter = sigma_parameter
        self.sigma_cutoff_parameter = sigma_cutoff_parameter
        self.pooling_method = pooling_method
        self.boundary_condition = boundary_condition
        self.riesz_transformed = riesz_order is not None
        self.riesz_order = copy.deepcopy(riesz_order)
        self.riesz_steering = riesz_steering
        self.riesz_sigma_parameter = riesz_sigma_parameter

        # Update image parameters using the template.
        if isinstance(template, GenericImage):
            self.update_from_template(template=template)

    def get_file_name_descriptor(self) -> List[str]:
        descriptors = super().get_file_name_descriptor()

        descriptors += [
            "log",
            "s", str(self.sigma_parameter)
        ]

        return descriptors

    def get_export_attributes(self) -> Dict[Any]:
        parent_attributes = super().get_export_attributes()

        attributes = [
            ("filter_type", "laplacian_of_gaussian"),
            ("sigma_parameter", self.sigma_parameter),
            ("sigma_cutoff_parameter", self.sigma_cutoff_parameter),
            ("boundary_condition", self.boundary_condition)
        ]

        if self.pooling_method is not None:
            attributes += [("pooling_method", self.pooling_method)]

        if self.riesz_transformed:
            attributes += [("riesz_order", self.riesz_order)]

            if self.riesz_steering:
                attributes += [("riesz_sigma_parameter", self.riesz_sigma_parameter)]

        parent_attributes.update(dict(attributes))

        return parent_attributes


class LawsTransformedImage(TransformedImage):
    def __init__(
            self,
            laws_kernel: Optional[str, List[str]] = None,
            delta_parameter: Optional[int] = None,
            energy_map: Optional[bool] = None,
            rotation_invariance: Optional[bool] = None,
            pooling_method: Optional[str] = None,
            boundary_condition: Optional[str] = None,
            riesz_order: Optional[int, List[int]] = None,
            riesz_steering: Optional[bool] = None,
            riesz_sigma_parameter: Optional[float] = None,
            template: Optional[GenericImage] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Filter parameters
        self.laws_kernel = laws_kernel
        self.delta_parameter = delta_parameter
        self.energy_map = energy_map
        self.rotation_invariance = rotation_invariance
        self.pooling_method = pooling_method
        self.boundary_condition = boundary_condition
        self.riesz_transformed = riesz_order is not None
        self.riesz_order = copy.deepcopy(riesz_order)
        self.riesz_steering = riesz_steering
        self.riesz_sigma_parameter = riesz_sigma_parameter

        # Update image parameters using the template.
        if isinstance(template, GenericImage):
            self.update_from_template(template=template)

    def get_file_name_descriptor(self) -> List[str]:
        descriptors = super().get_file_name_descriptor()

        descriptors += ["laws", self.laws_kernel]
        if self.energy_map:
            descriptors += ["energy", "delta", str(self.delta_parameter)]
        if self.rotation_invariance:
            descriptors += ["invar"]

        return descriptors

    def get_export_attributes(self) -> Dict[Any]:
        parent_attributes = super().get_export_attributes()

        attributes = [
            ("filter_type", "laplacian_of_gaussian"),
            ("laws_kernel", self.laws_kernel),
            ("energy_map", self.energy_map),
            ("rotation_invariance", self.rotation_invariance),
            ("boundary_condition", self.boundary_condition)
        ]

        if self.energy_map:
            attributes += [("delta_parameter", self.delta_parameter)]

        if self.pooling_method is not None:
            attributes += [("pooling_method", self.pooling_method)]

        if self.riesz_transformed:
            attributes += [("riesz_order", self.riesz_order)]

            if self.riesz_steering:
                attributes += [("riesz_sigma_parameter", self.riesz_sigma_parameter)]

        parent_attributes.update(dict(attributes))

        return parent_attributes
