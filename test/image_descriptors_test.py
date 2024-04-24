import numpy as np
import pytest

GENERIC_KWARGS = dict([
    ("image_data", np.ones((30, 29, 28), float)),
    ("separate_slices", False),
    ("translation", (1.0, 1.0, 1.0)),
    ("rotation_angle", 90.0),
    ("noise_iteration_id", 2),
    ("noise_level", 2.0),
    ("interpolated", True),
    ("interpolation_algorithm", "interpolationalgorithmtest"),
    ("discretisation_method", "discretisationmethodtest"),
    ("discretisation_bin_width", 3.0),
    ("discretisation_bin_number", 16),
    ("image_modality", "modalitytest"),
    ("image_spacing", (2.0, 2.0, 2.0)),
    ("image_origin", (-1.0, -1.0, -1.0)),
    ("image_dimensions", (30, 29, 28)),
    ("image_orientation", np.diag(np.ones(3, float))),
    ("sample_name", "samplenametest")
])


@pytest.mark.ci
def test_generic_image_descriptors():
    from mirp._images.generic_image import GenericImage

    image = GenericImage(**GENERIC_KWARGS)
    image.slice_id = -1

    descriptors = image.get_file_name_descriptor()
    assert isinstance(descriptors, list)

    attributes = image.get_export_attributes()
    assert isinstance(attributes, dict)


@pytest.mark.ci
def test_gabor_filtered_image_descriptors():
    from mirp._images.transformed_image import GaborTransformedImage

    image = GaborTransformedImage(
        sigma_parameter=1.0,
        gamma_parameter=2.0,
        lambda_parameter=3.0,
        theta_parameter=4.0,
        pool_theta=True,
        response_type="testresponsetype",
        rotation_invariance=True,
        pooling_method="testpoolingmethod",
        boundary_condition="testboundarycondition",
        riesz_order=1,
        riesz_steering=True,
        riesz_sigma_parameter=2.0,
        **GENERIC_KWARGS
    )
    descriptors = image.get_file_name_descriptor()
    assert isinstance(descriptors, list)

    attributes = image.get_export_attributes()
    assert isinstance(attributes, dict)
