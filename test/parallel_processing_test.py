import os

import numpy as np

from mirp import extract_features_and_images_generator
from mirp.extract_features_and_images import extract_features_and_images
from mirp.deep_learning_preprocessing import deep_learning_preprocessing, deep_learning_preprocessing_generator
from mirp.utilities.parallel import cluster_exists

# Find path to the test directory. This is because we need to read datafiles stored in subdirectories.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_parallel_feature_extraction():
    sequential_data = extract_features_and_images(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        perturbation_rotation_angles=[-5.0, 0.0, 5.0],
        base_feature_families="statistics",
        resegmentation_intensity_range=[-1000.0, 250.0]
    )

    assert len(sequential_data) == 3

    for parallel_backend in ["ray", "joblib"]:
        parallel_data = extract_features_and_images(
            num_cpus=2,
            parallel_backend=parallel_backend,
            write_features=False,
            export_features=True,
            write_images=False,
            export_images=False,
            image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
            mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
            roi_name="GTV-1",
            perturbation_rotation_angles=[-5.0, 0.0, 5.0],
            base_feature_families="statistics",
            resegmentation_intensity_range=[-1000.0, 250.0]
        )

        assert not cluster_exists(backend=parallel_backend)
        assert len(parallel_data) == 3

        for ii in range(len(sequential_data)):
            assert sequential_data[ii].equals(parallel_data[ii])


def test_parallel_feature_extraction_generator():
    sequential_data = []
    for processed_data in extract_features_and_images_generator(
        write_features=False,
        export_features=True,
        write_images=False,
        export_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
        roi_name="GTV-1",
        perturbation_rotation_angles=[-5.0, 0.0, 5.0],
        base_feature_families="statistics",
        resegmentation_intensity_range=[-1000.0, 250.0]
    ):
        sequential_data.append(processed_data)

    assert len(sequential_data) == 3

    for parallel_backend in ["joblib"]:
        parallel_data = []
        for processed_data in extract_features_and_images_generator(
            num_cpus=2,
            parallel_backend=parallel_backend,
            write_features=False,
            export_features=True,
            write_images=False,
            export_images=False,
            image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
            mask=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "mask"),
            roi_name="GTV-1",
            perturbation_rotation_angles=[-5.0, 0.0, 5.0],
            base_feature_families="statistics",
            resegmentation_intensity_range=[-1000.0, 250.0]
        ):
            parallel_data.append(processed_data)

        assert not cluster_exists(backend=parallel_backend)
        assert len(parallel_data) == 3

        for ii in range(len(sequential_data)):
            assert sequential_data[ii].equals(parallel_data[ii])


def test_parallel_processing_image_crop():
    data = deep_learning_preprocessing(
        output_slices=False,
        crop_size=[20, 50, 50],
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        perturbation_rotation_angles=[-5.0, 0.0, 5.0]
    )

    assert len(data) == 3
    for dataset in data:
        assert dataset[0][0]["image"].shape == (20, 50, 50)
        assert dataset[1][0]["mask"].shape == (20, 50, 50)

    for parallel_backend in ["ray", "joblib"]:
        parallel_data = deep_learning_preprocessing(
            output_slices=False,
            crop_size=[20, 50, 50],
            export_images=True,
            write_images=False,
            num_cpus=2,
            parallel_backend=parallel_backend,
            image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
            perturbation_rotation_angles=[-5.0, 0.0, 5.0]
        )

        assert not cluster_exists(backend=parallel_backend)
        assert len(parallel_data) == 3

        for ii, dataset in enumerate(parallel_data):
            assert dataset[0][0]["image"].shape == (20, 50, 50)
            assert dataset[1][0]["mask"].shape == (20, 50, 50)
            assert np.array_equal(dataset[0][0]["image"], data[ii][0][0]["image"])


def test_parallel_processing_image_crop_generator():
    # Sequential generator.
    data = []
    for processed_data in deep_learning_preprocessing_generator(
        output_slices=False,
        crop_size=[20, 50, 50],
        export_images=True,
        write_images=False,
        image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
        perturbation_rotation_angles=[-5.0, 0.0, 5.0]
    ):
        assert processed_data[0][0]["image"].shape == (20, 50, 50)
        assert processed_data[1][0]["mask"].shape == (20, 50, 50)
        data.append(processed_data)

    assert len(data) == 3

    # Ray is not available as a backend because this doesn't speed up the generator.
    for parallel_backend in ["joblib"]:
        parallel_data = []
        for processed_data in deep_learning_preprocessing_generator(
                output_slices=False,
                crop_size=[20, 50, 50],
                export_images=True,
                write_images=False,
                num_cpus=2,
                parallel_backend=parallel_backend,
                image=os.path.join(CURRENT_DIR, "data", "ibsi_1_ct_radiomics_phantom", "dicom", "image"),
                perturbation_rotation_angles=[-5.0, 0.0, 5.0]
        ):
            assert processed_data[0][0]["image"].shape == (20, 50, 50)
            assert processed_data[1][0]["mask"].shape == (20, 50, 50)
            parallel_data.append(processed_data)

        assert len(parallel_data) == 3
        assert not cluster_exists(backend=parallel_backend)

        for ii, dataset in enumerate(parallel_data):
            assert dataset[0][0]["image"].shape == (20, 50, 50)
            assert dataset[1][0]["mask"].shape == (20, 50, 50)
            assert np.array_equal(dataset[0][0]["image"], data[ii][0][0]["image"])


def test_limit_threads():
    from mirp.utilities.parallel_ray import limit_inner_threads

    limit_inner_threads()

    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["BLIS_NUM_THREADS"] == "1"
    assert os.environ["VECLIB_MAXIMUM_THREADS"] == "1"
    assert os.environ["NUMBA_NUM_THREADS"] == "1"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "1"
