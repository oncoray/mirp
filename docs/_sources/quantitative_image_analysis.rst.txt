Process image and compute quantitative image features
=====================================================

Two of the main uses for MIRP are to process images and compute quantitative features from images. Both use the same
standardized workflow that is compliant with the Image Biomarker Standardisation Initiative (IBSI) reference standards
[Zwanenburg2020]_, [Whybra2024]_.  Two versions of the image processing and feature computation function exist:

* :func:`~mirp.extract_features_and_images.extract_features_and_images`: conventional function that processes images and
  computes features.
* :func:`~mirp.extract_features_and_images.extract_features_and_images_generator`: generator that yields processed
  images and features computed therefrom.

For convenience, the above functions are wrapped to allow for only computing feature values (without exporting
images) and only processing images (without computing features):

* :func:`~mirp.extract_features_and_images.extract_features`: conventional function that only computes features.
* :func:`~mirp.extract_features_and_images.extract_features_generator`: generator that only yields feature values.
* :func:`~mirp.extract_features_and_images.extract_images`: conventional function that only processes images.
* :func:`~mirp.extract_features_and_images.extract_images_generator`: generator that yields processed images.

Examples
--------

MIRP can compute features from regions of interest in images. The features are described in [Zwanenburg2016]_.

Minimal example
^^^^^^^^^^^^^^^

The following computes features from a single image and mask:

.. code-block:: python

    from mirp import extract_features

    feature_data = extract_features(
        image="path to image",
        mask="path to mask",
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=32
    )

The ``base_discretisation_method`` and its corresponding parameters are required as long as any texture or
intensity-histogram features are involved.

Interpolation example
^^^^^^^^^^^^^^^^^^^^^

A more realistic example involves interpolation to ensure that voxel spacing is the same for all images in a dataset.
For example, a positron emission tomography (PET) dataset may be resampled to 3 by 3 by 3 mm isotropic voxels. This
is achieved by providing the ``new_spacing`` argument, i.e. ``new_spacing=3.0`` or ``new_spacing=[3.0, 3.0, 3.0]``.

.. code-block:: python

    from mirp import extract_features

    feature_data = extract_features(
        image="path to PET image",
        mask="path to PET mask",
        image_modality="PET",
        new_spacing=3.0,
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=32
    )

Here, ``image_modality="PET"`` is used to declare that the image is a PET image. If this is a DICOM image, this
argument is not necessary -- the modality can be inferred from the metadata.

Slice-wise example
^^^^^^^^^^^^^^^^^^

Sometimes, in-plane resolution is much higher than axial resolution. For example, in (older) computed tomography (CT)
images, in-plane resolution may be 1 by 1 mm, but the distance between slices can be 7 mm or greater.
Resampling to isotropic 1 by 1 by 1 mm voxels causes considerable data to be inferred between slices,
which may not be desirable. In that case, images may be better processed by slice-by-slice (*2D*).
This is achieved by providing the ``by_slice`` argument, i.e. ``by_slice=True``.

.. code-block:: python

    from mirp import extract_features

    feature_data = extract_features(
        image="path to CT image",
        mask="path to CT mask",
        image_modality="CT",
        by_slice=True,
        new_spacing=1.0,
        base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=32
    )

In the above example ``new_spacing=1.0`` causes all images to be resampled in-plane to a 1 mm resolution.

Fixed Bin Number discretisation example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The previous examples used the *Fixed Bin Number* to discretise intensities within the mask into a fixed number of bins.
For some imaging modalities, intensities carry a physical (or at least calibrated) meaning, such as Hounsfield units in
computed tomography and standardised uptake values in positron emission tomography. For these *Fixed Bin Size* (also
known as *Fixed Bin Width*) can be interesting, as this creates a mapping between intensities and bins that is
consistent across the dataset. MIRP sets the lower bound of the initial bin using the resegmentation range, or in
its absence, a default value (if any).

Below we compute features from a computed tomography image using a *Fixed Bin Size* discretisation method.
Because the resegmentation range is not set, the lower bound of the initial bin defaults to -1000 Hounsfield Units.

.. code-block:: python

    from mirp import extract_features

    feature_data = extract_features(
        image="path to CT image",
        mask="path to CT mask",
        image_modality="CT",
        new_spacing=1.0,
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=25.0
    )

Mask resegmentation example
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the region of interest contained in the mask in the above example covers soft tissue, this default might not be good.
We can change this by providing the ``resegmentation_intensity_range`` argument. Here, we provide a window more fitting
for soft tissues: ``resegmentation_intensity_range=[-200.0, 200.0]``. Thus the lower bound of the initial bin is set to
-200 Hounsfield Units, and 16 bins total are formed.

.. code-block:: python

    from mirp import extract_features

    feature_data = extract_features(
        image="path to CT image",
        mask="path to CT mask",
        image_modality="CT",
        new_spacing=1.0,
        resegmentation_intensity_range=[-200.0, 200.0],
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=25.0
    )

Basic image filter example
^^^^^^^^^^^^^^^^^^^^^^^^^^

The above examples all compute features from the base image. Filters can be applied to images to enhance patterns such
as edges. MIRP implements multiple filters [Depeursinge2020]_. In the example below, we compute features from a
Laplacian-of-Gaussian filtered image:

.. code-block:: python

    from mirp import extract_features

    feature_data = extract_features(
        image="path to image",
        mask="path to mask",
        new_spacing=1.0,
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=25.0,
        filter_kernels="laplacian_of_gaussian",
        laplacian_of_gaussian_sigma=2.0
    )

Image filter with additional features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, only statistical features are computed from filtered images, and features are still extracted from the
base image. You can change this by specifying ``base_feature_families="none"`` (to prevent computing features from
the base image) and specifying ``response_map_feature_families``. In the example below, we compute both statistical
features and intensity histogram features.

.. code-block:: python

    from mirp import extract_features

    feature_data = extract_features(
        image="path to image",
        mask="path to mask",
        new_spacing=1.0,
        base_feature_families="none",
        response_map_feature_families=["statistics", "intensity_histogram"],
        filter_kernels="laplacian_of_gaussian",
        laplacian_of_gaussian_sigma=2.0
    )

Even though intensity histogram features require discretisation, you don't have to provide a discretisation method
and associated parameters. This is because for many filters, intensities in the filtered images no longer represent a
measurable quantity such as Hounsfield Units. Hence a *Fixed Bin Number* algorithm is used by default, with 16 bins.
These parameters can be changed using the ``response_map_discretisation_method`` and
``response_map_discretisation_n_bins`` arguments.

API documentation
-----------------

.. autofunction:: mirp.extract_features_and_images.extract_features_and_images

.. autofunction:: mirp.extract_features_and_images.extract_features_and_images_generator

.. autofunction:: mirp.extract_features_and_images.extract_features

.. autofunction:: mirp.extract_features_and_images.extract_features_generator

.. autofunction:: mirp.extract_features_and_images.extract_images

.. autofunction:: mirp.extract_features_and_images.extract_images_generator

References
----------
.. [Depeursinge2020] Depeursinge A, Andrearczyk V, Whybra P, van Griethuysen J, Mueller H, Schaer R, et al.
  Standardised convolutional filtering for radiomics. arXiv [eess.IV]. 2020. doi:10.48550/arXiv.2006.05470

.. [Whybra2024] Whybra P, Zwanenburg A, Andrearczyk V, Schaer R, Apte AP, Ayotte A, et al. The Image Biomarker
  Standardization Initiative: Standardized Convolutional Filters for Reproducible Radiomics and Enhanced Clinical
  Insights. Radiology. 2024;310: e231319. doi:10.1148/radiol.231319

.. [Zwanenburg2016] Zwanenburg A, Leger S, Vallieres M, Loeck S. Image Biomarker Standardisation Initiative. arXiv
  [cs.CV] 2016. doi:10.48550/arXiv.1612.070035

.. [Zwanenburg2020] Zwanenburg A, Vallieres M, Abdalah MA, Aerts HJWL, Andrearczyk V, Apte A, et al. The Image
  Biomarker Standardization Initiative: Standardized Quantitative Radiomics for High-Throughput Image-based
  Phenotyping. Radiology. 2020;295: 328-338. doi:10.1148/radiol.2020191145
