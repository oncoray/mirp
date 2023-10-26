Process image and compute quantitative image features
=====================================================

Two of the main uses for MIRP are to process images and compute quantitative features from images. Both use the same
standardized, IBSI 1 and IBSI 2 compliant, workflow. Two versions of the image processing and feature computation
function exist:

* :func:`~mirp.extractFeaturesAndImages.extract_features_and_images`: conventional function that processes images and
  computes features.
* :func:`~mirp.extractFeaturesAndImages.extract_features_and_images_generator`: generator that yields processed
  images and features computed therefrom.

For convenience, the above functions are wrapped to allow for only computing feature values (without exporting
images) and only processing images (without computing features):

* :func:`~mirp.extractFeaturesAndImages.extract_features`: conventional function that only computes features.
* :func:`~mirp.extractFeaturesAndImages.extract_features_generator`: generator that only yields feature values.
* :func:`~mirp.extractFeaturesAndImages.extract_features_and_images`: conventional function that only processes images.
* :func:`~mirp.extractFeaturesAndImages.extract_features_and_images_generator`: generator that yields processed images.

Examples
--------

MIRP can compute features from regions of interest in images. The simplest example is:

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
        new_spacing=1.0,
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=25
    )

If the region of interest contained in the mask in the above example covers soft tissue, this default might not be good.
We can change this by providing the ``resegmentation_intensity_range`` argument. Here, we provide a window more fitting
for soft tissues: ``resegmentation_intensity_range=[-200.0, 200.0]``. Thus the lower bound of the initial bin is set to
-200 Hounsfield Units, and 16 bins total are formed.

.. code-block:: python

    from mirp import extract_features

    feature_data = extract_features(
        image="path to CT image",
        mask="path to CT mask",
        resegmentation_intensity_range=[-200.0, 200.0]
        new_spacing=1.0,
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=25
    )

TODO: Example using filter.

API documentation
-----------------

.. autofunction:: mirp.extractFeaturesAndImages.extract_features_and_images

.. autofunction:: mirp.extractFeaturesAndImages.extract_features_and_images_generator

.. autofunction:: mirp.extractFeaturesAndImages.extract_features

.. autofunction:: mirp.extractFeaturesAndImages.extract_features_generator

.. autofunction:: mirp.extractFeaturesAndImages.extract_images

.. autofunction:: mirp.extractFeaturesAndImages.extract_images_generator

