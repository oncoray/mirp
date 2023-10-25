Process image and compute quantitative image features
=====================================================

Two of the main uses for MIRP are to process images and compute quantitative features from images. Both use the same
standardized, IBSI 1 and IBSI 2 compliant workflow. Two versions of the image processing and feature computation
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

Example
-------

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

API documentation
-----------------

.. autofunction:: mirp.extractFeaturesAndImages.extract_features_and_images

.. autofunction:: mirp.extractFeaturesAndImages.extract_features_and_images_generator

.. autofunction:: mirp.extractFeaturesAndImages.extract_features

.. autofunction:: mirp.extractFeaturesAndImages.extract_features_generator

.. autofunction:: mirp.extractFeaturesAndImages.extract_images

.. autofunction:: mirp.extractFeaturesAndImages.extract_images_generator

