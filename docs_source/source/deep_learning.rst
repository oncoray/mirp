Preprocess images for deep learning
===================================

MIRP can be used to preprocess images for deep learning. Images are processed using the standard image
processing workflow that is compliant with Image Biomarker Standardisation Initiative (IBSI), with a final cropping
step (if any).

The deep learning preprocessing function comes in two versions:

* :func:`~mirp.deep_learning_preprocessing.deep_learning_preprocessing`: conventional function that processes images.
* :func:`~mirp.deep_learning_preprocessing.deep_learning_preprocessing_generator`: generator that yields processed images.

Example
-------

MIRP can be used to crop images, e.g. to make them conform to the input of convolutional neural networks:

.. code-block:: python

    from mirp import deep_learning_preprocessing

    processed_data = deep_learning_preprocessing(
        image="path to image",
        mask="path to mask",
        crop_size=[50, 224, 224]
    )

API documentation
-----------------
.. automodule:: mirp.deep_learning_preprocessing
   :members:
   :undoc-members:
   :show-inheritance:
