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

Parallel processing example
^^^^^^^^^^^^^^^^^^^^^^^^^^^

MIRP supports parallel processing using ``ray`` and ``joblib``. Using parallel processing, multiple images can be
processed at the same time. There two relevant parameters: ``num_cpus`` and ``parallel_backend``. ``num_cpus``
determines the number of workers that will be spawned. ``parallel_backend`` determines the backend using for parallel
processing, i.e. ``"ray"`` or ``"joblib"``.

In the example below, we extract features from images using 2 workers on a ``joblib`` backend.

.. code-block:: python

    from mirp import deep_learning_preprocessing

    feature_data = deep_learning_preprocessing(
        image="path to image",
        mask="path to mask",
        crop_size=[50, 224, 224]
        num_cpus=2,
        parallel_backend="joblib"
    )

``joblib`` can also be used within a generator context, i.e. with ``deep_learning_preprocessing_generator``, but ``ray`` cannot.
Both ``ray`` and ``joblib`` are optional dependencies of MIRP and need to be installed separately.

API documentation
-----------------
.. automodule:: mirp.deep_learning_preprocessing
   :members:
   :undoc-members:
   :show-inheritance:
