Image preprocessing for deep learning
=====================================

MIRP can be used to preprocess images for deep learning. Images are processed using the standard IBSI-compliant image
processing workflow, with a final cropping step (if any).

The deep learning preprocessing function comes in two versions:

* :func:`~mirp.deepLearningPreprocessing.deep_learning_preprocessing`: conventional function that processes images.
* :func:`~mirp.deepLearningPreprocessing.deep_learning_preprocessing_generator`: generator that yields processed images.

.. automodule:: mirp.deepLearningPreprocessing
   :members:
   :undoc-members:
   :show-inheritance:
