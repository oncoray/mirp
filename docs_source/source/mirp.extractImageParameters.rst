Extract image metadata
======================

Image metadata, such as acquisition and reconstruction parameters, are interesting to report. To facilitate their
reporting, MIRP can automatically extract relevant parameters from metadata.

.. note::
    Many relevant parameters can only extracted from DICOM files, because other file types lack the
    corresponding metadata.

Example
-------

Parameters of a single image can be extracted from their metadata as follows:

.. code-block:: python

    from mirp import extract_image_parameters

    image_parameters = extract_image_parameters(
        image="path to image"
    )

API documentation
-----------------

.. automodule:: mirp.extractImageParameters
   :members:
   :undoc-members:
   :show-inheritance:
