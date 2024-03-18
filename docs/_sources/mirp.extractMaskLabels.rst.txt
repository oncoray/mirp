Extract mask labels
===================

Mask files can contain labels for multiple regions of interest. You can use the
:func:`~mirp.extract_mask_labels.extract_mask_labels` function to obtain these labels.

Example
-------

Region of interest labels can be extract from mask files as follows:

.. code-block:: python

    from mirp import extract_mask_labels

    mask_labels = extract_mask_labels(
        mask="path to mask"
    )

API documentation
-----------------

.. automodule:: mirp.extract_mask_labels
   :members:
   :undoc-members:
   :show-inheritance:
