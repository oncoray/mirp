Quick-start
===========

Before you begin, you need to:
    1. Install MIRP (see :ref:`installation`).
    2. Have a dataset with imaging and corresponding masks.

Computing quantitative features
-------------------------------
Suppose you have a dataset of computed tomography (CT) DICOM images with corresponding segmentations. Suppose that both
images and masks are located split by patient directories ``path/to/data``. For each patient, the CT image is in the
``image`` directory, and its corresponding segmentation in ``mask``. For patient ``patient_003``, the full path to the
image directory is ``path/to/data/patient_003/image``, and to the mask directory is ``path/to/data/patient_003/mask``.

We want to compute features from the gross tumour mask (called ``GTV``). We are interested in the soft-tissue range,
with Hounsfield Units between -150 and 200 HU. To harmonise differences in resolution and slice distance
between CT images from different patients, all voxels are resampled to a 1.0 by 1.0 by 1.0 mm size. Histogram and
texture features are computed after discretisation using the `fixed bin size` method with a bin size of 25 Hounsfield
Units.

.. code-block:: python

    import pandas as pd
    from mirp import extract_features

    feature_data = extract_features(
        image="path/to/data",
        mask="path/to/data",
        image_sub_folder="image",
        mask_sub_folder="mask",
        roi_name="GTV",
        new_spacing=1.0,
        resegmentation_intensity_range=[-150.0, 200.0],
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=25.0
    )

The above code results in ``feature_data``, a list of ``pandas.DataFrame`` that contains feature values for every
patient. These can combined into a single ``pandas.DataFrame`` as follows:

.. code-block:: python

    feature_data = pd.concat(feature_data)

Computing quantitative features from filtered images
----------------------------------------------------

