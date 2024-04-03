Quick-start
===========

Before you begin, you need to:
    1. Install MIRP (see :ref:`installation`).
    2. Have a dataset with imaging and corresponding masks.

Computing quantitative features
-------------------------------
Suppose you have a dataset of computed tomography (CT) DICOM images with corresponding segmentation masks that you want
to use to compute quantitative features from. Now, suppose that both images and masks are seperated by patient
directories within a general ``path/to/data`` folder. For each patient, the CT image is in the ``image`` directory,
and its corresponding segmentation in ``mask``. For patient ``patient_003``, the full path to the
image directory is ``path/to/data/patient_003/image``, and to the mask directory is ``path/to/data/patient_003/mask``.

We want to compute features from a pre-defined gross tumour mask (called ``GTV``). We are interested in the soft-tissue
range, with Hounsfield Units between -150 and 200 HU. To harmonise differences in resolution and slice distance
between CT images from different patients, all voxels are resampled to a 1.0 by 1.0 by 1.0 mm size. Histogram and
texture features are computed after discretisation using the `fixed bin size` method with a bin size of 25 Hounsfield
Units.

MIRP can compute quantitative features using the function call below:

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

The above code results in ``feature_data`` which is a list of ``pandas.DataFrame`` that contains feature values for
every patient. These can combined into a single ``pandas.DataFrame`` as follows:

.. code-block:: python

    feature_data = pd.concat(feature_data)

Visualising filtered images
---------------------------
Image filters enhance aspects such as edges, blobs and directional structures. MIRP supports several filters (see
:ref:`quantitative_image_analysis`). Suppose you want to use a Laplacian-of-Gaussian filter, with the width of the
Gaussian equal to 2.0 mm.

We can first inspect the images visually using ``extract_images``. By default, ``export_images`` exports images and
masks as dictionary with ``numpy`` data and metadata (or as NIfTI files, in case ``write_dir`` is provided). These
can be used with external viewers, or your own scripts. MIRP also has a simple viewer for its internal native
image objects. To use this viewer, you can set ``image_export_format = "native"``.

.. code-block:: python

    from mirp import extract_images

    images = extract_images(
        image="path/to/data",
        mask="path/to/data",
        image_sub_folder="image",
        mask_sub_folder="mask",
        roi_name="GTV",
        new_spacing=1.0,
        resegmentation_intensity_range=[-150.0, 200.0],
        filter_kernels="laplacian_of_gaussian",
        laplacian_of_gaussian_sigma=2.0,
        image_export_format="native"
    )

Here, ``images`` is a list of images and masks, with one entry for each patient. Each entry consist of two nested
lists, one for images and the second for masks. In this case, the nested list of images contains two entries, and
that of masks only one (for the ``GTV`` region of interest). The first image is the CT image, after interpolation to
1.0 by 1.0 by 1.0 mm voxels. The second image is the Laplacian-of-Gaussian filtered image. Each image can be viewed
using the ``show`` method:

.. code-block:: python
    patient_1_images, patient_1_mask = images[0]
    patient_1_ct_image, patient_1_log_image = patient_1_images

    # View the CT image
    patient_1_ct_image.show()

    # View the Laplacian-of-Gaussian filtered image
    patient_1_log_image.show()

Computing quantitative features from filtered images
----------------------------------------------------

Of course, features can also be computed from filtered images (also known as response maps). By default, only
statistical features [Zwanenburg2016]_ are computed from filtered images.

.. code-block:: python

    feature_data = extract_features(
        image="path/to/data",
        mask="path/to/data",
        image_sub_folder="image",
        mask_sub_folder="mask",
        roi_name="GTV",
        new_spacing=1.0,
        resegmentation_intensity_range=[-150.0, 200.0],
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=25.0,
        filter_kernels="laplacian_of_gaussian",
        laplacian_of_gaussian_sigma=2.0
    )

    feature_data = pd.concat(feature_data)

``feature_data`` is a ``pandas.DataFrame`` similar to the first example, but with features computed from the
Laplacian-of-Gaussian image appended as new columns.

Processing images
-----------------

Computing quantitative features is nice, but what if you use deep learning instead? Suppose you just want to process
images as input for a VGG16 network [Simonyan2015]_. These networks have a default input size of 224 by 224 pixels.
For many deep learning applications, images should be provided 1-by-1, and we therefore will use a generator. By
providing the name of the region of interest, images will be cropped based on the center of the mask:

.. code-block:: python

    from mirp import deep_learning_preprocessing_generator

    image_generator = deep_learning_preprocessing_generator(
        image="path/to/data",
        mask="path/to/data",
        image_sub_folder="image",
        mask_sub_folder="mask",
        roi_name="GTV",
        new_spacing=1.0,
        crop_size=[224, 224],
        output_slices=True
    )

    image_slices, mask_slices = next(image_generator)

The generator yields a set of image slices (each 224 by 224 pixels) with corresponding masks. ``output_slices=False``
can be used to generate 3D volumes.

Extracting metadata
-------------------

DICOM files contains metadata that are relevant to report in studies. MIRP can extract and collect such metadata:

.. code-block:: python

    from mirp import extract_image_parameters

    image_parameters = extract_image_parameters(
        image="path/to/data",
        image_sub_folder="image"
    )

``image_parameters`` is a ``pandas.DataFrame`` that contains relevant parameters extracted from DICOM metadata, such
as image resolution, scanner type and vendor as well as modality-specific attributes such as tube voltage for CT.
Note that metadata for other file types (e.g. NIfTI) are considerably more limited.

References
----------

.. [Simonyan2015] Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv
  [cs.CV] 2014. doi:10.48550/arXiv.1409.1556
.. [Zwanenburg2016] Zwanenburg A, Leger S, Vallieres M, Loeck S. Image Biomarker Standardisation Initiative. arXiv
  [cs.CV] 2016. doi:10.48550/arXiv.1612.070035
