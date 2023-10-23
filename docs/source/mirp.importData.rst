Configuring image and mask import
=================================

Most relevant MIRP functions require images, masks or both as input. MIRP is flexible when it comes to input:

* By specifying the directory where images and/or masks are found:
    * **Nested flat layout**: In a nested flat layout all images and masks are separated for each sample. For
      example, an image dataset of 128 samples may be organised as follows::

        image_root_directory
        ├─ sample_001
        │   └─ ...
        ├─ ...
        └─ sample_127
            └─ image_sub_folder
            ├─ CT_dicom_000.dcm
            ├─ ...
            └─ CT_dicom_255.dcm
            └─ mask.dcm

      Images and mask files are directly under the sample directory. Only one keyword argument is required:

      .. code-block:: python

          some_function(
              ...,
              image = "image_root_directory",
              ...
          )

      MIRP is generally able to determine which files are images and which files are masks. However, there may be
      cases where MIRP is unable to determine if a file is an image or a mask. In those cases, additional keyword
      arguments may be provided:

      .. code-block:: python

          some_function(
              ...,
              image = "image_root_directory",
              image_name = "CT_dicom_*",
              mask_name = "mask"
              ...
          )

      Here, ``image_name`` and ``mask_name`` contain patterns for image and mask files, respectively. ``"CT_dicom_*"``
      contains a wildcard character (``*``) that matches any pattern starting with ``"CT_dicom_"``. File extensions are
      never of the pattern.

    * **Fully nested structure**: In a nested structure all images and masks are separated for each. Unlike the above
      example, image and mask data may be organised into different subdirectory structures::

        image_root_directory
        ├─ sample_001
        │   └─ image_sub_folder
        │   │   └─ ...
        │   └─ mask_sub_folder
        │       └─ ...
        ├─ ...
        └─ sample_127
            └─ image_sub_folder
            │   ├─ CT_dicom_000.dcm
            │   ├─ ...
            │   └─ CT_dicom_255.dcm
            └─ mask_sub_folder
                └─ mask.dcm

      Here the directory for each sample contains consistently named subdirectory structures (``image_sub_folder``
      and ``mask_sub_folder``), that contains the set of DICOM images and a mask, respectively. Then the following
      keyword arguments may be specified:

      .. code-block:: python

          some_function(
              ...,
              image = "image_root_directory",
              image_sub_folder = "image_sub_folder",
              mask_sub_folder = "mask_sub_folder",
              ...
          )

      The ``mask`` keyword argument is automatically assumed to be equal to ``image``, i.e. images and masks are
      under the same root directory. If this is not the case, ``mask`` should be specified as well.

    * **Flat layout**: In a flat layout, all image and mask files are contained in the same
      directory::

        image_root_directory
            ├─ sample_001_CT_dicom_000.dcm
            ├─ ...
            ├─ sample_001_CT_dicom_319.dcm
            ├─ sample_127_CT_dicom_000.dcm
            ├─ ...
            ├─ sample_127_CT_dicom_255.dcm
            ├─ sample_001_mask.dcm
            ├─ ...
            └─ sample_127_mask.dcm

      Flat layouts are somewhat more challenging for MIRP, as sample identifiers have to be inferred, and images and
      masks may be hard to associate. For DICOM images sample names and other association data typically can be
      obtained from the DICOM metadata. For other types of images, e.g. NIfTI or numpy, in a flat layout,
      ``image_name`` and ``mask_name`` should be provided:

      .. code-block:: python

          some_function(
              ...,
              image = "image_root_directory",
              image_name = "#_CT_dicom_*",
              mask_name = "#_mask",
              ...
          )

      The above example contain two wildcards: ``#`` and ``*`` that fulfill different roles. While ``*`` matches any
      pattern, ``#`` matches any pattern and uses that pattern as the sample name. This way, sample identifiers can
      be determined for flat layouts.

* By providing a direct path to image and mask files:
    * **Single image and mask**: A path to an image and mask may be provided as follows:

      .. code-block:: python

          some_function(
              ...,
              image = "image_directory/image.nii.gz",
              mask = "mask_directory/mask.nii.gz",
              ...
          )

      Here ``"image.nii.gz"`` is an image file in NIfTI format, located in the ``"image_directory"`` directory.
      Similarly, ``"mask.nii.gz"`` is a mask file (containing integer-value labels) that is located in the
      ``"mask_directory"`` directory.

    * **Multiple images and masks**: Multiple images and masks can be provided as lists.

      .. code-block:: python

          some_function(
              ...,
              image = ["image_directory/image_001.nii.gz", "image_directory/image_002.nii.gz"],
              mask = ["mask_directory_001/mask.nii.gz", "mask_directory_002/mask.nii.gz"],
              ...
          )

      .. note::
        It is possible to provide multiple masks for each image as long as their is some way to associate the image
        with its masks, e.g. on sample name or frame of reference.

      .. note::
        In absence of any further identifiers for associating images and masks, MIRP will treat image and mask lists of
        equal length as being sorted by element, and associate the first mask with the first image, the second mask
        with the second image, and so forth.

* By providing the image and mask directly:
    Images and masks can be provided directly using ``numpy.ndarray`` objects.

    .. warning::
      Even though images can be directly provided as ``numpy`` arrays, this should only be done if all data has
      the same (physical) resolution, or if physical resolution does not matter. This is because ``numpy`` arrays only
      contain values, and no metadata concerning pixel or voxel spacing. Internally, MIRP will use a default value of
      1.0 × 1.0 × 1.0.

    * **Single image and mask**: Let ``numpy_image`` and ``numpy_mask`` be two ``numpy`` arrays with the same
      dimension. Then, these objects can be provided as follows:

      .. code-block:: python

          some_function(
              ...,
              image = numpy_image,
              mask = numpy_mask,
              ...
          )

    * **Multiple images and masks**: Multiple images and masks can be provided as lists of ``numpy`` arrays:

      .. code-block:: python

          some_function(
              ...,
              image = [numpy_image_001, numpy_image_002]
              mask = [numpy_mask_001, numpy_mask_002],
              ...
          )
      .. warning::
        While it is possible to provide multiple masks for each image, in practice there is no safe way to do so. The
        only way to associate image and masks is by their image dimension, which may be the same for different images.
        with its masks, e.g. on sample name or frame of reference. Hence, providing one mask per image is recommended.
        MIRP will treat image and mask lists of equal length as being sorted by element, and associate the first mask
        with the first image, the second mask with the second image, and so forth.

* By specifying the configuration in a stand-alone settings ``xml`` file. An empty copy of the ``xml`` file can be
  created using :func:`mirp.utilities.config_utilities.get_data_xml`. The tags of the``xml`` file are the same as the
  arguments of :func:`~mirp.importData.importImageAndMask.import_image_and_mask`.

Image and mask import
---------------------

.. note:: The :func:`~mirp.importData.importImageAndMask.import_image_and_mask` function is called internally by other
  functions. These function pass through keyword arguments to
  :func:`~mirp.importData.importImageAndMask.import_image_and_mask`.

.. autofunction:: mirp.importData.importImageAndMask.import_image_and_mask


Creating a data xml file
----------------------------

.. autofunction:: mirp.utilities.config_utilities.get_data_xml
