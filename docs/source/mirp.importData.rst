Configuring image and mask import
=================================

Many relevant MIRP functions require images, masks or both as input. This section provides details on how image and
mask import is configured.

Specifying input
----------------

MIRP processes and analyses images and masks. There are multiple ways to provide images and masks:

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

      .. note::
        MIRP will interpret the name of the directory that is neither part of the root directory or the subdirectory
        structures as the sample name, unless the sample name can be determined from metadata (i.e. DICOM files). In
        the example above, sample names based on the directory structure would be ``"sample_001"`` to ``"sample_127"``.

    * **Flat layout**: In a flat layout, all image and mask files are contained in the same directory::

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

* By specifying the configuration in a stand-alone data ``xml`` file. An empty copy of the ``xml`` file can be
  created using :func:`mirp.utilities.config_utilities.get_data_xml`. The tags of the``xml`` file are the same as the
  arguments of :func:`~mirp.importData.importImageAndMask.import_image_and_mask`, that are listed below.

Selecting specific images and masks
-----------------------------------
On occasion, input should be more selective. This can be done by specifying additional arguments:

* Select specific samples using ``sample_name``:
    Sample names can be provided as a list of strings to filter images and masks and exclude those that do not appear
    in the provided list.

    .. note::
      If sample names cannot be determined from metadata, directory structure or file names, MIRP cannot filter image
      and mask files using the provided sample names. In this case, should the list of provided sample names equal
      that of the images, the provided sample names will be associated one-to-one with images. Otherwise, MIRP will
      randomly generate sample names.

* Select specific image and mask files based on their file names using ``image_name`` and ``mask_name``:
    MIRP can filter image and mask files based on file names. ``image_name`` and ``mask_name`` arguments each take a
    single string as argument. This string is matched exactly, and only file names that match that string are selected.
    File extensions are ignored.

    To allow for some flexibility, wildcard characters can be used. MIRP recognises two types of wildcard characters:
    ``*`` and ``#``. ``*`` denotes any character. For example, if files are named ``image_001.nii.gz``,
    ``image_002.nii.gz`` and ``another_image_001.nii.gz``, using ``image_name="image_*"`` will select
    ``image_001.nii.gz``, ``image_002.nii.gz``. Using ``image_name="*image_*"`` will select all three.

    The other wildcard character (``#``) denotes the part of the file name that is the sample name. For example, if
    files are named ``sample_001_image_001.nii.gz``, ``sample_001_image_002.nii.gz`` and
    ``sample_002_image_001.nii.gz``, using ``image_name="#_image_*"`` will select all three files, and assign
    ``sample_001``, ``sample_001`` and ``sample_002`` as sample names, respectively.

    The ``mask_name`` argument functions exactly the same as ``image_name``.

* Select the image and mask file types using ``image_file_type`` and ``mask_file_type``:
    MIRP can filter image and mask files based on the file type. MIRP currently supports DICOM (``"dicom"``), NIfTI
    (``"nifti"``), NRRD (``"nrrd"``) and numpy (``"numpy"``) files as file format.

* Select image files based on image modality using ``image_modality``:
    MIRP can filter image files based on the image modality. Aside from generic image modality, MIRP specifically
    checks for the following modalities:
    * Computed tomography (CT): ``"ct"``
    * Positron emission tomography (PET): ``"pet"`` or ``"pt"``
    * Magnetic resonance imaging (MRI): ``"mri"`` or ``"mr"``

    Images from other modalities are currently not fully supported, and a default ``"generic"`` image modality will
    be assigned.

    .. note::
        Image modality is important because it adapts the image processing workflow to the requirements and
        possibilities of each modality. For example, bias-field correction can only be performed on MR imaging, and
        Hounsfield units are automatically rounded for CT imaging.

    .. warning::
        Only DICOM images contain metadata concerning image modality. Images from other file types are interpreted as
        ``"generic"`` by default and cannot be filtered using ``image_modality``. For these image, the
        ``image_modality`` argument sets the actual image modality.

* Select mask files based on mask modality using ``mask_modality``:
    MIRP can filter mask files based on the modality of the mask. Aside form generic masks, MIRP specifically checks for
    radiotherapy structure (RTSTRUCT) files.

    .. note::
        Only DICOM images contain metadata concerning mask modality. Masks from other file types are interpreted as
        ``"generic_mask"`` by default and cannot be filtered using ``mask_modality``.

    .. note::
        Support for DICOM segmentation (SEG) files is being implemented.

* Select the specific regions of interest using ``roi_name``:
    A mask file may contain multiple masks. By default, MIRP will assess all masks in a file. The ``roi_name`` argument
    can be used to specify the list of regions of interest that should be assessed. For DICOM mask files, names of
    regions of interest are provided in the metadata. For other mask file types, masks are either boolean, or
    non-negative integers. For these, ``False`` or ``0`` are interpreted as background, and not assessed. If, for
    example, regions of interest are labelled with ``1``, ``2`` and ``3``, MIRP will recognize both
    ``roi_name=["1", "2", "3"]`` and ``roi_name=["region_1", "region_2", "region_3"]``.

    You can use the :func:`~mirp.extractMaskLabels.extract_mask_labels` function to identify the names of the regions
    of interest in mask files.

Image and mask import function arguments
----------------------------------------

.. note:: The :func:`~mirp.importData.importImageAndMask.import_image_and_mask` function is called internally by other
  functions. These function pass through keyword arguments to
  :func:`~mirp.importData.importImageAndMask.import_image_and_mask`.

.. autofunction:: mirp.importData.importImageAndMask.import_image_and_mask


Creating a data xml file
----------------------------

.. autofunction:: mirp.utilities.config_utilities.get_data_xml
