General design
==============

The overall design of MIRP is divided into three layers, as shown in the figure below.

.. image:: images/mirp_general_design.svg
   :align: left

The most visible layer to the user is formed by the functions that are part of the public API. These functions, such as
`extract_features` and `extract_mask_labels`, form entry points that revolve around specific tasks.

The second layer is still public, but rarely directly addressed by users. This layer consists of import routines for
images and masks, as well as settings. The functions from the first, fully public layer, pass arguments to these
functions. Internally, these functions create objects that are then used in the calling function.

* `import_images` creates `ImageFile` objects, or subclasses thereof. These are found in the `mirp._data_import` module.
* `import_masks` creates `MaskFile` objects, or subclasses thereof. Like `ImageFile` (from which `MaskFile` inherits),
  these objects are defined in the `mirp._data_import` module.
* `import_images_and_masks` creates both `ImageFile` and `MaskFile` objects (or subclasses thereof).
  `import_images_and_masks` also associates `ImageFile` objects with their corresponding `MaskFile` objects.
* `import_configuration_settings` creates `SettingsClass` objects, which itself contains several underlying objects for
  configuring various steps in workflows (more on workflows below). These object classes are defined in the
  `mirp.settings` module.

The third layer is fully abstracted from the user. `deep_learning_preprocessing` and `extract_features`
(and similar functions) all work by first determining which data to load (`import_images_and_masks`) and how to process
them (`import_configuration_settings`). Based on the data and processing parameters, a workflow object
(`StandardWorkflow`) is created for each image with its associated masks. Depending on processing parameters
(e.g. multiple rotations) multiple workflow objects may be created instead. Each workflow defines a single experiment,
containing the relevant parameters and with a specific imaging dataset to import and process.

After creating workflow objects, `deep_learning_preprocessing` calls their `deep_learning_conversion` methods, whereas
`extract_features` and co. call their `standard_extraction` methods. Internally, both first access the
`standard_image_processing` generator method, which performs image processing according to a pipeline that is compliant
with the Image Biomarker Standardisation Initiative. This pipeline starts by loading the image and its mask(s) using
`read_images_and_masks`. It then converts them to their internal representations: GenericImage (and subclasses) and
BaseMask objects, respectively. `standard_image_processing` then relies on methods of these objects for further
image and mask processing. Finally, if filter is to be applied to an image, the workflow's `transform_images` method is
called.

After yielding the processed (and transformed) images, the `standard_image_processing` generator stops. The
`deep_learning_conversion` method then performs some final processing of the yielded images and masks, notably cropping
to the desired output format, if specified. `extract_images` directly yields the processed images and masks.
`extract_features` and `extract_features_and_images` do a bit more work. Features are computed from each image and each
associated mask using the workflow's `_compute_radiomics_features` method.

Submodules
----------

MIRP contains several submodules. The following submodules are part of the public API:

* `data_import`: Contains the `import_image`, `import_mask` and `import_image_and_mask` functions that are used for
  organising the available image and mask data.
* `settings`: Contains functions and class definitions related to configuring workflows, e.g. image processing and
  feature computation.
* `utilities`: Contains various utility functions.

The bulk of MIRPs functionality is located in private submodules:

* `_data_import`: Contains classes for image and mask files and modalities.
* `_featuresets`: Contains functions and classes involved in computing features.
* `_image_processing`: Contains functions that help process images. Most of these internally use methods associated
  with `GenericImage` (and subclasses) as well as `BaseMask` that are defined in the `_images` and `_masks` submodules
  respectively.
* `_imagefilters`: Contains classes for various convolutional filters and function transformations.
* `_images`: Contains classes that form the internal image representation, divided by image modality.
* `_masks`: Contains classes that form the internal mask representation.
* `_workflows`: Contains workflow-related class definitions, importantly the `StandardWorkflow` class that facilitates
  image and mask processing and feature computation.

Features
--------
Feature computation is called from `StandardWorkflow._compute_radiomics_features`. At the moment feature computation is
mostly functional, and organised by feature family.

Future directions
^^^^^^^^^^^^^^^^^
Feature computation is currently family-based, i.e. an entire family of features is computed at once. It may be
preferable to move to a more directive-based approach by treating feature as an object. There are no current direct
benefits to doing so, and would make feature computation a bit harder to program to avoid doing unnecessary work. For
example, computation of a grey level co-occurrence matrix feature relies on two prior steps: discretisation and
computing the GLCM. This would mean that feature objects have to be sorted by discretisation required, and the set of
parameters.

A directive-based approach would have at least the following two advantages:

* Features can be exported with metadata, instead of just their values. These metadata could include their IBSI
  identifiers of the features, as well as processing-related metadata that are currently included in the feature name.
  This would make it easier to export structured reports of image data.

* Features can be defined that are not generated by default, e.g. different intensity-volume histogram features.

Filters
-------
All filters are implemented as objects, defined in the `_imagefilters` submodule. The filters themselves are accessed
in the `StandardWorkflow.transform_images` method, yielding specific transformed image objects (defined in
`_images.transformed_images`).

Future directions
^^^^^^^^^^^^^^^^^
We are generally happy with the current implementation of image filters. It is relatively straightforward to implement new
filters should there be a need.

Internal image representation
-----------------------------
All internal image representations derive from `_images.generic_image.GenericImage`, which implements general methods.
These objects are created by the `read_image` and `read_image_and_masks` functions, that process `ImageFile` objects by
first converting them to the internal format using `ImageFile.to_object` (or override methods of subclasses),
and then promoting them to the correct image modality-specific subclass using the `GenericImage.promote` method.

These modality-specific subclasses allow for implementing modality-specific processing steps and parameters. For example,
bias-field correction is only implemented for `MRImage` objects. As another example, subclasses such as `CTImage`
override the `get_default_lowest_intensity` method to provide modality-specific default values.

`MaskImage` also derives from  `GenericImage`, and is designed to contain mask information. Its implementation is
comparatively extensive because it contains or overrides methods that act upon masks specifically.
One notable aspect of `MaskImage` is that the mask data are typically run-length encoded, and only decoded upon use, to
provide better memory utilisation.

Future directions
^^^^^^^^^^^^^^^^^
The current implementation of internal image representations is sufficient. It is relatively straightforward to
implement objects for new image modalities, for which existing classes such as `CTImage` and `PETImage` can be used as
templates.

Some objects may receive additional attributes to represent relevant metadata on the value representations, e.g. the
type of SUV conversion used to create a `PETImage`.

In addition, all internal image representations are volumetric. They contain a merged stack of image slices. However,
in rare occasions, the original input data may contain image slices that are not equidistant, i.e. with variable slice
spacing. It is safer to handle DICOM imaging, prior to resampling (`interpolation` in
`StandardWorkflow.standard_image_processing`), as a stack of separate slices.

Internal mask representation
----------------------------
Masks are internally represented by `_masks.base_mask.BaseMask`. `BaseMask` objects are containers for the actual masks,
which are `_images.mask_image.MaskImage`. In fact, each `BaseMask` contains up to three variants of masks, notably the
original mask, the morphological mask and the intensity mask. Whereas the original mask and morphological mask are
currently direct

Future directions
^^^^^^^^^^^^^^^^^
The current implementation of internal image representations is sufficient.