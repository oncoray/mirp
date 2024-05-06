Contributing
============

MIRP is open-source software, hosted on `GitHub <https://github.com/oncoray/mirp>`_. Contributions that enable
new DICOM modalities are especially welcome! If you have ideas or code to contribute, please first open an
`issue <https://github.com/oncoray/mirp/issues>`_ and describe your ideas.

To help you get an overview of how MIRP is structured internally, we describe the overall design of MIRP here:
:doc:`../design`

Everyone likes high-quality and easy-to-maintain code. Though nobody writes perfect code from scratch, the following can
help you make useful and enduring contributions to MIRP:

* MIRP styles its code according to `PEP8 <https://peps.python.org/pep-0008/>`_, but allows for longer line lengths
  (120 characters). Using a linter or IDE with built-in linter may help stick to PEP8.
* Testing your code enables discovering if it actually works as you intend it to. We wrote a short guide on how tests
  are performed in MIRP: :doc:`../contributing_tests`.
* If you are contributing code that should become part of the public API (see :doc:`../design`), you should document how the
  user can use that functionality. In MIRP, functions, classes and methods are documented using Numpy-flavoured
  `docstrings <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
  `Long-form documentation <https://oncoray.github.io/mirp/>`_ is partially created from such
  documentation, embedded in restructured text files in the ``docs_source`` directory.
  Providing examples or even a tutorial to highlight new functionality can go a long way to help users use your
  contribution.
* Generally, please ensure that you use descriptive variable, function, class, etc., names in your contributions.
  If your code does something that is not readily apparent from reading it, please comment it.
  For longer pieces of code, commenting the main steps is also helpful for understanding the code. Your aim should be
  to write code you will be able to understand a year or more into the future.
* Because MIRP serves an international audience, your contributions should be in English.
