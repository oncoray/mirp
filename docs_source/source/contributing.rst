Contributing
============

MIRP is open-source software, hosted on `GitHub <https://github.com/oncoray/mirp>`_. Contributions that enable
new DICOM modalities are especially welcome! If you have ideas or code to contribute, please first open an
`issue <https://github.com/oncoray/mirp/issues>`_ and describe your ideas.

Technical documentation is forthcoming. For now, please keep the following in mind when contributing:

* The main branch of MIRP is protected. You can therefore make a pull request for your contribution to a
  development branch of the intended future version.
* If you introduce new functionality, this functionality should be tested as part of the tests in the ``test``
  directory. After implementation, please ensure that all tests complete without errors by running ``pytest`` from your
  IDE or console using ``python -m pytest test`` from the mirp main directory. Likewise existing automated tests should
  pass.
* Code is styled according to `PEP8 <https://peps.python.org/pep-0008/>`_. Using a linter or IDE with automated linter
  is recommended.
* Function, class and method documentation is done using Numpy-flavoured
  `docstrings <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
  `Long-form documentation <https://oncoray.github.io/mirp/>`_ is partially created from function, class and method
  documentation, embedded in restructured text files in the ``docs_source`` directory.
