Tests
=====

MIRP uses tests to ensure that it functions as expected. These tests are located in the `test` directory, and build on
the `pytest framework <https://pytest.org>`_. There are two kinds of tests in MIRP:

* automated tests: these tests are decorated using `@pytest.mark.ci`, and are run as part of continuous integration
  using GitHub actions. As such, each automated test should complete in a few seconds.
* standard tests: these are standard tests that are usually run manually because of longer run times.

If you contribute new code to MIRP, we would heavily recommend also writing tests for this code. This helps ensure
that your code will work as intended. `pytest` has some peculiarities:

* Your the name of your test file needs to start or end with `test`, e.g., `test_awesome_new_code.py` or
  `my_awesome_new_code_test.py`.
* The test itself in the test file should start or end with `test`.
* If your test requires data, please add the data to `test/data`. You can then refer to your data by specifying its path
  relative to the test file. Several existing tests define the current test directory using
  `CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))`.
* If your test exports to a file, you can declare a the temporary directory as part of the test definition:

  .. code-block:: python

     def first_great_test(tmp_path):
        # Your test starts here
        ...

  `pytest` will automatically pick up `tmp_path` and create a separate, unique, temporary directory. This helps your
  test play nicely with `pytest` extensions for parallel test processing such as `pytest-xdist`.

After having written your test, you can run your test using your IDE, or from the overall `mirp` directory
(not the `mirp/mirp` source directory) using the terminal: `python -m pytest test_awesome_new_code.py::first_great_test`.
See the `pytest documentation <https://pytest.org>`_ for more information.

Coverage
--------
If your contribution involves a substantial code contribution, it may be helpful to measure how well your and other
tests cover your code. This can be done using the `coverage.py package <https://coverage.readthedocs.io>`_. If important
parts of your code are not covered by a test, consider write a specific test for that part.
