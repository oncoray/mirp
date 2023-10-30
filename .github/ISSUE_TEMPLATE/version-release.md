---
name: Version Release
about: Version release checklist
title: ''
labels: release
assignees: ''

---

- [ ] Update version
  - [ ] `mirp\pyproject.toml`
  - [ ] `mirp\docs_source\source\conf.py`
  - [ ] `mirp\conda\meta.yaml` (if build for conda)
- [ ] Check that all unit tests successfully pass.
- [ ] Update `NEWS.md`.
- [ ] Update `README.md` if required.
- [ ] Check long-form documentation locally.
  - [ ] Open a terminal.
  - [ ] Navigate to `mirp\docs_source`.
  - [ ] Run `make` script: `.\make html`
  - [ ] Check locally build html pages for obvious errors.
- [ ] Copy updated documentation to `mirp\docs`.
  - [ ] Empty the `docs` directory, except for `.nojekyll`. *This prevents documentation files from being processed again by GitHub*
  - [ ] Copy the contents of `mirp\docs_source\build\html` into `mirp\docs`. `index.html` should be in the root of the `docs` folder, as GitHub will use this as the landing page.
- [ ] Package for PyPI
  - [ ] Open a terminal.
  - [ ] Navigate to `mirp` root directory.
  - [ ] Build package: `python -m build`.
  - [ ] Check package content in `mirp\dist\*.tar.gz`.
  - [ ] Upload package to testPyPI:
    - [ ] Run `python -m twine upload --repository testpypi dist\*`
	- [ ] username: `__token__`
	- [ ] password: testpypi API token
  - [ ] Check package on testPyPI:
	- [ ] Check landing page for obvious errors.
	- [ ] Create virtual environment or temporary conda environment
	- [ ] Install mirp: `pip install mirp --index-url https://test.pypi.org/mirp/ --no-deps
  - [ ] Upload package to PyPI:
    - [ ] Run `python -m twine upload dist\*
	- [ ] username: `__token__`
	- [ ] password: pypi API token
- [ ] Merge with main branch.
- [ ] Create release.
  - [ ] Copy `NEWS.md`.
  - [ ] Attach package and wheel files from `mirp\dist` as binary files.
