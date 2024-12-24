# seqabpy
![python-package-build](https://github.com/NPodlozhniy/seqabpy/actions/workflows/python-package.yml/badge.svg)
[![python-package-coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/NPodlozhniy/seqabpy/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/NPodlozhniy/seqabpy/blob/python-coverage-comment-action-data/htmlcov/index.html)
![PyPI - Version](https://img.shields.io/pypi/v/seqabpy?label=pypi%20version&color=green)

Sequential A/B Testing Framework in Python

### Getting started

Easy installation via `pip`

```
$ pip install seqabpy
```

### Workflow

With each push to `master` building workflow is triggered that,
besides the build itself, checks linters, applies tests and measures the coverage.

What is more, when the tag is pushed, PyPI workflow is triggered,
that publishes a package and, in addition, builds GitHub release.


### Development

If you would like to contribute to the project yo can do the following


1. Copy the repo
```
$ git clone https://github.com/NPodlozhniy/seqabpy.git
```

2. Test the package
```
$ python -m pip install pytest coverage
$ coverage run --source=src --module pytest --verbose tests && coverage report --show-missing
```

3. Install requirements for developers
```
$ pip install -r requirements_dev.txt
```

4. Make changes and then release a new version
```
$ python -m build
$ python -m twine upload --repository testpypi dist/*
```
