# seqabpy
![python-package-build](https://github.com/NPodlozhniy/seqabpy/actions/workflows/python-package.yml/badge.svg)
[![python-package-coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/NPodlozhniy/seqabpy/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/NPodlozhniy/seqabpy/blob/python-coverage-comment-action-data/htmlcov/index.html)

Sequential A/B Testing Framework in Python

### CI/CD

With each push to `master` building workflow is triggered that,
besides the build itself, checks linters, applies tests and measures the coverage.

What is more, only if the commit is tagged, PyPI workflow is triggered,
that publishes a package and, in addition, builds GitHub release.

### Getting started

Easy installation via `pip`

```
$ pip install seqabpy
```

### For developers

If you would like to contribute to the project yo can do the following

1. Create a new virtual environment
```
$ python -m venv my_env
$ source my_env/bin/activate
```

2. Copy the repo
```
$ git clone https://github.com/NPodlozhniy/seqabpy.git
```

3. To test the package run
```
$ python -m pip install pytest coverage
$ coverage run --source=src --module pytest --verbose tests && coverage report --show-missing
```

4. Install requirements for developers
```
$ pip install -r requirements_dev.txt
```

5. Make changes and then release version to PyPI
```
$ python -m build
$ python -m twine upload --repository testpypi dist/*
```